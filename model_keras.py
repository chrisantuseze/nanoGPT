import math
import inspect
from dataclasses import dataclass
import tensorflow as tf
import tensorflow.keras.layers as layers


class CausalSelfAttention(layers.Layer):
    def __init__(self, config):
        super(CausalSelfAttention, self).__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = layers.Dense(3 * config.n_embd, use_bias=config.bias)
        self.c_proj = layers.Dense(config.n_embd, use_bias=config.bias)

        self.attn_dropout = layers.Dropout(config.dropout)
        self.resid_dropout = layers.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = self.n_embd//self.n_head
        self.dropout = config.dropout

    def call(self, x):
        B, T, C = x.shape
        # print("x.shape:", x.shape)

        # self.head_size = C // self.n_head

        qkv = self.c_attn(x)
        q, k, v  = tf.split(qkv, 3, axis=-1)

        q = tf.reshape(q, (-1, T, self.n_head, C // self.n_head))
        k = tf.reshape(k, (-1, T, self.n_head, C // self.n_head))
        v = tf.reshape(v, (-1, T, self.n_head, C // self.n_head))

        q = tf.transpose(q, perm=[0,2,1,3])
        k = tf.transpose(k, perm=[0,2,1,3])
        v = tf.transpose(v, perm=[0,2,1,3])

        attn = tf.matmul(q, k, transpose_b=True) * (C // self.n_head ** -0.5)

        # Get the shape of the existing matrix
        attn_shape = tf.shape(attn)

        mask = tf.linalg.band_part(tf.ones(attn_shape), -1, 0) #lower triangular matrix
        attn = tf.where(mask == 1, attn, float('-inf')) #set upper triangular part to -inf
        attn = tf.nn.softmax(attn, axis=-1) #output shape: (B, n_heads, T, T)
        attn = self.attn_dropout(attn) #shape: (B, n_heads, T, head_size)

        out = tf.matmul(attn, v)
        out = tf.transpose(out, perm=[0,2,1,3]) # shape: (B, T, n_heads, head_size)
        out = tf.reshape(out, (-1, T, C)) #shape: (B, T, C)

        out = self.resid_dropout(self.c_proj(out))
        return out
    

class MLP(layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.c_fc = layers.Dense(4 * config.n_embd, use_bias=config.bias, activation=tf.keras.activations.gelu, trainable= True)
        self.c_proj = layers.Dense(config.n_embd, use_bias=config.bias, trainable= True)
        self.dropout = layers.Dropout(config.dropout, trainable= True)

    def call(self, x):
        x = self.c_fc(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    

class Block(layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = layers.LayerNormalization(epsilon=config.epsilon, center=False, scale=True)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = layers.LayerNormalization(epsilon=config.epsilon, center=False, scale=True)
        self.mlp = MLP(config)

    def call(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    epsilon: float = 0.0
    batch_size: int = 0
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    input_file_path: str = ""
    epochs: int = 5
    lr: float = 6e-4


class GPT(layers.Layer):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = {
            # 'input': tf.keras.Input(shape=(config.block_size,)),
            'wte': layers.Embedding(config.vocab_size, config.n_embd, input_length=config.block_size),
            'wpe': layers.Embedding(self.config.block_size, config.n_embd),
            'drop': layers.Dropout(config.dropout),
            'h': [Block(config) for _ in range(config.n_layer)],
            'ln_f': layers.LayerNormalization(epsilon=config.epsilon, center=False, scale=True)
        }
        self.lm_head = layers.Dense(config.vocab_size, use_bias=False)

    def call(self, inputs, targets=None):
        # print("Input shape:", inputs.shape)

        pos = tf.range(0, self.config.block_size, dtype=tf.int64)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer['wte'](inputs) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer['wpe'](pos) # position embeddings of shape (t, n_embd)
        
        # print("tf.shape(tok_emb):", tok_emb.shape)
        # print("tf.shape(pos_emb):", pos_emb.shape)

        x = self.transformer['drop'](tok_emb + pos_emb)
        for block in self.transformer['h']:
            x = block(x)
        x = self.transformer['ln_f'](x)

        # print("x:", x.shape)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x) # note: using list [-1] to preserve the time dim
            loss = None

        return logits#, loss

    def generate_text(self, tokenizer, initial_prompt, generation_length=100):
        generated_text = initial_prompt
        for _ in range(generation_length):
            # Convert the current prompt into model inputs
            input_ids = tokenizer.encode(generated_text)

            # Pad or truncate the input sequence to match the model's expected input length
            if len(input_ids) > self.config.block_size:
                input_ids = input_ids[-self.config.block_size:]
            else:
                padding_length = self.config.block_size - len(input_ids)
                input_ids = [0] * padding_length + input_ids  # Prepend zeros for padding

            input_ids = tf.expand_dims(input_ids, 0)

            # Get predictions for the next token
            logits = self(input_ids)
            predictions = logits[:, -1, :]

            # Sample the output (using tf.random.categorical) to generate token ID
            token_id = tf.random.categorical(predictions, num_samples=1)[0, 0].numpy()

            # Convert token ID back to character
            token = tokenizer.decode([token_id])

            # Append the token to the generated text
            generated_text += token

        return generated_text


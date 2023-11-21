import os
import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as layers

from model_keras import GPT, GPTConfig
from prepare import get_dataset
from text_generator import TextGenerator

def create_model(gptconf):
    inputs = layers.Input(shape=(gptconf.block_size,), dtype=tf.int32)
    gpt = GPT(gptconf)
    outputs = gpt(inputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        "adam", loss=[loss_fn, None],
    )  # No loss and optimization based on word embeddings from transformer block
    return model

def train(config, model, train_dataset, test_dataset, text_gen_callback):
    # Compile and train the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr)#0.001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss_fn)
    history = model.fit(
        train_dataset, 
        epochs=config.epochs, 
        validation_data=test_dataset,
        callbacks=[text_gen_callback]
    )
    # Save the model using model.save
    model.save("nano_GPT_model")

def finetune(config, model, train_dataset, test_dataset, text_gen_callback):
    fine_tune_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # Continue training the model for fine-tuning
    fine_tune_epochs = 1#5

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model.compile(optimizer=fine_tune_optimizer, loss=loss_fn)
    history_fine_tune = model.fit(
        train_dataset,
        epochs=fine_tune_epochs,
        validation_data=test_dataset,
        callbacks=[text_gen_callback, early_stopping]
    )

# def sample_from(logits):
#     logits, indices = tf.math.top_k(logits, k=10, sorted=True)
#     indices = np.asarray(indices).astype("int32")
#     preds = tf.keras.activations.softmax(tf.expand_dims(logits, 0))[0]
#     preds = np.asarray(preds).astype("float32")
#     return np.random.choice(indices, p=preds)

# def detokenize(index_to_word, number):
#     return index_to_word[number]

# # def gen_text(config, model, start_tokens, vocab_train):
# #     start_tokens = [_ for _ in start_tokens]

# #     num_tokens_generated = 0
# #     tokens_generated = []
# #     while num_tokens_generated <= config.block_size:
# #         pad_len = config.block_size - len(start_tokens)
# #         sample_index = len(start_tokens) - 1
# #         if pad_len < 0:
# #             x = start_tokens[:config.block_size]
# #             sample_index = config.block_size - 1
# #         elif pad_len > 0:
# #             x = start_tokens + [0] * pad_len
# #         else:
# #             x = start_tokens
# #         x = np.array([x])
# #         print(x)
# #         y = model(x)
# #         print
# #         sample_token = sample_from(y[0][sample_index])
# #         tokens_generated.append(sample_token)
# #         start_tokens.append(sample_token)
# #         num_tokens_generated = len(tokens_generated)
# #     txt = " ".join(
# #         [detokenize(vocab_train, _) for _ in start_tokens + tokens_generated]
# #     )
# #     print(f"generated text:\n{txt}\n")


def main():
    # data
    input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
    batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size = 40#256 #1024
    vocab_size = 1000#5000  # Limits parameters in model.

    num_tokens_generated = 40#500

    # model
    n_layer = 12
    n_head = 12
    n_embd = 768
    dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    bias = False # do we use bias inside LayerNorm and Linear layers?
    
    # adamw optimizer
    learning_rate = 6e-4 # max learning rate
    
    epochs = 5#10

    # model init
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, 
                      bias=bias, dropout=dropout, batch_size=batch_size, input_file_path=input_file_path,
                      epsilon=1e-6, epochs=epochs, lr=learning_rate, vocab_size=vocab_size)#50304) # start with model_args from command line

    # Create the decoder model
    gptconf = GPTConfig(**model_args)
    model = create_model(gptconf)

    train_dataset, val_dataset, vocab_train, vocab_test = get_dataset(config=gptconf)

    # Tokenize starting prompt
    word_to_index = {}
    for index, word in enumerate(vocab_train):
        word_to_index[word] = index

    start_prompt = "\n"
    start_tokens = [word_to_index.get(_, 1) for _ in start_prompt.split()]
    text_gen_callback = TextGenerator(gptconf, num_tokens_generated, start_tokens, vocab_train)

    train(gptconf, model, train_dataset, val_dataset, text_gen_callback)
    # finetune(gptconf, model, train_dataset, val_dataset, text_gen_callback)

    # # Load the saved model
    # loaded_model = tf.keras.models.load_model("nano_GPT_model")

    # gen_text(gptconf, loaded_model, start_tokens=start_tokens, vocab_train=vocab_train)


if __name__ == "__main__":
    main()

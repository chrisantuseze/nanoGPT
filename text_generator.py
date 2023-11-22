import tensorflow as tf
import numpy as np

class TextGenerator(tf.keras.callbacks.Callback):
    """A callback to generate text from a trained model.
    1. Feed some starting prompt to the model
    2. Predict probabilities for the next token
    3. Sample the next token and add it to the next input

    Arguments:
        max_tokens: Integer, the number of tokens to be generated after prompt.
        start_tokens: List of integers, the token indices for the starting prompt.
        index_to_word: List of strings, obtained from the TextVectorization layer.
        top_k: Integer, sample from the `top_k` token predictions.
        print_every: Integer, print after this many epochs.
    """

    def __init__(self, config, start_prompt, vectorizer, max_new_tokens, top_k=10, print_every=1):
        self.config = config
        
        self.start_prompt = start_prompt
        self.vectorizer = vectorizer

        self.max_new_tokens = max_new_tokens

        self.print_every = print_every
        self.k = top_k

    def sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = tf.keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def detokenize(self, number):
        vocabulary = self.vectorizer.get_vocabulary()
        if len(vocabulary) <= number or vocabulary[number] == '[UNK]':
            return ""
        
        return vocabulary[number]
    
    def on_epoch_end_(self, epoch, logs=None):
        if (epoch + 1) % self.print_every != 0:
            return

        # Vectorize the start string
        input_eval = self.vectorizer([self.start_prompt])[:, :-1]  # This will be a 2D tensor with shape (1, seq_length)

        if input_eval.shape[1] < self.config.block_size:
            input_eval = tf.pad(input_eval, [[0, 0], [0, self.config.block_size - tf.shape(input_eval)[1]]])
            print("input_eval.shape:", input_eval.shape)

        # Empty string to store our results
        text_generated = []

        # Reset states before processing a new sequence
        self.model.reset_states()
        for _ in range(self.max_new_tokens):
            predictions = self.model(input_eval)
            # Remove the batch dimension
            predictions = tf.squeeze(predictions, 0)

            # Use a categorical distribution to predict the character returned by the model
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

            # Append the predicted character to the generated text
            text_generated.append(self.detokenize(predicted_id))

            # The predicted ID is used as the next input to the model
            input_eval = tf.expand_dims([predicted_id], 0)

        txt = self.start_prompt + ' '.join(text_generated)
        print(f"\nGenerated text:\n{txt}\n")

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.print_every != 0:
            return
        
        # Tokenize starting prompt
        word_to_index = {}
        for index, word in enumerate(self.vectorizer.get_vocabulary()):
            word_to_index[word] = index
        self.start_tokens = [word_to_index.get(_, 1) for _ in self.start_prompt.split()]
        
        start_tokens = [_ for _ in self.start_tokens]
        tokens_generated = []

        # Reset states before processing a new sequence
        self.model.reset_states()
        for _ in range(self.max_new_tokens):
            pad_len = self.config.block_size - len(start_tokens)
            sample_index = len(start_tokens) - 1

            if pad_len < 0:
                x = start_tokens[:self.config.block_size]
                sample_index = self.config.block_size - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens

            x = np.array([x])
            y = self.model.predict(x)

            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)

        txt = " ".join([self.detokenize(_) for _ in self.start_tokens + tokens_generated])
        print(f"\nGenerated text:\n{txt}\n")

    def generate_text(self, model):
        self.model = model
        self.on_epoch_end(epoch=1)
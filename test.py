import tensorflow as tf
import os

# Sample text data
# text_data = ["This is an example sentence.", "Another sentence for testing."]

input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
with open(input_file_path, 'r') as f:
    text_data = f.read()

text_data = text_data[:100]
# Remove empty lines and strip whitespace from each line
# text_data = [line.strip() for line in text_data if line.strip()]

# Create a TextVectorization layer
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=1000,  # Adjust this based on your data and vocabulary size
    output_mode='int',  # You can use 'int' or 'binary' depending on your needs
    output_sequence_length=None  # None means variable sequence length
)

# Adapt the TextVectorization layer to your text data
vectorize_layer.adapt([text_data])

# Tokenize the text using the TextVectorization layer
tokenized_data = vectorize_layer(text_data)

############################################################################
print("Original text:\n", "".join(text_data))

print("Numerical data:\n", tokenized_data)

# Get the vocabulary
vocabulary = vectorize_layer.get_vocabulary()
print("Vocabulary:\n", vocabulary)

# Reconvert numerical vectors to words
reconstructed_text = []
for vector in tokenized_data.numpy():
    words = vocabulary[vector]
    reconstructed_text.append(" ".join(words))

print("Reconstructed text:\n", reconstructed_text)
############################################################################

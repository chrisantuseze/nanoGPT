import os
import requests
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

def load_and_split_data(config):
    if not os.path.exists(config.input_file_path):
        data_url = 'https://gutenberg.org/cache/epub/4099/pg4099.txt' #The Angel in the House by Coventry Patmore
        with open(config.input_file_path, 'w') as f:
            f.write(requests.get(data_url).text)

    with open(config.input_file_path, 'r') as f:
        data = f.read()

    if len(data) == 0:
        raise Exception("File is empty")

    n = len(data)
    train = data[:int(n*0.9)]
    val = data[int(n*0.9):]

    return train, val

def prepare_dataset(config, data):
    # Create a vectorization layer and adapt it to the text
    vectorizer = TextVectorization(
        max_tokens=config.vocab_size - 1,
        output_mode="int",
        output_sequence_length=config.block_size + 1,
    )

    # Adapt the layer to the data
    vectorizer.adapt([data])

    """
    Shift word sequences by 1 position so that the target for position (i) is
    word at position (i+1). The model will use all words up till position (i)
    to predict the next word.
    """
    text = tf.expand_dims(data, -1)
    tokenized_sentences = vectorizer(text)
    # print("tokenized_sentences.shape:", tokenized_sentences.shape)

    x = tokenized_sentences[:, :-1]
    # print("x.shape:", x.shape)

    y = tokenized_sentences[:, 1:]
    # print("y.shape:", y.shape)

    # Create a tf.data.Dataset from the vectorized texts and labels
    dataset = tf.data.Dataset.from_tensor_slices((x, y))

    # Batch the dataset
    dataset = dataset.batch(config.batch_size)

    # Optionally shuffle the dataset
    dataset = dataset.shuffle(buffer_size=len(data))

    # # Iterate over the batches for training
    # for batch in dataset.take(1):
    #     inputs, targets = batch
    #     print(inputs, targets)
    #     # Your training step goes here

    return dataset, vectorizer

def get_dataset(config):
    train, val = load_and_split_data(config)

    train_data, train_vectorizer = prepare_dataset(config, train)
    val_data, val_vectorizer = prepare_dataset(config, val)


    # ############################################################################
    # print("Original text:\n", "".join(train_data))

    # # Tokenize the text using the TextVectorization layer
    # tokenized_data = train_vectorizer(train_data)

    # print("Numerical data:\n", tokenized_data)

    # # Get the vocabulary
    # vocabulary = train_vectorizer.get_vocabulary()
    # print("Vocabulary:\n", vocabulary)

    # # Reconvert numerical vectors to words
    # reconstructed_text = []
    # for vector in tokenized_data.numpy():
    #     words = vocabulary[vector]
    #     reconstructed_text.append(" ".join(words))

    # print("Reconstructed text:\n", reconstructed_text)
    # ############################################################################


    # Print the number of batches in each dataset
    print(f"Number of batches in training dataset: {len(train_data)}")
    print(f"Number of batches in validation dataset: {len(val_data)}")
    print("Length of vocab:", len(train_vectorizer.get_vocabulary()))


    return train_data, val_data, train_vectorizer, val_vectorizer
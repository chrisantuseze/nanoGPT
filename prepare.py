import os
import requests
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

def prepare_lm_inputs_labels(text, vectorize_layer):
    """
    Shift word sequences by 1 position so that the target for position (i) is
    word at position (i+1). The model will use all words up till position (i)
    to predict the next word.
    """
    text = tf.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y

def get_vectorizer_ds(ds, config):
    # Create a vectorization layer and adapt it to the text
    vectorize_layer = TextVectorization(
        max_tokens=config.vocab_size - 1,
        output_mode="int",
        output_sequence_length=config.block_size + 1,
    )

    # Adapt the layer to the data
    vectorize_layer.adapt(ds)
    vocab = vectorize_layer.get_vocabulary()  # To get words back from token indices

    return vocab, prepare_lm_inputs_labels(ds, vectorize_layer)

def batch_ds(batch_size, vectorized_texts, vectorized_labels, raw_texts):
    # Create a tf.data.Dataset from the vectorized texts and labels
    dataset = tf.data.Dataset.from_tensor_slices((vectorized_texts, vectorized_labels))

    # Batch the dataset
    dataset = dataset.batch(batch_size)

    # Optionally shuffle the dataset
    dataset = dataset.shuffle(buffer_size=len(raw_texts))

    return dataset

def get_dataset(config):
    if not os.path.exists(config.input_file_path):
        data_url = 'https://gutenberg.org/cache/epub/72132/pg72132.txt'
        with open(config.input_file_path, 'w') as f:
            f.write(requests.get(data_url).text)

    with open(config.input_file_path, 'r') as f:
        data = f.read()
    n = len(data)
    train_list = data[:int(n*0.9)]
    val_list = data[int(n*0.9):]

    train_list = list(train_list)
    val_list = list(val_list)
    # print(type(train_list))

    # Tokenize your text using the TextVectorization layer
    vocab_train, (vectorized_train, vectorized_train_label) = get_vectorizer_ds(train_list, config)
    # train_data = train_data.prefetch(tf.data.AUTOTUNE)
    train_data = batch_ds(config.batch_size, vectorized_train, vectorized_train_label, train_list)

    vocab_test, (vectorized_val, vectorized_val_label) = get_vectorizer_ds(val_list, config)
    val_data = batch_ds(config.batch_size, vectorized_val, vectorized_val_label, val_list)

    # val_data = get_vectorizer(val_list, config)
    # val_data = val_data.map(prepare_lm_inputs_labels)
    # val_data = val_data.prefetch(tf.data.AUTOTUNE)

    # Print the number of batches in each dataset
    print(f"Number of batches in training dataset: {len(train_data)}")
    print(f"Number of batches in validation dataset: {len(val_data)}")

    # # Iterate over the batches for training
    # for batch in train_data.take(1):
    #     inputs, targets = batch
    #     print(inputs, targets)
    #     # Your training step goes here

    # # Iterate over the batches for validation
    # for batch in val_data.take(1):
    #     inputs, targets = batch
    #     print(inputs, targets)
    #     # Your validation step goes here

    return train_data, val_data, vocab_train, vocab_test
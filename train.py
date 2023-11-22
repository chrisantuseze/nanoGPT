import os
import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as layers

from model import GPT, GPTConfig
from prepare import get_dataset
from text_generator import TextGenerator

def create_model(gptconf):
    inputs = layers.Input(shape=(gptconf.block_size,), dtype=tf.int32)
    gpt = GPT(gptconf)
    outputs = gpt(inputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def train(config, model, train_dataset, test_dataset, base_path, text_gen_callback):
    # Compile and train the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss_fn)
    model.fit(
        train_dataset,
        epochs=config.epochs,
        validation_data=test_dataset,
        callbacks=[text_gen_callback]
    )
    # Save the model using model.save
    model.save(base_path + "/model")

def finetune(config, model, train_dataset, test_dataset, text_gen_callback):
    fine_tune_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # Continue training the model for fine-tuning
    fine_tune_epochs = 1#5

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model.compile(optimizer=fine_tune_optimizer, loss=loss_fn)
    model.fit(
        train_dataset,
        epochs=fine_tune_epochs,
        validation_data=test_dataset,
        callbacks=[text_gen_callback, early_stopping]
    )
    
def main():
    # data
    base_path = os.path.dirname(__file__)
    # base_path = "/content/drive/MyDrive/OSU/Academic-Resource/TA/ML-23/Assign4"
    input_file_path = base_path + '/input.txt'
    batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size = 512#1024
    vocab_size = 3383 #50304  # Limits parameters in model.

    max_new_tokens = 100

    # model
    n_layer = 8
    n_head = 8
    n_embd = 768
    dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    bias = False # do we use bias inside LayerNorm and Linear layers?

    learning_rate = 1e-43
    epochs = 20

    # model init
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                      bias=bias, dropout=dropout, batch_size=batch_size, input_file_path=input_file_path,
                      epsilon=1e-6, epochs=epochs, lr=learning_rate, vocab_size=vocab_size) # start with model_args from command line

    # Create the decoder model
    gptconf = GPTConfig(**model_args)

    train_dataset, val_dataset, train_vectorizer, val_vectorizer = get_dataset(config=gptconf)

    text_gen_callback = TextGenerator(gptconf, start_prompt="\n", vectorizer=train_vectorizer, max_new_tokens=max_new_tokens)

    # Load the saved model
    model_path = "" #base_path + "/model" #TODO Uncomment this to load from saved model if it exists
    if os.path.exists(model_path):
        print(f"Saved model exists. It is is {model_path} \nLoading model for text generation.")
        model = tf.keras.models.load_model(model_path)

        text_gen_callback.generate_text(model)
    else:
        print("No saved model. Training a model for text generation.")

        model = create_model(gptconf)
        train(gptconf, model, train_dataset, val_dataset, base_path, text_gen_callback)
        # finetune(gptconf, model, train_dataset, val_dataset, text_gen_callback)

if __name__ == "__main__":
    main()

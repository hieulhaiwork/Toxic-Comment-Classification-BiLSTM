import os
from dotenv import load_dotenv
load_dotenv()

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def tokenize_words(train_comments):

    # Create a tokenize, which transforms a sentence to a list of ids by turning text into sequences of integers (where each integer represents the index of a word in a dictionary).
    tokenizer = Tokenizer(num_words=os.getenv('MAX_NB_WORDS'))

    # Create a word index based on the frequency of words in the provided text corpus.
    tokenizer.fit_on_texts(train_comments)

    # Transform training/testing sentences to training/testing sequences using the word index created by fit_on_texts.
    train_sequences = tokenizer.texts_to_sequences(train_comments)

    # Ensure that all sequences in a list have the same length by padding shorter sequences and truncating longer ones.
    train_data = pad_sequences(train_sequences, maxlen=os.getenv('MAX_SEQUENCE_LENGTH'))

    return train_data
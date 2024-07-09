import os
from dotenv import load_dotenv
import numpy as np
import tensorflow as tf
from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, BatchNormalization, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
load_dotenv()

MAX_SEQUENCE_LENGTH = os.getenv('MAX_SEQUENCE_LENGTH')
MAX_TOKENS = os.getenv('MAX_TOKENS')
EMBEDDING_DIM = os.getenv('EMBEDDING_DIM')

def get_text_lstm():
    input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,))
    embedding_layer = Embedding(MAX_TOKENS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)(input_layer)

    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(embedding_layer)
    x = BatchNormalization()(x)
    x = Bidirectional(LSTM(32, return_sequences=False, dropout=0.3, recurrent_dropout=0.3))(x) #
    x = BatchNormalization()(x)
    
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation="relu")(x)

    output_layer = Dense(7, activation="softmax")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def train_model_single_fold(X, y, batch_size, get_model_func):
    # Initialize callbacks for early stopping and model checkpointing
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    #model_checkpoint = ModelCheckpoint('model/FinalModel.h5', save_best_only=True, save_weights_only=True)

    # Assuming a simple 80-20 train-validation split
    split_idx = int(len(X) * 0.8)
    train_x, val_x = X[:split_idx], X[split_idx:]
    train_y, val_y = y[:split_idx], y[split_idx:]

    # Get the model and ensure it's compiled with the necessary metrics
    model = get_model_func()  # Ensure your model inside this function is compiled with accuracy, Precision(), and Recall() metrics

    # Train the model on the dataset
    model.fit(train_x, train_y,
              validation_data=(val_x, val_y),
              epochs=20, batch_size=batch_size, shuffle=True,
              callbacks=[early_stopping])
    
    model.save('model/FinalModel.h5')

    return model
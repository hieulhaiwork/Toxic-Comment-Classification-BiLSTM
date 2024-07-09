from embedding import *
from tensorflow.keras.models import load_model

model = load_model('model/FinalModel.h5')

CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate","no_threat"]

def preprocess_input_sentence(sentence, tokenizer, max_len=100):
    #Converts a sentence to a sequence of tokens padded to a maximum length.
    seq = tokenizer.texts_to_sequences([sentence])
    padded_seq = pad_sequences(seq, maxlen=max_len)
    return padded_seq

def predict_toxicity(sentence, model, tokenizer, classes):
    #Predicts the toxicity levels of an input sentence.
    preprocessed_sentence = preprocess_input_sentence(sentence, tokenizer, max_len=100)
    prediction = model.predict(preprocessed_sentence)
    return dict(zip(classes, prediction[0]))

def classify_toxicity(sentence):
    predictions = predict_toxicity(sentence, model, tokenizer, CLASSES)
    return [predictions[class_name] for class_name in CLASSES]
import os
import numpy as np
import pandas as pd
from preprocess import clean_text
from embedding import tokenize_words
from model import get_text_lstm, train_model_single_fold

# Load data
train_df = pd.read_csv('dataset/train.csv')

# Since there are sentences with no threats (i.e., the labels of all 6 attributes are zero), we add an additional field 'no_threat' with a value of 1 to ensure the model learns accurately
# Choose number columns
numeric_cols = train_df.select_dtypes(include='number').columns

# Create new attribute with condition: sum of all other number columns equals to 0
train_df['no_threat'] = train_df[numeric_cols].apply(lambda row: 1 if row.sum() == 0 else 0, axis=1)

# Divide train data and train labels
list_sentences_train = train_df["comment_text"].fillna("no comment").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate","no_threat"]
train_labels = train_df[list_classes].values

# Preprocess the sentences in train data
train_comments = [clean_text(text) for text in list_sentences_train]

# View a sentence after preprocessing
for i in range(1):
    print("Cleaned\n", train_comments[i] + '\n')
    print("Raw\n", train_df.iloc[i]['comment_text'] + '\n')
    print("------------------")

# Embedding words
train_data = tokenize_words(train_comments)

# Train model 
model = train_model_single_fold(train_data, train_labels, batch_size=256, get_model_func=get_text_lstm)





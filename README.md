<br />
<div align="center">

<h1 align="center">Toxic Comment Classification using BiLSTM</h1>

  <p align="center">
    This is my personal project to create a model to predict a sentence toxic or not. 
    <br />
    <br />
    <a href="https://drive.google.com/file/d/1_7ovjM1mcIwo-BQ1TiK9G0t7g9HdaZXx/view?usp=drive_link">View Demo</a>
  </p>
</div>

## About The Project

In this project, I use dataset from <a href="https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge">Toxic Comment Classification Challenge</a> from Kaggle.

I have applied preprocessing libraries such as regex and nltk to generate clean training data for the model. The BiLSTM model generally meets some requirements for classifying toxic comments.

I also attempted to apply BERT, but unfortunately, my hardware isn't sufficient to handle such large datasets.

## Limitations and Future works

This model is unable to differentiate toxic comments based on context, primarily relying on individual words and surrounding language.

Future improvements:

- Implement pretrained deep learning models like BERT, GPT for contextual classification.
- Try cluster-based word embeddings.
- Utilize embedding models such as BERTTokenize, CohereTokenize, etc., for better contextual representation of language.

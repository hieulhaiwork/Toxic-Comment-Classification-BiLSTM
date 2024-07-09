import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

special_character_removal=re.compile(r'[^a-z\d ]',re.IGNORECASE)
replace_numbers=re.compile(r'\d+',re.IGNORECASE)

def clean_text(text, stem_words=False):
    # Normalize text to lowercase
    text = text.lower()

    # Replace contractions and abbreviations
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r"\'s", " ", text)

    # Normalize spacing and punctuations
    text = re.sub('\n', ' \n ', text)
    text = re.sub(r'([a-zA-Z]+)([/!?.])([a-zA-Z]+)', r'\1 \2 \3', text)
    text = re.sub(r'([*!?\'])\1\1{2,}', r'\1\1\1', text)
    text = re.sub(r'([*!?\']+)', r' \1 ', text)
    text = re.sub(r'([a-zA-Z])\1{2,}\b', r'\1\1', text)
    text = re.sub(r'([a-zA-Z])\1\1{2,}\B', r'\1\1\1', text)
    text = re.sub(r',', " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"\s{2,}", " ", text)

    # Remove numbers
    text = replace_numbers.sub('', text)

    # Remove special characters
    text = special_character_removal.sub('', text)
    
    # Convert sentences to semantic tokens
    text = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    stop_words_rm = [word for word in text if word not in stop_words]
    
    # Convert words to their origin form
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in stop_words_rm]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmas_words = [lemmatizer.lemmatize(word, pos='v') for word in stemmed_words]
    
    cleaned_text = ' '.join(lemmas_words)
    
    return cleaned_text

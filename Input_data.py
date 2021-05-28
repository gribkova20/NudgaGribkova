# -*- coding: utf-8 -*-

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils
from Text_processing import text
import pandas as pd

"""Created on 10.05.2021

@author: Nikita

The module is designed to prepare input data for a neural network. The dataset consists of more than 1 million news 
articles that were taken from an open source.There are 4 topics in total:
                                                                         1) World politics
                                                                         2) Sports
                                                                         3) Business
                                                                         4) Sci/Tech 
"""

# Maximum number of words.
max_words = 5000
# Maximum news length.
max_len = 400
# Number of news classes.
nb_classes = 4

# Load data from open source.
news = pd.read_csv(r'C:\PythonProjects\Jobs\Eng_version\Datasets\train.csv',
                   header=None,
                   names=['class', 'title', 'text'])
training_news = news['text']
news.pop('title')

# Preprocessing of articles.
text = [' '.join(item) for item in text(training_news, english=True)]
df = pd.DataFrame(text, columns=['new_text'])
df = pd.concat([news, df], axis=1)


def tokenizer():
    tokenizers = Tokenizer(num_words=max_words)
    tokenizers.fit_on_texts(df['new_text'])
    return tokenizers


def input_data():
    tokenizers = tokenizer()
    sequences = tokenizers.texts_to_sequences(df.new_text)

    # Input data.
    x = pad_sequences(sequences, maxlen=max_len)
    y = utils.to_categorical(news['class'] - 1, nb_classes)

    # Divide the train data into training and validation data sets.
    x_train, X_test, y_train, Y_test = train_test_split(x,
                                                        y,
                                                        train_size=0.7,
                                                        random_state=42,
                                                        stratify=y)
    return x_train, X_test, y_train, Y_test

# -*- coding: utf-8 -*-

from tensorflow.keras.preprocessing.text import text_to_word_sequence
from nltk.corpus import stopwords
import pymorphy2

"""Created on 15.03.2021

@author: Nikita

This is code designed for processing rus and eng text. The code has two methods, the first is a static method and is 
used to add additional Russian stopwords to the stopwords bag. The second method processes the text, removes stopwords, 
punctuation in sentences, and reduces all words to lowercase and to a single (initial) form. This module outputs the 
result in the form of word sequences.
"""


# Add russian stopwords
def rus_stopwords():
    russian_stopwords = stopwords.words('russian')
    with open(r'C:\PythonProjects\Neuro_net\Stopwords.txt', encoding='utf-8') as f:
        words = f.read()
        words = words.split(" ")
        for word in words:
            russian_stopwords.append(word)
    return russian_stopwords


# Preprocessing text
def text(text_list: list, del_stopwords: bool = True, russian: bool = False, english: bool = False):
    token_sequence, new_text = [], []
    for line in text_list:
        tokens = text_to_word_sequence(line, filters='!"#$%&amp;()*+,-./:;&lt;=>?@[\\]^_`{|}~\t\n\ufeff',
                                       lower=True, split=' ')
        token_sequence.append(tokens)
    if russian:
        morph = pymorphy2.MorphAnalyzer()
        for words in token_sequence:
            words = [w for w in words if w.isalpha()]  # del tokens, that are not symbols
            if del_stopwords:
                words = [w for w in words if w not in rus_stopwords()]  # del stopwords from the text
            words = [w for w in words if
                     w == 'чс' or len(list(w)) > 3]  # del words consisting of 1-3 letters
            words = [morph.parse(w)[0].normal_form for w in words]  # bringing the words to a single(normal) form
            new_text.append(words)
    if english:
        english_stopwords = stopwords.words('english')
        for words in token_sequence:
            words = [w for w in words if w.isalpha()]  # del tokens, that are not symbols
            if del_stopwords:
                words = [w for w in words if w not in english_stopwords]  # del stopwords from the text
            words = [w for w in words if len(list(w)) > 3]  # del words consisting of 1-3 letters
            new_text.append(words)
    return new_text

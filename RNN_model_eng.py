# -*- coding: utf-8 -*-

from tensorflow.keras.layers import Dense, Embedding,  LSTM, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from Input_data import input_data, tokenizer
import matplotlib.pyplot as plt
import numpy as np

"""Created on 14.05.2021

@author: Nikita

This module presents the architecture of the neural network model. This module is the main project file. The RNN model 
predicts the topic of the text. There are 4 topics in total:
1) World politics
2) Sports
3) Business
4) Sci/Tech
The accuracy of the model is 0.85. If you need to re-train the neural network and output error graphs, then you need 
to call the show_model() function. If you need to predict the topic of the text, then call the prediction() function.
"""


# Maximum number of words.
max_words = 5000
# Maximum news length.
max_len = 400
# Number of news classes.
nb_classes = 4

x_train, X_test, y_train, Y_test = input_data()


def models():
    model = Sequential()
    model.add(Embedding(max_words, 256, input_length=max_len))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(4))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


model = models()


def train_model():
    model_lstm_save_path = 'weights.h5'
    checkpoint_callback_lstm = ModelCheckpoint(model_lstm_save_path,
                                               monitor='val_accuracy',
                                               save_best_only=True,
                                               verbose=1)
    history_lstm = model.fit(x_train,
                             y_train,
                             epochs=3,
                             batch_size=200,
                             validation_data=(X_test, Y_test),
                             callbacks=[checkpoint_callback_lstm])
    return history_lstm


def show_model():
    history_lstm = train_model()
    plt.plot(history_lstm.history['accuracy'],
             label='The number of correct answers in the percentage on the training set')
    plt.plot(history_lstm.history['val_accuracy'],
             label='The number of correct answers in the percentage on the test set')
    plt.xlabel('The Age of Learning')
    plt.ylabel('Percentage of correct predictions')
    plt.legend()
    plt.show()


def text_prediction():
    model.load_weights('weights.h5')
    input_text = input("Enter the text: ")
    tokenizers = tokenizer()
    test_sequences = tokenizers.texts_to_sequences([input_text])
    x = pad_sequences(test_sequences, maxlen=max_len)
    result = model.predict(x)
    res = np.argmax(result) + 1
    if res == 1:
        print('Neural Network class: World politics')
    elif res == 2:
        print('Neural Network class: Sports')
    elif res == 3:
        print('Neural Network class: Business')
    else:
        print('Neural Network class: Sci/Tech')


# If const_prediction=False, then the function triggers a single topic prediction for the text or
# Const_prediction=True, then there will be infinite texts predictions.
def prediction(const_prediction: bool = False):
    text_prediction()
    while const_prediction:
        text_prediction()


if __name__ == "__main__":
    prediction(const_prediction=True)

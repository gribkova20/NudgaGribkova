## Text Classification

The *purpose* of this repository is to create a deep learning neural network model for classifying texts on 4 topics: 

*World politics*, *Sports*, *Business*, *Sci/Tech*.

---
The input data was a sample consisting of more than 1 million news articles that were taken from an open source. The data was divided into 3 data sets: training, validation, and test datasets. The purpose of the model was to recognize the subject of the submitted text. The model was evaluated using the "*accuracy*" metric, and the model's accuracy was 85%. The model is at the stage of optimization and selection of parameters for better accuracy.


## Usage
1. The model is located in `RNN_model_eng.py.`
2. Run python `RNN_model_eng.py` to predict the topic of a text, if you need to train the model, you need to call the function `show_model()`.

## Components of the model

The block contains a brief excerpt of the project files, a more detailed description is located inside each module.


`RNN_model_eng.py` - The module is designed to predict the topic of a text.

`Input_data.py` - The module is designed to prepare input data for a neural network.

`Text_processing.py` - This is a code designed for processing text in Russian and English.

`"weight. h5"` - Coefficients of the trained neural network.

`"train.csv"` - News articles for the training dataset.

`"test.csv"` - News articles for the test dataset (To check how the neural network was trained).


## Version

Python 3.8

Tensorflow 2.4.1

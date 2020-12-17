# NLP Project - To Be Or Not To Be

In this project we displayed the use of Recurrent neural network architectures for text generation. We compare the three main architectures SimpleRNN, GRU, LSTM along with the following three optimizers Adam, RMSProp and SGD. The main goal of this project is to examine the efficiency of a recurrent neural network to produce a text by being trained on dataset of words and punctuation rather than by characters.

## Data Preprocessing 
The data set that has been used is the dataset provided by keras included a collection of Shakespear's plays. We process the data in the script data_processing.py. The procedure that have been followed includes the tokenization of words and punctuation of the text. We define the vocabualry and enumerate the it. Then we transform the text into an array of integers. We use the nltk library for that purpose and a fucntion we created as the word_tokenize module of the nltk library does not recognize the escape sequence "\n" indicating a new line for the forthcoming text. 

Then we make sequences of 35 words and punctuation and we feed to the model the first 34 elements of each sequence and we divide the dataset of sequencies so that 60% would be the training set, 20% be the validation set and 20% be the test set.

## Architecture Comparison
To compare the different architectures we have trained three models with three layers. We use the sequential model of keras. First we add and embedding layer to the model of dimension 200, which is followed by a RNN layer of different architecture each time and then an Dense layer of dimension eqaul to the size of the vocabulary as an output layer with a softmax activation function. Below we give a description of the each model:

### SimpleRNN: 
    Model: "sequential_2"
    ________________________________________________________________________________
    Layer (type)                        Output Shape                    Param #     
    ================================================================================
    embedding_2 (Embedding)             (20, None, 200)                 2842800     
    ________________________________________________________________________________
    simple_rnn_2 (SimpleRNN)            (20, None, 600)                 480600      
    ________________________________________________________________________________
    dense_2 (Dense)                     (20, None, 14214)               8542614     
    ================================================================================
    Total params: 11,866,014
    Trainable params: 11,866,014
    Non-trainable params: 0
    ________________________________________________________________________________

### GRU:
    Model: "sequential_8"
    ________________________________________________________________________________
    Layer (type)                        Output Shape                    Param #     
    ================================================================================
    embedding_8 (Embedding)             (20, None, 200)                 2842800     
    ________________________________________________________________________________
    gru_2 (GRU)                         (20, None, 600)                 1443600     
    ________________________________________________________________________________
    dense_8 (Dense)                     (20, None, 14214)               8542614     
    ================================================================================
    Total params: 12,829,014
    Trainable params: 12,829,014
    Non-trainable params: 0
    ________________________________________________________________________________

### LSTM:
    Model: "sequential_5"
    ________________________________________________________________________________
    Layer (type)                        Output Shape                    Param #     
    ================================================================================
    embedding_5 (Embedding)             (20, None, 200)                 2842800     
    ________________________________________________________________________________
    lstm_2 (LSTM)                       (20, None, 600)                 1922400     
    ________________________________________________________________________________
    dense_5 (Dense)                     (20, None, 14214)               8542614     
    ================================================================================
    Total params: 13,307,814
    Trainable params: 13,307,814
    Non-trainable params: 0
    ________________________________________________________________________________
###
We build the models via the build_model.py script which also exports the above information into a text file for future reference as well as a graph of the model in folder named summary_graph.

To compare the models; we train them each one with the following three optimizers Adam, RMSProp, SGD. In the script build_models.py there is a fucntion to help us set some callbacks in order to monitor the training procedure and saves each models' weights. We compare the models through means of sparse categorical crossentropy for measuring the loss and sparse categorical accuracy for accuracy. We train each pair for 30 epochs.

## Results
The previous comparison, anyone can refer to the plots, shows that the models with the GRU architecture trained wiht the Adam optimizer scores the highest validation accuracy score around 0.63. 
We decided to evaluate the performance of the GRU model by adding another GRU layer which performed purely.
At the end we chose the GRU model with one layer to be our model for text generation. We have trained with the adam oprimizer for 100 epochs.

For space reasons I have not uploaded the the callbacks of the models which I intend to do in the future.

## Predicitons
The script text_generator.py the previous trained model is applied with a given input.

## Future Work
Due to limited time, in the future would some more experiments to be conducted for a better parameter tuning.
Also the previous models where trained to predict just one word to be followed in a giver string and not more. This lead to an somewhat inaccurate text generation as the more words are predicted the less the sentences make sense so in future work we could try to talke that problem also.
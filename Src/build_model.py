import tensorflow as tf
from __main__ import *
import datetime
import time
import os

def summary_graph(model,index,hidden_layers, date_time):

    architectures = ["SimpleRNN","GRU","LSTM"]

    try:
        os.mkdir("Src/models/{}_{}".format(architectures[index],hidden_layers))
        try:
            os.mkdir("Src/models/{}_{}/{}".format(architectures[index],hidden_layers,date_time))
            try:
                os.mkdir("Src/models/{}_{}/{}/summary_graph".format(architectures[index],hidden_layers,date_time))
            except:
                pass
        except:
            pass
    except:
        pass

    file_path = "Src/models/{}_{}/{}/summary_graph".format(architectures[index],hidden_layers,date_time)

    tf.keras.utils.plot_model(model=model,
                            to_file=os.path.join(file_path,"graph.png"),
                            show_shapes=False,
                            show_layer_names=True,
                            rankdir='TB',
                            expand_nested=False,
                            dpi=96)

    with open(os.path.join(file_path,"report.txt"),'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))


def set_callbacks(architecture,hidden_layers, date_time,optimizer):

    try:
        os.mkdir("Src/models/{}_{}/{}/{}_history".format(architecture,hidden_layers,date_time,optimizer))
    except:
        pass

    history ="Src/models/{}_{}/{}/{}_history".format(architecture,hidden_layers,date_time,optimizer)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=history, histogram_freq=1)

    try:
        os.mkdir("Src/models/{}_{}/{}/{}_checkpoint".format(architecture,hidden_layers,date_time,optimizer))
    except:
        pass
    checkpoint_path = "Src/models/{0}_{1}/{2}/{3}_checkpoint/{3}_checkpoint".format(architecture,hidden_layers,date_time,optimizer)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    return [callback, tensorboard_callback, cp_callback]

def SimpleRNN(vocab_size,embedding_dim,hidden_layers,rnn_units,batch_size,optimizer,loss,metrics,drop=0.0,rec_drop=0.0):

    index = 0

    if hidden_layers <= 0 :
        print("Invalide hidden layers number!")

    else:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                    batch_input_shape=[batch_size, None]))
        for _ in range(hidden_layers):
            model.add(tf.keras.layers.SimpleRNN(units = rnn_units,
                                            return_sequences=True,
                                            dropout=drop,
                                            recurrent_dropout=rec_drop,
                                            stateful=True,
                                            recurrent_initializer='glorot_uniform'))
        model.add(tf.keras.layers.Dense(vocab_size, activation='softmax'))

        model.compile(optimizer=optimizer, 
                        loss=loss,
                        metrics=metrics)

        summary_graph(model=model,index=index,hidden_layers=hidden_layers,date_time=date_time)
               
        return model

def GRU(vocab_size,embedding_dim,hidden_layers,rnn_units,batch_size,optimizer,loss,metrics,drop=0.0,rec_drop=0.0):

    index = 1

    if hidden_layers <= 0 :
        print("Invalide hidden layers number!")

    else:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                    batch_input_shape=[batch_size, None]))
        for _ in range(hidden_layers):
           model.add(tf.keras.layers.GRU(units = rnn_units,
                                            return_sequences=True,
                                            dropout=drop,
                                            recurrent_dropout=rec_drop,
                                            stateful=True,
                                            recurrent_initializer='glorot_uniform'))
        model.add(tf.keras.layers.Dense(vocab_size, activation='softmax'))

        model.compile(optimizer=optimizer, 
                        loss=loss,
                        metrics=metrics)

        summary_graph(model=model,index=index,hidden_layers=hidden_layers,date_time=date_time)
               
        return model

def LSTM(vocab_size,embedding_dim,hidden_layers,rnn_units,batch_size,optimizer,loss,metrics,drop=0.0,rec_drop=0.0):

    index = 2

    if hidden_layers <= 0 :
        print("Invalide hidden layers number!")

    else:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                    batch_input_shape=[batch_size, None]))
        for _ in range(hidden_layers):
           model.add(tf.keras.layers.LSTM(units = rnn_units,
                                            return_sequences=True,
                                            dropout=drop,
                                            recurrent_dropout=rec_drop,
                                            stateful=True,
                                            recurrent_initializer='glorot_uniform'))
        model.add(tf.keras.layers.Dense(vocab_size, activation='softmax'))

        model.compile(optimizer=optimizer, 
                        loss=loss,
                        metrics=metrics)

        summary_graph(model=model,index=index,hidden_layers=hidden_layers,date_time=date_time)
               
        return model
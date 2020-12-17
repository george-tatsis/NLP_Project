import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import datetime
import time
date_time = datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")

import tensorflow as tf
from config import data_file_path, seq_length, BUFFER_SIZE, BATCH_SIZE
from data_preprocessing import ProcessData
import build_model
import get_plot
import os

print("\n\n########################################################################")
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("########################################################################")


data = ProcessData()
data.clean_data(train=0.6,valid=0.2,test=0.2)

print("\n\n########################################################################")
print("train samples: {}".format(len(list(data.train))))
print("validation samples: {}".format(len(list(data.val))))
print("test samples: {}".format(len(list(data.test))))
print("########################################################################")

print("\nBuilding and compiling the model...\n")

# Parameters 
embedding_dim=200
rnn_units=600
num_hidden_layers=1


for _model_ in list(build_model.__dict__.keys())[-3:]:
    optimizers = ['adam','sgd','rmsprop']
    history_logs = []
    for optimizer in optimizers:
        model = build_model.__dict__[_model_](vocab_size = len(data.vocab),
                                    embedding_dim=embedding_dim,
                                    rnn_units=rnn_units,
                                    hidden_layers=num_hidden_layers,
                                    batch_size=BATCH_SIZE,
                                    optimizer=optimizer,
                                    loss='sparse_categorical_crossentropy',
                                    metrics=['sparse_categorical_accuracy'])


        print("\ntraining the model...\n")
        EPOCHS = 30

        callbacks = build_model.set_callbacks(architecture=_model_,
                                            hidden_layers=num_hidden_layers, 
                                            date_time=date_time, 
                                            optimizer=optimizer)

        history_logs.append(model.fit(x=data.train, 
                            validation_data=data.val,
                            epochs=EPOCHS,
                            callbacks=callbacks))

    get_plot.get_plot(history=history_logs,
                    epochs=EPOCHS,
                    architecture=_model_,
                    hidden_layers=num_hidden_layers,
                    date_time=date_time, 
                    labels=optimizers, 
                    metric='val_loss', 
                    title='Validation Loss', 
                    ylabel='Loss')

    get_plot.get_plot(history=history_logs, 
                    epochs=EPOCHS,
                    architecture=_model_,
                    hidden_layers=num_hidden_layers, 
                    labels=optimizers,
                    date_time=date_time, 
                    metric='val_sparse_categorical_accuracy', 
                    title='Validation Accuracy', 
                    ylabel='Accuracy')

    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
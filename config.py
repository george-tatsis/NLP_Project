import os
import tensorflow as tf

#Project directories

root_path = os.path.dirname(os.path.realpath(__file__))

data_folder = os.path.join(root_path , 'Data')
src_folder = os.path.join(root_path , 'Src')


#Dataset used in the project

data_file_path = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt') 

#Parameters
seq_length = 35
BATCH_SIZE = 20
BUFFER_SIZE = 10000
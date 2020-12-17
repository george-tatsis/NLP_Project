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

# Parameters 
embedding_dim=200
rnn_units=600
num_hidden_layers=1
optimizer='adam'
history_logs = []

data = ProcessData()
data.clean_data(train=0.6,valid=0.2,test=0.2)

model = build_model.GRU(vocab_size = len(data.vocab),
            embedding_dim=embedding_dim,
            rnn_units=rnn_units,
            hidden_layers=num_hidden_layers,
            batch_size=1,
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy'])


model.load_weights("models/GRU_1/2020.12.16-11.13.11/adam_checkpoint/adam_checkpoint")

model.build(tf.TensorShape([1, None]))


def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 500 

  # Converting our start string to numbers (vectorizing)
  token_start_string = nltk.tokenize.word_tokenize(start_string, language='english')
  input_eval = [data.word2idx[s] for s in token_start_string]
  input_eval = tf.expand_dims(input_eval, 0) 

  # Empty string to store our results
  text_generated = [] 

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0 

  # Here batch size == 1
  model.reset_states() 
  for _ in range(num_generate): 
      predictions = model(input_eval) 
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)  

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature  
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()  

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0) 

      text_generated.append(data.idx2word[predicted_id])

  return (start_string + ' '.join(text_generated))

print(generate_text(model, start_string=u"First Citizen:\nBefore we proceed any further, hear me speak."))


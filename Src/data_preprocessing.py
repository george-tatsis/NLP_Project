import tensorflow as tf
from config import data_file_path, seq_length, BUFFER_SIZE, BATCH_SIZE
from sklearn.model_selection import train_test_split
import numpy as np
import nltk

def word_tokenize(text):
    def intersperse(lst, item):
        result = [item] * (len(lst) * 2 - 1)
        result[0::2] = lst
        return result

    sentences = [t.split('\n') for t in text.split('\n\n')]
    sentences = intersperse([intersperse(t, '\n') for t in sentences], ['\n\n'])
    words = [[item] for sublist in sentences for item in sublist]
    words[0::2] = [nltk.tokenize.word_tokenize(item[0], language='english') for item in words if item[0] not in ['\n','\n\n']]
    words = [item for sublist in words for item in sublist]
    return words

class ProcessData:
    nltk.download('punkt')

    def __init__(self):
        self.file = data_file_path
        self.data = open(self.file, 'rb').read().decode(encoding='utf-8')
        self.words = None
        self.vocab = None
        self.word2idx = None
        self.idx2word = None
        self.text_as_int = None
        self.train = None
        self.val = None
        self.test = None

    def define_vocab(self):

        nltk.download('punkt')
        print("\n\n########################################################################")
        print("separating words...")
        self.words = word_tokenize(self.data)
        print("defining the vocabulary...")
        self.vocab = sorted(set(self.words))
        print("########################################################################")
        
    def stats(self):
        '''
        Print the number of total words and characters in the text and the number 
        of unique words and characters in the text.
        :return: None
        '''
        print("\n\n########################################################################")
        print("Words and characters in total: {}".format(len(self.words)))
        print("Unique words and characters: {}".format(len(self.vocab)))
        print("########################################################################")

    def vectorize_the_text(self):
        print("\n\n########################################################################")
        print("creating a dictionary assigning to each word a integer...")
        self.word2idx = {u:i for i, u in enumerate(self.vocab)}
        print("transforming the list of the unique words and characters to a vector...")
        self.idx2word = np.array(self.vocab)
        print("transforming the text into integers...")
        self.text_as_int = np.array([self.word2idx[c] for c in self.words]) 
        print("########################################################################")

    def clean_data(self,train,valid,test):
        self.define_vocab()
        self.stats()
        self.vectorize_the_text()
        word_dataset = tf.data.Dataset.from_tensor_slices(self.text_as_int)
        sequences = word_dataset.batch(seq_length+1, drop_remainder=True)

        def split_input_target(chunk):
            input_text = chunk[:-1]
            target_text = chunk[1:]
            return input_text, target_text

        dataset = sequences.map(split_input_target)
        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

        print("\n\n########################################################################")
        print("Dividing the dataset into the following:")
        print("Train: {}%".format(train*100))
        self.train = dataset.take(int(0.6 * len(list(dataset))))

        print("Validation: {}%".format(valid*100))
        test_dataset = dataset.skip(int(0.6 * len(list(dataset))))
        self.val = test_dataset.skip(int(0.5 * len(list(test_dataset))))
        
        print("Test: {}%".format(test*100))
        self.test = test_dataset.take(int(0.5 * len(list(test_dataset))))
        
        print("DONE!")
        print("########################################################################")
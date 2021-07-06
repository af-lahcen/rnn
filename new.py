import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, LSTM,GRU, Embedding, Activation, Lambda, Bidirectional,SimpleRNN
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from sklearn.utils import class_weight
from keras import backend as K
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json
import os
import pydot
import graphviz
from keras.utils import np_utils
from tensorflow import keras
from tensorflow.keras import layers

EPOCHS = 50 #  an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation.
BATCH_SIZE = 10 # a set of N samples. The samples in a batch are processed` independently, in parallel. If training, a batch results in only one update to the model.
INPUT_DIM = 4 # a vocabulary of 4 words in case of fnn sequence (ATCG)
OUTPUT_DIM = 10 # Embedding output
RNN_HIDDEN_DIM = 10
DROPOUT_RATIO = 0.2 # proportion of neurones not used for training
MAXLEN = 1 # cuts text after number of these characters in pad_sequences
checkpoint_dir ='checkpoints'
os.path.exists(checkpoint_dir)
input_file = 'data.csv'
matplotlib.use('pdf')
import pandas as pd
import numpy as np
import sys
import csv

csv.field_size_limit(sys.maxsize)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.client.session import Session

train_file_path = tf.keras.utils.get_file("data1.csv", "file:///Users/lafkir/Desktop/untitled folder 3/my/data1.csv")


train_dataset = tf.data.experimental.make_csv_dataset(
    train_file_path,
    batch_size=1, 
    label_name='cancer_type',
    num_epochs=1,
    header=True,
    select_columns = [15,16],
    ignore_errors=True,)

def letter_to_index(letter):
    _alphabet = 'ATGC'
    return next((i for i, _letter in enumerate(_alphabet) if _letter == letter.upper()), 0)

def get_sequence(x):
    if (x is None):
        raise None
    return np.array([int(letter_to_index(e)) for e in str(x)] , np.short)
def get_class(x):
    return np_utils.to_categorical(x,num_classes=34,dtype='short')

@tf.function()
def pack_features_vector(features, labels):
    """Pack the features into a single array."""
    features = tf.numpy_function(get_sequence, [features['Sequence'][0]], [np.short])
    labels = tf.numpy_function(get_class, [labels[0]], [np.short])
    return features, labels


train_dataset=train_dataset.map(pack_features_vector)


#sequence , label = next(iter(train_dataset))

#print(sequence.shape)
#print(label.shape)

model = Sequential()
model.add(Embedding(input_dim = INPUT_DIM, output_dim = OUTPUT_DIM, name='embedding_layer'))
model.add(SimpleRNN(RNN_HIDDEN_DIM , return_sequences=False))
model.add(Dense(34, activation='softmax'))
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])


model.compile(
    optimizer="adam",  
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    run_eagerly=True
)

history = model.fit(
    train_dataset,
    epochs=EPOCHS
)

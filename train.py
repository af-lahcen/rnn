import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('pdf')
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
EPCOHS = 300 #  an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation.
BATCH_SIZE = 500 # a set of N samples. The samples in a batch are processed` independently, in parallel. If training, a batch results in only one update to the model.
INPUT_DIM = 4 # a vocabulary of 4 words in case of fnn sequence (ATCG)
OUTPUT_DIM = 34 # Embedding output
RNN_HIDDEN_DIM = 34
DROPOUT_RATIO = 0.2 # proportion of neurones not used for training
MAXLEN = 2000000 # cuts text after number of these characters in pad_sequences
checkpoint_dir ='checkpoints'
os.path.exists(checkpoint_dir)

input_file = 'data.csv'

def letter_to_index(letter):
    _alphabet = 'ATGC'
    return next((i for i, _letter in enumerate(_alphabet) if _letter == letter.upper()), None)

def load_data(test_split = 0.1, maxlen = MAXLEN):
    print ('Loading data...')
    df = pd.read_csv(input_file)
    df['Sequence'] = df['Sequence'].apply(lambda x: [int(letter_to_index(e)) for e in x])
    df = df.reindex(np.random.permutation(df.index))
    train_size = int(len(df) * (1 - test_split))
    X_train = df['Sequence'].values[:train_size]
    y_train = np.array(df['cancer_type'].values[:train_size])
    X_test = np.array(df['Sequence'].values[train_size:])
    y_test = np.array(df['cancer_type'].values[train_size:])
    y_train=np_utils.to_categorical(y_train,num_classes=34)
    y_test=np_utils.to_categorical(y_test,num_classes=34)
    print('Average train sequence length: {}'.format(np.mean(list(map(len, X_train)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, X_test)), dtype=int)))
    return pad_sequences(X_train, maxlen=maxlen), y_train, pad_sequences(X_test, maxlen=maxlen), y_test

def create_lstm(input_length, rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO):
    model = Sequential()
    model.add(Embedding(input_dim = INPUT_DIM, output_dim = output_dim, input_length = input_length, name='embedding_layer'))
    model.add(LSTM(rnn_hidden_dim))
    model.add(Dense(34, activation='softmax'))
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    return model

def create_gru(input_length, rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO):
    model = Sequential()
    model.add(Embedding(input_dim = INPUT_DIM, output_dim = output_dim, input_length = input_length, name='embedding_layer'))
    model.add(GRU(rnn_hidden_dim))
    model.add(Dense(34, activation='softmax'))
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    return model

def create_rnn(input_length, rnn_hidden_dim = RNN_HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM, dropout = DROPOUT_RATIO):
    model = Sequential()
    model.add(Embedding(input_dim = INPUT_DIM, output_dim = output_dim, input_length = input_length, name='embedding_layer'))
    model.add(SimpleRNN(rnn_hidden_dim))
    model.add(Dense(34, activation='softmax'))
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    return model

def create_plots(history,name):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(name+'-accuracy.png')
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(name+'-loss.png')
    plt.clf()
from keras.utils import np_utils
def main(model,name):
    # train
    X_train, y_train, X_test, y_test = load_data() 
    # save checkpoint
    filepath= checkpoint_dir + "/"+ name +"/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    print ('Fitting model...')
    #class_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    #print(class_weight)
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE,epochs=EPCOHS, callbacks=callbacks_list, validation_split = 0.1, verbose = 1)

    # serialize model to JSON
    model_json = model.to_json()
    with open(name+"-model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("-model.h5")
    print("Saved model to disk")
    create_plots(history,name)
    #plot_model(model, to_file=name+'-model.png')

    # validate model on unseen data
    score, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    print('Validation score:', score)
    print('Validation accuracy:', acc)


model = create_rnn(None) 
main(model,"rnn")
model = create_lstm(None) 
main(model,"lstm")
model = create_gru(None) 
main(model,"gru")

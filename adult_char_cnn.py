""" ConvNet adult query classifier.

Heavily inspired by/cribbed from the Keras
imdb_cnn and pretrained_embeddings examples.
"""
from __future__ import print_function
from glob import glob
from datetime import datetime
from time import time
import numpy as np
# import json
import cPickle as pickle
from sklearn.metrics import classification_report
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Merge
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import *
# import ipdb


# DEFINE SOME USEFUL STUFF
VALIDATION_SPLIT = 0.1
EMBEDDING_DIM = 32
MAX_SEQUENCE_LENGTH = 60
BATCH_SIZE = 64
FILTERS = 100
KERNEL_SIZES = (3, 4, 5, 6)
KERNEL_SIZE = 3
HIDDEN_DIMS = 250
P_DROPOUT = 0.25
EPOCHS = 5

# START TRACKING TIMING
tick = time()

# LOAD/PREPROCESS DATA
print('Train/val data loading')
texts = []
labels = []
for f in glob("data/train/*"):
    if "clean" in f:
        label = 0
    else:
        label = 1
    for line in open(f):
        texts.append(line)
        labels.append(label)
print('Found %d texts' % (len(texts),))
print('##### %d seconds elapsed #####' % (time() - tick,))

print('Tokenizing')
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %d unique tokens' % (len(word_index),))
print('##### %d seconds elapsed #####' % (time() - tick,))

print('Padding, encoding, train/dev split')
data = pad_sequences(sequences,
                     maxlen=MAX_SEQUENCE_LENGTH,
                     padding='post',
                     truncating='post')
labels = np.asarray(labels)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
# print('Shape of data tensor:', data.shape)
# print('Shape of label tensor:', labels.shape)

nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
x_train = data[:-(nb_validation_samples*2)]
x_val = data[-(nb_validation_samples*2):-nb_validation_samples]
y_train = labels[:-(nb_validation_samples*2)]
y_val = labels[-(nb_validation_samples*2):-nb_validation_samples]
# print('Shape of training set:', x_train.shape)
# print('Shape of validation set:', x_val.shape)
print('##### %d seconds elapsed #####' % (time() - tick,))


print('Build multi-kernel-width char model')
submodels = []
for kw in KERNEL_SIZES:
    submodel = Sequential()
    # trainable char-embeddings
    submodel.add(Embedding(len(word_index) + 1,
                           EMBEDDING_DIM,
                           input_length=MAX_SEQUENCE_LENGTH,
                           trainable=True))
    submodel.add(Conv1D(FILTERS,
                        kw,
                        padding='valid',
                        activation='relu',
                        strides=1))
    submodel.add(GlobalMaxPooling1D())
    submodels.append(submodel)
char_model = Sequential()
char_model.add(Merge(submodels, mode="concat"))
char_model.add(Dense(HIDDEN_DIMS))
char_model.add(Dropout(P_DROPOUT))
char_model.add(Activation('relu'))
char_model.add(Dense(1))
char_model.add(Activation('sigmoid'))
print('Compiling model')
char_model.compile(loss='binary_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])
print(char_model.summary())
print('##### %d seconds elapsed #####' % (time() - tick,))


print('Training char model with KERNEL_SIZES=%s' % str(KERNEL_SIZES))
callbacks = [EarlyStopping(patience=0, min_delta=0.001, verbose=1)]
# replicate inputs across all "filter" channels (for train and val sets)
hist = char_model.fit([x_train, x_train, x_train, x_train],
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      validation_data=([x_val, x_val, x_val, x_val], y_val),
                      callbacks=callbacks)
print('##### %d seconds elapsed #####' % (time() - tick,))

# serialize model to JSON and weights to HDF5
now = datetime.now().strftime("%Y%d_%H%M")
model_json = char_model.to_json()
with open("char_model_%s.json" % now, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
char_model.save_weights("char_model_%s.wts" % now)
print("Saved model to char_model_%s" % now)
# # load json and create model: note that you need to compile
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

print('##### %d seconds elapsed #####' % (time() - tick,))

# compare against existing/deployed sklearn adult_stat_detector
pipeline = pickle.load(open("adult.stat.20150702_1534.pkl", "rb"))
# data prep
texts = np.asarray(texts)
texts = texts[indices]
x_dev_raw = texts[-nb_validation_samples:]
x_dev = data[-nb_validation_samples:]
y_dev = labels[-nb_validation_samples:]

# compare evaluation
print('\nComparative Eval on held-out\n')
print('##### AdultDetectorStat #####')
print(classification_report(y_dev, pipeline.predict(x_dev_raw), digits=4))
print('\n##### ConvNet eval #####')
# ipdb.set_trace()
print(classification_report(y_dev,
                            np.round(char_model.predict([x_dev, x_dev, x_dev, x_dev])).astype('int32'),
                            digits=4))

print('\nComparative Eval on test\n')
# LOAD/PREPROCESS DATA
print('Test data loading')
test_texts = []
test_labels = []
for f in glob("data/dev/*"):
    if "clean" in f:
        label = 0
    else:
        label = 1
    for line in open(f):
        test_texts.append(line)
        test_labels.append(label)
print('Found %d texts' % (len(test_texts),))
print('##### %d seconds elapsed #####' % (time() - tick,))

print('Tokenizing/sequencifying')
test_sequences = tokenizer.texts_to_sequences(test_texts)
word_index = tokenizer.word_index
print('Found %d unique tokens' % (len(word_index),))
print('##### %d seconds elapsed #####' % (time() - tick,))

print('Padding/encoding')
test_data = pad_sequences(test_sequences,
                          maxlen=MAX_SEQUENCE_LENGTH,
                          padding='post',
                          truncating='post')
test_labels = np.asarray(test_labels)
test_indices = np.arange(test_data.shape[0])
np.random.shuffle(test_indices)
test_texts = np.asarray(test_texts)[test_indices]
test_data = test_data[test_indices]
test_labels = test_labels[test_indices]
print('##### %d seconds elapsed #####' % (time() - tick,))


print('AdultDetectorStat eval')
print(classification_report(test_labels, pipeline.predict(test_texts), digits=4))
print('##### %d seconds elapsed #####' % (time() - tick,))
print('\nConvNet eval')
# ipdb.set_trace()
print(classification_report(test_labels,
                            np.round(char_model.predict([test_data,
                                                         test_data,
                                                         test_data,
                                                         test_data])).astype('int32'), digits=4))

print('##### %d seconds elapsed TOTAL #####\n' % (time() - tick,))

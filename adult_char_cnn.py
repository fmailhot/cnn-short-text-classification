""" ConvNet adult query classifier.

Heavily inspired by/cribbed from the Keras
imdb_cnn and pretrained_embeddings examples.
"""
from __future__ import print_function
from glob import glob
from datetime import datetime
from time import time
import numpy as np
import json
from sklearn.metrics import classification_report
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Merge
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import *
import ipdb


# DEFINE SOME USEFUL STUFF
VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 50
MAX_SEQUENCE_LENGTH = 60
BATCH_SIZE = 32
FILTERS = 100
KERNEL_SIZES = (3, 4, 5, 6)
KERNEL_SIZE = 3
HIDDEN_DIMS = 250
P_DROPOUT = 0.25
EPOCHS = 5

# START TRACKING TIMING
tick = time()

# LOAD/PREPROCESS DATA
print('Data loading')
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
# labels = to_categorical(np.asarray(labels))
labels = np.asarray(labels)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
x_train = data[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_train = labels[:-nb_validation_samples]
y_val = labels[-nb_validation_samples:]
print('Shape of training set:', x_train.shape)
print('Shape of validation set:', x_val.shape)
print('##### %d seconds elapsed #####' % (time() - tick,))

# # PREPARE THE EMBEDDINGS INDEX AND MATRIX
# print('Preparing embedding index')
# embeddings_index = {}
# for line in open("glove/glove.6B.%dd.txt" % EMBEDDING_DIM):
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# print('Found %s word vectors.' % len(embeddings_index))

# print('Preparing embedding matrix')
# embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
# for word, i in word_index.items():
#     embedding_vector = embeddings_index.get(word, None)
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector
# print('##### %d seconds elapsed #####' % (time() - tick,))

# BUILD SIMPLE MODEL
# print('Building model')
# # N.B. Since we're using conv layers with different kernel
# # sizes, we need to split out the inputs into replicated "channels"
# # and then merge them past the pooling
# model = Sequential()
# # Add a (non-trainable) embedding layer
# embedding_layer = Embedding(len(word_index) + 1,
#                             EMBEDDING_DIM,
#                             weights=[embedding_matrix],
#                             input_length=MAX_SEQUENCE_LENGTH,
#                             trainable=False)
# model.add(embedding_layer)
# # Add a Convolution1D, which will learn FILTERS
# # word group filters of size KERNEL_SIZE
# model.add(Conv1D(FILTERS,
#                  KERNEL_SIZE,
#                  padding='valid',
#                  activation='relu',
#                  strides=1))
# # Add a max pooling layer
# # See here for good explanation of Global Max Pooling
# # https://stats.stackexchange.com/questions/257321/...
# # what-is-global-max-pooling-layer-and-what-is-its-advantage-over-maxpooling-layer
# model.add(GlobalMaxPooling1D())
# # Add a vanilla hidden layer, trained w/ Ddropout
# #TODO: optimize HIDDEN_DIMS and P_DROPOUT
# model.add(Dense(HIDDEN_DIMS))
# model.add(Dropout(P_DROPOUT))
# model.add(Activation('relu'))
# # Project onto a single unit output, squashed with sigmoid
# model.add(Dense(1))
# model.add(Activation('sigmoid'))
# # Compile w/ fairly standard loss & optimizer
# print('Compiling model')
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
# print(model.summary())
# print('##### %d seconds elapsed #####' % (time() - tick,))

# Build "multi-channel" (i.e. diff kernel widths model)
print('Build multi-kernel-width model')
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
big_model = Sequential()
big_model.add(Merge(submodels, mode="concat"))
big_model.add(Dense(HIDDEN_DIMS))
big_model.add(Dropout(P_DROPOUT))
big_model.add(Activation('relu'))
big_model.add(Dense(1))
big_model.add(Activation('sigmoid'))
print('Compiling model')
big_model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
print(big_model.summary())
print('##### %d seconds elapsed #####' % (time() - tick,))

### TRAIN ###

# print('Training simple model')
# # add early-stopping to militate against overfitting
# callbacks = [EarlyStopping(patience=1, min_delta=0.001, verbose=1)]
# # save the epoch-wise avg loss, via automagic history callback
# hist = model.fit(x_train, y_train,
#                  batch_size=BATCH_SIZE,
#                  epochs=EPOCHS,
#                  validation_data=(x_val, y_val),
#                  callbacks=callbacks)
# print('History')
# print(json.dumps(hist.history, indent=2))
# now = datetime.now().strftime("%Y%d_%H%M")
# model.save('model_%s.dat' % (now,))
# print('##### %d seconds elapsed #####' % (time() - tick,))

print('Training big model')
callbacks = [EarlyStopping(patience=0, min_delta=0.001, verbose=1)]
# replicate inputs across all "filter" channels (for train and val sets)
hist = big_model.fit([x_train, x_train, x_train, x_train],
                     y_train,
                     batch_size=BATCH_SIZE,
                     epochs=EPOCHS,
                     validation_data=([x_val, x_val, x_val, x_val], y_val),
                     callbacks=callbacks)
print('##### %d seconds elapsed #####' % (time() - tick,))
print('History')
print(json.dumps(hist.history, indent=2))
now = datetime.now().strftime("%Y%d_%H%M")
big_model.save('big_model_%s.dat' % (now,))
print('##### %d seconds elapsed #####' % (time() - tick,))

# # compare against sklearn w/ same pipeline as adult_stat_detector and
# # trained on same data
# print('Building and training sklearn pipeline')
# from sklearn.feature_extraction.text import CountVectorizer as CVec
# from sklearn.pipeline import FeatureUnion
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LogisticRegressionCV as LRCV
# # vals obtained from fairly thorough grid-search
# word_vec = CVec(analyzer="word",
#                 ngram_range=(1, 2),
#                 max_features=None,
#                 max_df=0.85,
#                 min_df=100,
#                 stop_words="english",
#                 lowercase=True,
#                 binary=False)
# char_vec = CVec(analyzer="char",
#                 ngram_range=(3, 5),
#                 max_features=None,
#                 max_df=0.85,
#                 min_df=100,
#                 stop_words="english",
#                 lowercase=True,
#                 binary=False)
# trig_vec = CVec(analyzer="word",
#                 ngram_range=(1, 3),
#                 max_features=10000,
#                 max_df=0.85,
#                 min_df=100,
#                 stop_words=None,
#                 lowercase=True,
#                 binary=False)

# # vec = FeatureUnion([("word_vec", word_vec), ("char_vec", char_vec)])
# lr = LRCV()
# # pipeline = Pipeline([("vec", vec), ("clf", lr)])
# pipeline = Pipeline([("vec", trig_vec), ("clf", lr)])
# # data prep
# texts = np.asarray(texts)
# texts = texts[indices]
# x_train_raw = texts[:-nb_validation_samples]
# y_train_raw = labels[:-nb_validation_samples]
# x_dev_raw = texts[-100:]
# y_dev_true = labels[-100:]
# # fit model
# pipeline.fit(x_train_raw, y_train_raw)
# print('##### %d seconds elapsed #####' % (time() - tick,))

# # compare evaluation
# print('\nComparative Eval\n')
# print('AdultDetectorStat eval')
# print(classification_report(y_dev_true, pipeline.predict(x_dev_raw), digits=3))
print('ConvNet eval')
# ipdb.set_trace()
print(classification_report(labels[-100:],
                            np.round(big_model.predict([x_val[-100:],
                                                        x_val[-100:],
                                                        x_val[-100:],
                                                        x_val[-100:]])).astype('int32'),
                            digits=4))

print('##### %d seconds elapsed TOTAL #####' % (time() - tick,))

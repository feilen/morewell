#!/usr/bin/env python2
import re
import unicodecsv as csv
import numpy as np
import random
import theano
import theano.tensor as T
import sys
import json

from glove import Corpus, Glove
import gensim
from gensim import utils

from scipy.stats import norm

from lasagne import layers
from lasagne.updates import nesterov_momentum, adagrad, adam
from lasagne.nonlinearities import softmax, rectify, tanh
from nolearn.lasagne import NeuralNet
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from nn_utils import make_nn, vectify, read_double_corpus, make_filtered_dict

with open('settings.json', 'r') as settingsfile:
    settings = json.load(settingsfile)

# 5 seems to do best?
number_components = settings['number_components']
# Drop probablity, set until train/val is stable
drop_probability = float(settings['drop_probability'])

# Bundle groups of symbols and make sure words are otherwise alone
print("Parsing CSV log")
with open(sys.argv[1], 'r') as myfile:
    mycsv = csv.reader(myfile)
    csvsequence = list(mycsv)

texts = []
classes = []
for row in csvsequence:
    texts.append([token.encode('utf8') for token in utils.tokenize(row[3], lower=True, errors='ignore')])
    classes.append(row[0])

try:
    corpus = Corpus()
    print("Loading pretrained corpus...")
    corpus = Corpus.load("cache/corpus.p")
except:
    corpus = Corpus(make_filtered_dict(read_double_corpus()))
    print("Training corpus...")
    corpus.fit(read_double_corpus(), ignore_missing=True)
    print("Saving corpus...")
    corpus.save("cache/corpus.p")

# Calculate distribution, to account for 95th percentile of messages.
max_sentence_length = int(np.mean([len([y for y in x if y in corpus.dictionary]) for x in texts]) + (norm.ppf(0.95) * np.std([len(x) for x in texts])))

print("Max sentence length: {}, put that in settings.json.".format(max_sentence_length))

glove = Glove(no_components=number_components, learning_rate=0.05)
try:
    print("Loading pretrained GloVe vectors...")
    glove = Glove.load("cache/glove.p")
except:
    print("Training GloVe vectors...")
    # More epochs seems to make it worse
    glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    glove.save("cache/glove.p")

# Convert input text
print("Vectorizing input sentences...")
X = vectify(texts, None, glove.dictionary, max_sentence_length)
y = np.array([x == u'1' for x in classes]).astype(np.int32)

X, y, texts = X[:207458], y[:207458], texts[:207458]

def print_accurate_forwards(net, history):
    X_train, X_valid, y_train, y_valid = net.train_split(X, y, net)
    y_classified = net.predict(X_valid)
    acc_fwd = np.mean([x == y_ and y_ == 1 for x, y_ in zip(y_valid, y_classified)])/np.mean(y_valid)
    fls_pos = np.mean([x != y_ and y_ == 0 for x, y_ in zip(y_classified, y_valid)])/(np.mean(y_valid))
    print('Accurately forwarded: {:.4f}'.format(acc_fwd) + ', False Positives: {:.4f}'.format(fls_pos) + ', Valid forwards: {:.4f}'.format((acc_fwd / (acc_fwd + fls_pos))) )

net = make_nn(max_sentence_length, glove.word_vectors, drop_probability)
net.on_epoch_finished = [print_accurate_forwards]
net.fit(X, y)

## Train and run the network
classified = net.predict(X)
with open('chat-optimized.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile)
    index = 0
    for index, (row, predicted_class, actual_class) in enumerate(zip(csvsequence, classified, y)):
        class_written = ''
        if predicted_class == 1 and actual_class == 1:
            class_written = 1
        if predicted_class == 0 and actual_class == 1:
            class_written = 'false_negative'
        if predicted_class == 1 and actual_class == 0:
            class_written = 'false_positive'
        spamwriter.writerow(
            [class_written] +
            [row[1]] +
            [row[2]] +
            [row[3]] +
            [index]
        )
        index += 1

net.save_params_to("cache/model.p")

#!/usr/bin/env python2
import re
import csv
import numpy as np
import random
import json
import theano

from glove import Corpus, Glove

from sklearn.preprocessing import Imputer
from lasagne import layers
from lasagne.updates import nesterov_momentum, adagrad, adam
from lasagne.nonlinearities import softmax, rectify, tanh
from nolearn.lasagne import NeuralNet
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

max_sentence_length = 23
number_components = 4

logfile = open('After_Dark_Furry_Femboys.jsonl', 'r')

jsons = []
for line in logfile.readlines():
    try:
        json_parsed = json.loads(line)
        if 'text' in json_parsed:
            jsons.append(json_parsed)
    except KeyError:
        pass

# Create a fast lookup map
id_lookup = dict()
for index, message in enumerate(jsons):
    id_lookup[message['id']] = index

def vectify(sentence_array):
    vectorized = np.empty((len(sentence_array), max_sentence_length, number_components), dtype=np.float32)
    for index, sentence in enumerate(sentence_array):
        for w_index, word in reversed(list(enumerate(sentence))):
            if max_sentence_length - len(sentence) + w_index > 0:
                #try:
                vectorized[index][max_sentence_length - len(sentence) + w_index ] = glove.word_vectors[glove.dictionary[word.lower()]]
                #except:
                #    vect_sentence.extend([np.nan] * number_components)
    return vectorized

# Bundle groups of symbols and make sure words are otherwise alone
print("Parsing CSV log")
regex_find = re.compile(r"([^a-z 0-9]+)")
trimspace = re.compile(r" +")

def clean(sentence):
    return re.sub(trimspace, " ", re.sub(regex_find, r" \1 ", sentence.lower()))

myfile = open('input.csv', 'r')
mycsv = csv.reader(myfile)

texts = []
classes = []
for row in mycsv:
    texts.append(clean(row[3]).split())
    classes.append(row[0])

corpus = Corpus()
try:
    print("Loading pretrained corpus...")
    corpus = Corpus.load("corpus.p")
except:
    print("Training corpus...")
    corpus.fit(texts, window=max_sentence_length)
    corpus.save("corpus.p")

glove = Glove(no_components=number_components, learning_rate=0.05)
try:
    print("Loading pretrained GloVe vectors...")
    glove = Glove.load("glove.p")
except:
    print("Training GloVe vectors...")
    # More epochs seems to make it worse
    glove.fit(corpus.matrix, epochs=200, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    glove.save("glove.p")

# Convert input text
X = vectify(texts)
X_context = np.empty((len(X), max_sentence_length * 2, number_components), dtype=np.float32)
# Feed context. Context being 'replied to', 'prior message' or 'NaN'
for index, vector in enumerate(X):
    try:
        X_context[index] = np.concatenate((X[index], (X[id_lookup[jsons[index]['reply_id']]])))
    except (KeyError, IndexError):
        try:
            X_context[index] = np.concatenate((X[index], (X[index + 1])))
        except IndexError:
            pass

X = X_context

#impute input text
for sentence in X:
    imp = Imputer(missing_values='NaN', strategy='mean')
    imp.fit(sentence)
    sentence = imp.transform(sentence)

y = np.array([x is '1' for x in classes]).astype(np.int32)

#X = X[:207458]
#y = y[:207458]

# Split the data into train and test instances
#paired = zip(X, y, texts)
#shuffle(X, y, texts)

test_X = X[len(X) * 7 / 10:]
X = X[:len(X) * 7 / 10]

test_y = y[len(y) * 7 / 10:]
y = y[:len(y) * 7 / 10]

test_texts = texts[len(texts) * 7 / 10:]
#train_unzip = [np.array(t) for t in zip(*train_instances)]
#test_unzip = [np.array(t) for t in zip(*test_instances)]

#X = train_unzip[0]
#y = train_unzip[1]

#test_X = test_unzip[0]
#test_y = test_unzip[1]

#scaler = StandardScaler()
#scaler.fit(X)
#X = scaler.transform(X)
#test_X = scaler.transform(test_X)

#X = X.reshape(
#        -1,  # number of samples, -1 makes it so that this number is determined automatically
#        max_sentence_length * 2,   # 1 color channel, since images are only black and white
#        number_components # second image dimension (horizontal)
#    )

#test_X = test_X.reshape(
#        -1,  # number of samples, -1 makes it so that this number is determined automatically
#        max_sentence_length * 2,   # 1 color channel, since images are only black and white
#        number_components # second image dimension (horizontal)
#    )

# Train over data, produce regression NN, spit out the middle 10% and run statistics
net = NeuralNet(
        layers=[  # three layers: one hidden layer
            ('input', layers.InputLayer),
            #('drop', layers.DropoutLayer),
            ('lstm1', layers.LSTMLayer),
            ('lstm2', layers.LSTMLayer),
#            ('lstm3', layers.LSTMLayer),
            ('output', layers.DenseLayer),
            ],
        # layer parameters:
        input_shape = (None, max_sentence_length * 2, number_components),  # this code won't compile without SIZE being set

        lstm1_num_units = number_components,
        #lstm1_nonlinearity = tanh,
        lstm2_num_units = number_components,
        #lstm2_nonlinearity = tanh,
        #lstm3_num_units = number_components,
        #lstm3_nonlinearity = tanh,

        output_nonlinearity = softmax,  # output layer uses identity function
        output_num_units = 2,  # this code won't compile without OUTPUTS being set

        # optimization method:
        update=adam,
        update_learning_rate=0.01,

        regression=False,  # If you're doing classification you want this off
        max_epochs=125,  # more epochs can be good,
        verbose=1, # enabled so that you see meaningful output when the program runs
        )

## Train the network
net.fit(X, y)

# Test classifier
classified = net.predict(X)
error = np.mean( classified != y )
print('Accuracy on train data: {}'.format(1-error))
test_classified = net.predict(test_X)
test_error = np.mean( test_classified != test_y )
print('Accuracy on test data: {}'.format(1-test_error))
print('Accurately forwarded: {}'.format(np.mean([x == y and y == 1 for x, y in zip(test_y, test_classified)])/np.mean(test_y)))
for index, item in enumerate(test_classified):
    if item == 1:
        print(test_texts[index])

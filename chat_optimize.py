#!/usr/bin/env python2
import re
import unicodecsv as csv
import numpy as np
import random
import json
import theano
import theano.tensor as T

from glove import Corpus, Glove

from scipy.stats import norm

from sklearn.preprocessing import Imputer
from lasagne import layers
from lasagne.updates import nesterov_momentum, adagrad, adam
from lasagne.nonlinearities import softmax, rectify, tanh
from nolearn.lasagne import NeuralNet
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# 5 seems to do best?
number_components = 5
# Include the last few messages from the same conversation, same user in train data
contextual = True
# How many max posts previous to check for the same username
max_context_backstep = 5
# Drop probablity, set until train/val is stable
drop_probability = 0.5

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

# Create a reply chain map
previous_message = dict()
for index, message in enumerate(jsons):
    for jindex, jmessage in enumerate(jsons[index:index+max_context_backstep]):
        if 'username' in message['from'] and 'username' in jmessage['from'] and (message['from']['username'] == jmessage['from']['username']):
            previous_message[index] = jindex
            break
        else:
            previous_message[index] = -1
    #try:
    #    previous_message[index] = id_lookup[jsons[index]['reply_id']]
    #except KeyError:
    #    previous_message[index] = index + 1

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

# Calculate distribution, to account for 95th percentile of messages.
max_sentence_length = int(np.mean([len(x) for x in texts]) + (norm.ppf(0.95) * np.std([len(x) for x in texts])))

def vectify(sentence_array):
    vectorized = np.zeros((len(sentence_array), max_sentence_length), dtype=np.int32)
    for index, sentence in enumerate(sentence_array):
        # Construct a list of word indexes, such the current classified message is at the end, and previous messages are in every spot leading up to it.
        # Ex: "hi how are you ." "hi there" -> ". are you hi there"
        sentence_index = max_sentence_length
        cur_chain_message_index = index
        if contextual:
            while sentence_index > 0:
                try:
                    working_sentence = texts[cur_chain_message_index]
                except IndexError:
                    pass # Just feed in the same sentence I guess
                # Overwrite portion of sentence
                np.put(vectorized[index], np.arange(len(working_sentence)) + (sentence_index - len(working_sentence)), [glove.dictionary[word.lower()] for word in working_sentence], mode='clip')
                sentence_index -= len(working_sentence)
                # Move back one message until we've filled it
                try:
                    cur_chain_message_index = previous_message[cur_chain_message_index]
                except KeyError:
                    pass
        else:
            working_sentence = texts[cur_chain_message_index]
            np.put(vectorized[index], np.arange(len(working_sentence)) + (sentence_index - len(working_sentence)), [glove.dictionary[word.lower()] for word in working_sentence], mode='clip')


    return vectorized


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
    glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    glove.save("glove.p")

# Convert input text
X = vectify(texts)
y = np.array([x == u'1' for x in classes]).astype(np.int32)

X, y, jsons, texts = X[:207458], y[:207458], jsons[:207458], texts[:207458]

#X, y, jsons, texts, indexes = shuffle(X, y, jsons, texts, np.arange(len(X)))

def print_accurate_forwards(net, history):
    X_train, X_valid, y_train, y_valid = net.train_split(X, y, net)
    y_classified = net.predict(X_valid)
    print('Accurately forwarded: {}'.format(np.mean([x == y_ and y_ == 1 for x, y_ in zip(y_valid, y_classified)])/np.mean(y_valid)) + ', False Positives: {}'.format(np.mean([x != y_ and y_ == 0 for x, y_ in zip(y_classified, y_valid)])/(np.mean(y_valid))))

net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('embed', layers.EmbeddingLayer),
            ('drop', layers.DropoutLayer),
            ('lstm1', layers.LSTMLayer),
            ('lstm2', layers.LSTMLayer),
            ('lstm3', layers.LSTMLayer),
            ('output', layers.DenseLayer),
            ],
        # layers
        input_shape = (None, max_sentence_length),
        input_input_var=T.imatrix(),

        embed_input_size = glove.word_vectors.shape[0],
        embed_output_size = glove.word_vectors.shape[1],
        embed_W = np.array(glove.word_vectors, dtype=np.float32),

        drop_p = drop_probability,

        lstm1_num_units = number_components,
        #lstm1_nonlinearity = tanh,
        lstm2_num_units = number_components,
        #lstm2_nonlinearity = tanh,
        lstm3_num_units = number_components,
        #lstm2_nonlinearity = tanh,

        output_nonlinearity = softmax,
        output_num_units = 2,

        # optimization
        update=adam,
        update_learning_rate=0.005,

        regression=False,
        max_epochs=45,

        verbose=1,
        on_epoch_finished = [print_accurate_forwards],
        )

net.fit(X, y)

## Train and run the network
classified = net.predict(X)
with open('chat-optimized.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile)
    index = 0
    for index, (json, predicted_class, actual_class) in enumerate(zip(jsons, classified, y)):
        name = (json['from']['first_name'] if 'first_name' in json['from'] else "") + " " + (json['from']['last_name'] if 'last_name' in json['from'] else "")
        class_written = ''
        if predicted_class == 1 and actual_class == 1:
            class_written = 1
        if predicted_class == 0 and actual_class == 1:
            class_written = 'false_negative'
        if predicted_class == 1 and actual_class == 0:
            class_written = 'false_positive'
        spamwriter.writerow(
            [class_written] +
            [name] +
            ([json['from']['username']] if 'username' in json['from'] else ['']) +
            ([json['text']] if 'text' in json else ['']) +
            [index]
        )
        index += 1

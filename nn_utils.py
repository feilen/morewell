from nolearn.lasagne import NeuralNet
import numpy as np
import theano.tensor as T
import re

from lasagne import layers
from lasagne.updates import nesterov_momentum, adagrad, adam
from lasagne.nonlinearities import softmax, rectify, tanh
from gensim import utils
import gensim

def make_nn(max_sentence_length, word_vectors, drop_probability, regress = False):
    return NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('embed', layers.EmbeddingLayer),
            ('drop', layers.DropoutLayer),
            #('conv1', layers.Conv1DLayer),
            ('lstm1', layers.LSTMLayer),
            ('lstm2', layers.LSTMLayer),
            ('lstm3', layers.LSTMLayer),
            #('pool', layers.MaxPool1DLayer),
            ('output', layers.DenseLayer),
            ],
        # layers
        input_shape = (None, max_sentence_length),
        input_input_var=T.imatrix(),

        embed_input_size = word_vectors.shape[0],
        embed_output_size = word_vectors.shape[1],
        embed_W = np.array(word_vectors, dtype=np.float32),

        drop_p = drop_probability,

        lstm1_num_units = word_vectors.shape[1],
        lstm2_num_units = word_vectors.shape[1],
        lstm3_num_units = word_vectors.shape[1],
        #conv1_num_filters=max_sentence_length, conv1_filter_size=3,# pool_pool_size=4,

        output_nonlinearity = None if regress else softmax,
        output_num_units = 1 if regress else 2,

        # optimization
        update=adam,
        update_learning_rate=0.005,

        regression=regress,
        max_epochs=45,

        verbose=1,
        )

def vectify(sentence_array, message_lookup_dictionary, word_lookup_dictionary, max_sentence_length):
    vectorized = np.zeros((len(sentence_array), max_sentence_length), dtype=np.int32)
    for index, sentence in enumerate(sentence_array):
        # Construct a list of word indexes, such the current classified message is at the end, and previous messages are in every spot leading up to it.
        sentence_index = max_sentence_length
        cur_chain_message_index = index
        try:
            working_sentence = sentence_array[cur_chain_message_index]
        except IndexError:
            break # Leave rest padded with noise
        # Overwrite portion of sentence
        current_vector= [word_lookup_dictionary[token] for token in working_sentence if token in word_lookup_dictionary]
        np.put(vectorized[index], np.arange(len(current_vector)) + (sentence_index - len(current_vector)), current_vector, mode='clip')
        sentence_index -= len(current_vector)
        # Move back one message until we've filled it

    return vectorized

def read_double_corpus(texts):
    corp = gensim.corpora.WikiCorpus("cache/simplewiki.xml.bz2", dictionary={})

    for text in corp.get_texts():
        yield text

    for text in texts:
        yield text

def make_filtered_dict(generating_function):
    dictionary = gensim.corpora.Dictionary(generating_function)
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=None)
    dictionary.compactify()
    return {v: k for k, v in dictionary.iteritems()}



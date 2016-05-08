from nolearn.lasagne import NeuralNet
import numpy as np
import theano.tensor as T
import re

from lasagne import layers
from lasagne.updates import nesterov_momentum, adagrad, adam
from lasagne.nonlinearities import softmax, rectify, tanh

regex_find = re.compile(r"([^a-z 0-9]+)")
trimspace = re.compile(r" +")

def make_nn(max_sentence_length, word_vectors, drop_probability):
    return NeuralNet(
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

        embed_input_size = word_vectors.shape[0],
        embed_output_size = word_vectors.shape[1],
        embed_W = np.array(word_vectors, dtype=np.float32),

        drop_p = drop_probability,

        lstm1_num_units = word_vectors.shape[1],
        #lstm1_nonlinearity = tanh,
        lstm2_num_units = word_vectors.shape[1],
        #lstm2_nonlinearity = tanh,
        lstm3_num_units = word_vectors.shape[1],
        #lstm3_nonlinearity = tanh,

        output_nonlinearity = softmax,
        output_num_units = 2,

        # optimization
        update=adam,
        update_learning_rate=0.005,

        regression=False,
        max_epochs=45,

        verbose=1,
        )

def clean(sentence):
    return re.sub(trimspace, " ", re.sub(regex_find, r" \1 ", sentence.lower()))

def vectify(sentence_array, message_lookup_dictionary, word_lookup_dictionary, max_sentence_length, contextual):
    vectorized = np.zeros((len(sentence_array), max_sentence_length), dtype=np.int32)
    for index, sentence in enumerate(sentence_array):
        # Construct a list of word indexes, such the current classified message is at the end, and previous messages are in every spot leading up to it.
        # Ex: "hi how are you ." "hi there" -> ". are you hi there"
        sentence_index = max_sentence_length
        cur_chain_message_index = index
        while sentence_index > 0:
            try:
                working_sentence = sentence_array[cur_chain_message_index]
            except IndexError:
                break # Leave rest padded with noise
            # Overwrite portion of sentence
            np.put(vectorized[index], np.arange(len(working_sentence)) + (sentence_index - len(working_sentence)), [word_lookup_dictionary[word.lower()] if word.lower() in word_lookup_dictionary else 0 for word in working_sentence], mode='clip')
            sentence_index -= len(working_sentence)
            # Move back one message until we've filled it
            if contextual:
                try:
                    cur_chain_message_index = message_lookup_dictionary[cur_chain_message_index]
                    if cur_chain_message_index == -1:
                        break
                except KeyError:
                    break # Leave rest padded with noise
            else:
                break

    return vectorized



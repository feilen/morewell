#!/usr/bin/python2
from icalendar import Calendar
import numpy as np
import random
import theano
import theano.tensor as T
import sys
import json
import urllib2
import datetime
from sets import Set

from glove import Corpus, Glove
import pickle
from scipy.stats import norm

from lasagne import layers
from lasagne.updates import nesterov_momentum, adagrad, adam
from lasagne.nonlinearities import softmax, rectify, tanh
from nolearn.lasagne import NeuralNet, PrintLog, PrintLayerInfo
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from nn_utils import make_nn, vectify, read_double_corpus, make_filtered_dict
import gensim
from gensim import utils

def user_sort(x, y):
    print("Which do you like more?")
    print('------------------------------------------')
    print(u"1: {}".format(unicode(x[1]['summary'])))
    print('------------------------------------------')
    print(u"{}".format(unicode(x[1]['description'])))
    print('------------------------------------------')
    print(u"2: {}".format(unicode(y[1]['summary'])))
    print('------------------------------------------')
    print(u"{}".format(unicode(y[1]['description'])))
    print('------------------------------------------')
    choice = raw_input(":")
    return 1 if choice == "1" else -1



input_calendars = ['https://www.meetup.com/SF-EDM/events/ical/']
try:
    parameters = pickle.load(open("cache/parameters.p", "rb"))
except:
    parameters = dict()
    parameters['number_components'] = 5
    parameters['drop_probability'] = 0.5


# Create GloVe corpus
events = []
for webcal in input_calendars:
    print("Requesting " + webcal)
    request = urllib2.Request(webcal,
            headers = {#"Host": "www.meetup.com",
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:28.0) Gecko/20100101 Firefox/28.0",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Content-Type": "application/x-www-form-urlencoded"})
    response = urllib2.urlopen(request)
    ical = Calendar.from_ical(response.read())
    # Sanity check, only work on events this year
    events += [event for event in ical.walk('vevent') if type(event.decoded('dtstart')) is datetime.datetime and event.decoded('dtstart').date() >= datetime.date.today() and event.decoded('dtstart').year == datetime.date.today().year]



summaries = [[token.encode('utf8') for token in utils.tokenize(event['summary'], lower=True, errors='ignore')] for event in events]
descriptions = [[token.encode('utf8') for token in utils.tokenize(event['description'], lower=True, errors='ignore')] for event in events]

if not 'max_summary_length' in parameters:
    parameters['max_summary_length'] = int(np.mean([len(x) for x in summaries]) + (norm.ppf(0.95) * np.std([len(x) for x in summaries])))
    parameters['max_description_length'] = int(np.mean([len(x) for x in descriptions]) + (norm.ppf(0.95) * np.std([len(x) for x in descriptions])))
    pickle.dump(parameters, open("cache/parameters.p", "wb"))

print("Max summary length: {}".format(parameters['max_summary_length']))
print("Max description length: {}".format(parameters['max_description_length']))

texts = summaries + descriptions

corpus = Corpus()
try:
    print("Loading pretrained corpus...")
    corpus = Corpus.load("cache/calendar_corpus.p")
except:
    corpus = Corpus(make_filtered_dict(read_double_corpus(texts)))
    print("Training corpus...")
    corpus.fit(read_double_corpus(texts), ignore_missing=True)
    print("Saving corpus...")
    corpus.save("cache/calendar_corpus.p")


glove = Glove(no_components=parameters['number_components'], learning_rate=0.05)
try:
    glove = Glove.load("cache/calendar_glove.p")
except:
    print("Training GloVe vectors...")
    # More epochs seems to make it worse
    glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    glove.save("cache/calendar_glove.p")

X_summaries = vectify(summaries, dict(), glove.dictionary, parameters['max_summary_length'])
X_descriptions = vectify(descriptions, dict(), glove.dictionary, parameters['max_description_length'])

# Cram all the data possible from ical VEVENTS into a neural network, with the -1 to 1 rating being the output
net = make_nn(parameters['max_summary_length'] + parameters['max_description_length'], glove.word_vectors, parameters['drop_probability'], True)
try:
    net.load_params_from("cache/calendar_model.p")
except:
    try:
        samples = pickle.load(open("cache/user_choices.p", "rb"))
    except:
        # Select random 150? events
        samples = random.sample(list(enumerate(events)), 150)
        # Use hand-sorting to put them least to most favorite, assign a value from -1 to 1 based on that
        samples.sort(cmp=user_sort)
        pickle.dump(samples, open("cache/user_choices.p", "wb"))

    weights = []
    for tuple_tuple in enumerate(samples):
        # Map to (index, [-1, 1])
        weights.append((tuple_tuple[1][0], (float(tuple_tuple[0]) / (len(samples) - 1))))

    X = np.array([ np.append(X_summaries[idx_weight[0]], X_descriptions[idx_weight[0]]) for idx_weight in weights], dtype=np.int32)
    y = np.array([ idx_weight[1] for idx_weight in weights ], dtype=np.float32)

    net.fit(X, y)
    # Cache the parameters and network
    net.save_params_to("cache/calendar_model.p")

# Print the resulting ordering
X_output = np.append(X_summaries, X_descriptions, axis=1)
y_output = net.predict(X_output)

predictions = sorted(zip(events, y_output), key=lambda pair: pair[1], reverse=True)

# Calculate value gained / time taken per item
predicted_value_densities = []
for pair in predictions:
    predicted_value_densities.append((pair[0], pair[1]/((pair[0].decoded('dtend') - pair[0].decoded('dtstart')).total_seconds())))

# TODO: This could work better for pure value or value density... should figure that out

def conflicts(t1start, t1end, t2start, t2end):
    return (t1start <= t2start < t1end) or (t2start <= t1start < t2end)

final_calendar = Calendar()
pvd = sorted(predicted_value_densities, key=lambda pair: pair[1], reverse=True)
# Switch 'predictions' and 'pvd' here to try each method
for pair in predictions:
    # if not conflicting with current calendar
        if not any(conflicts(existing_event.decoded('dtstart'), existing_event.decoded('dtend'), pair[0].decoded('dtstart'), pair[0].decoded('dtend')) for existing_event in final_calendar.walk('vevent')):
            final_calendar.add_component(pair[0])
            print("\"" + pair[0]['summary'] + "\" picked")
        else:
            print("\"" + pair[0]['summary'] + "\" squashed")

with open('../idealized.ics', 'wb') as caloutput:
    caloutput.write(final_calendar.to_ical())

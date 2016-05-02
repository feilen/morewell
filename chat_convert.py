#!/usr/bin/env python2
import json
import re

from glove import Corpus, Glove
import unicodecsv as csv

mods = ['feilen', 'femboybot', 'kentfreimann', 'sphinxie', 'sergalbooty', 'sogak', 'tinyfawks', 'foxyfluff']

# Bundle groups of symbols and make sure words are otherwise alone
print("Parsing JSON log")

logfile = open('After_Dark_Furry_Femboys.jsonl', 'r')
adminfile = open('Fem_Admins.jsonl', 'r')

jsons = []
logs = []
adminlogs = []
for line in logfile.readlines():
    try:
        json_parsed = json.loads(line)
        logs.append(json_parsed['text'])
        jsons.append(json_parsed)
    except KeyError:
        pass

for line in adminfile.readlines():
    try:
        json_parsed = json.loads(line)
        if 'fwd_from' in json_parsed and len(json_parsed['text']) > 7:
            adminlogs.append(json_parsed['text'])
    except KeyError:
        pass

intersects = set.intersection(set(logs), set(adminlogs))

matching_indexes = []
for index, item in enumerate(logs):
    if item in intersects:
        matching_indexes.append(index)

# Flag all matching forwarded messages
for idx in matching_indexes:
    jsons[idx]['flagged'] = '1'

# Create a fast lookup map
id_lookup = dict()
for index, message in enumerate(jsons):
    id_lookup[message['id']] = index

for message in jsons:
    # Flag slurs
    if any( key in message['text'].lower() for key in ['fag', 'bitch', 'shemale', 'boipussy', 'boypussy', 'retard', 'asshole', 'nigg', 'cunt', 'dumbass']):
        message['flagged'] = '1'

    # Flag mod replies with mod names
    try:
        if 'reply_id' in message and 'username' in message['from'] and any( name in message['from']['username'].lower() for name in mods):
            jsons[id_lookup[message['reply_id']]]['flagged'] = message['from']['username']
            print("Labeling mod reply")
    except KeyError:
        pass

for message in jsons:
    # Mods are blameless
    if 'username' in message['from'] and any( key in message['from']['username'].lower() for key in mods):
        message['flagged'] = ''

with open('chat.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile)
    index = 0
    for message in jsons:
        name = (message['from']['first_name'] if 'first_name' in message['from'] else "") + " " + (message['from']['last_name'] if 'last_name' in message['from'] else "")
        spamwriter.writerow(
                    ([message['flagged']] if 'flagged' in message else ['']) +
                    [name] +
                    ([message['from']['username']] if 'username' in message['from'] else ['']) +
                    ([message['text']] if 'text' in message else ['']) +
                    [index]
                    #([message['fwd_from']['first_name'] + " " + message['fwd_from']['last_name']] if 'fwd_from' in message else [''])
                )
        index += 1

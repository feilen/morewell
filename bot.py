#!/usr/bin/env python2

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import numpy as np
from glove import Corpus, Glove
from nolearn.lasagne import NeuralNet
from nn_utils import make_nn, clean, vectify

import logging

max_sentence_length = 22
drop_probability = 0.4
adminchat = -54852278


# Enable logging
logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO)

logger = logging.getLogger(__name__)

glove = Glove()
glove = Glove.load("glove.p")

net = make_nn(max_sentence_length, glove.word_vectors, drop_probability)
net.load_params_from("model.p.gz")


# Define a few command handlers. These usually take the two arguments bot and
# update. Error handlers also receive the raised TelegramError object in error.
def start(bot, update):
    bot.sendMessage(update.message.chat_id, text='Hi!')


def help(bot, update):
    bot.sendMessage(update.message.chat_id, text='Help!')


def filterforward(bot, update):
    if update.message.chat_id != adminchat:
        test = dict()
        result = net.predict(vectify([clean(update.message.text).split()], test, glove.dictionary, max_sentence_length, False))
        if result[0] == 1:
            bot.forwardMessage(chat_id=adminchat, from_chat_id=update.message.chat_id, message_id=update.message.message_id)


def error(bot, update, error):
    logger.warn('Update "%s" caused error "%s"' % (update, error))


def main():
    # Create the EventHandler and pass it your bot's token.
    updater = Updater("194494925:AAEy49BqWvY2SbJogiQvusVCZJtb6jRK860")

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    #dp.addHandler(CommandHandler("start", start))
    #dp.addHandler(CommandHandler("help", help))

    # on noncommand i.e message
    dp.addHandler(MessageHandler([Filters.text], filterforward))

    # log all errors
    dp.addErrorHandler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until the you presses Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()

if __name__ == '__main__':
    main()

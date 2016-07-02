# morewell
A natural language, deep learning bot for automated Telegram chat moderation.

This bot uses Python-Nolearn atop Lasagne, paired with GloVe to allow you to moderate a large Telegram chatroom (or chatrooms) by teaching it what counts as a message it should notify you about. Once the model is trained, it will classify each new message it sees in every chat it moderates as either 'unflagged' or 'flagged', and forward all of the 'flagged' messages to a chat of your choice.

## Usage

First you will need to export your target chatroms to json, using telegram-history-dump. Once you have these, you can generate a converted chat file by feeding it to chat_convert.py, optionally providing a log of the admin chat (which will allow it to see what messages you've already forwarded and preflag them for you). You will want to configure the bot to whatever slurs you normally look into, moderators for it to auto-unflag, etc.

```
./chat_convert.py input_chatroom.jsonl <admin_chatroom.jsonl>
```

This will spit out chat-converted.csv. Bring this into your favorite spreadsheet editor and look for obvious messages to flag. The bot automatically places names in the 'flagged' category of any usernames directly replied to by an admin, to help you know where to look. Once you're satisfied, pass it into chat_optimize.py:

```
./chat_optimize.py chat-flagged.csv
```

The function of chat_optimize is twofold. It first creates a model that the bot.py can use to administrate and forward messages each time you use it. It also, once done, spits out a chat-optimized.csv file, which is the result of applying the current model to the input. This can help you find trickier messages that may have slipped past you, or messages that were accidentally flagged but the bot thinks are innocuous. You can continually smooth out the data by passing it back into chat_optimize as many times as you wish, until you're satisfied with the model's performance.

Once you are finished, you can simply spin up the bot with bot.py:

```
./bot.py
``` 

and you should be in buisness!

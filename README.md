# a conversational application with python

This is an adaptation of [this](https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077) tensorflow chatbot 

The conversational intents needed to be defined. Each conversational intent contains:

* a tag (a unique name)
* patterns (sentence patterns for our neural network text classifier)
* responses (one will be used as a response)
* contextual elements.

The intents file as a json can be viewed [here](https://github.com/aktivkohle/python-talks/blob/master/intents2.json).

The model does not need to be rebuilt unless the intent patterns change. 

The article above mentions the possibility for several hundred intents and thousands of patterns which puts my intents file for this prototype in perspective..

[This](https://github.com/aktivkohle/python-talks/blob/master/display_sample_chat.ipynb) notebook shows how a sample chat went. It was going better before I last modified the intents file. More work needed here.. In addition to working on the intents file it would be worth experimenting with `ERROR_THRESHOLD` in generate_responses.py to see what happens.

Something else in that program is `userID='123'` which suggests this system might easily fit to a web frontend with multiple users.

Sections of the terminal output from the programs generate_responses.py and train_model.py can be found [here](https://github.com/aktivkohle/python-talks/blob/master/terminal_log_samples.txt)

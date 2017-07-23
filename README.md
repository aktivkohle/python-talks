# a conversational application with python

This is an adaptation of [this](https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077) tensorflow chatbot. 

If you want to look straight at a sample chat before reading all this text, **[this notebook](https://github.com/aktivkohle/python-talks/blob/master/display_sample_chat.ipynb)** displays a sample chat.

The conversational intents needed to be defined. Each conversational intent contains:

* a tag (a unique name)
* patterns (sentence patterns for our neural network text classifier)
* responses (one will be used as a response)
* contextual elements.

From the above it is important to note that *only* the patterns go into the classifier, so creating the intents file those fields need the most care. Responses are simply selected randomly so it only matters that it makes sense, the neural network never sees it and tags are just handles, not processed, just for debugging or viewing probabilities, so the programmer can identify them. Importantly however, *unique*.

The intents file as a json can be viewed [here](https://github.com/aktivkohle/python-talks/blob/master/intents2.json).

# Notes on the thought process that went into creating the intents file 

The csv file with it's 25 rows of products needed to be transformed into an intents file so the chatbot can communicate the necessary information in a conversational style. The natural split within the table is the category column, Phones&Tablets, Drones, Gaming&VR, Computing, Wearables, SmartHome. There are however several keyword ways to also split up the data, and given that the text in the *patterns* is going to be training the neural networks, it takes some thought. So 'Apple' is not a unique keyword here, it could be a phone, a notebook or a watch - so deliberately opt to use the keywords 'iPhone', 'MacBook', and 'Apple Watch' - all the exact terms Apple also likes to use. It was also helpful to create a tree diagram around the brand split. Of course if it were a human not a chatbot, or a much more sophisticated chatbot, this kind of discipline with the intents file would not be necessary, but to make it work within the confines of this model it has to be that way. Be careful not to use ambiguous keywords in the text. If the table were a lot bigger, there would need to be several intents to deal with ambigious responses, eg just the work 'polar' if there were several polar watches.

'Samsung' for example will be used only for the phones while the vacuum cleaner will always be called a POWERbot. There are plenty of rather unique keywords in there like 'Polar' which will always be the watch here. 

Actually, although the responses are not in any way being processed, their content will affect the language that the human interacting with the chatbot uses. We don't want to encourage the user to use language that would confuse the chatbot, so it is again important to restrict the responses to the relevant keywords, eg Samsung for the phone and POWERbot for the vacuum cleaner.

The model does not need to be rebuilt unless the intent patterns change. 

The article above mentions the possibility for several hundred intents and thousands of patterns which puts my intents file for this prototype in perspective.
 
In addition to working on the intents file it would be worth experimenting with `ERROR_THRESHOLD` in generate_responses.py to see what happens.

Something else in that program is `userID='123'` which suggests this system might easily fit to a web frontend with multiple users.

Sections of the terminal output from the programs generate_responses.py and train_model.py can be found [here](https://github.com/aktivkohle/python-talks/blob/master/terminal_log_samples.txt)



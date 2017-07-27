# a conversational application with python

This is an adaptation of [this](https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077) tensorflow chatbot. An interesting [article](http://pavel.surmenok.com/2016/09/11/chatbot-architecture/) about chatbot architecture splits them into the categories of business or entertainment and generative models or retrieval based, this is a business one with a retrieval based model. Retrieval-based is easier to make and there is the advantage that grammar will always be right in the responses and a huge training corpus is not required. 

If you want to look straight at a sample chat before reading all this text, **[this notebook](https://github.com/aktivkohle/python-talks/blob/master/display_sample_chat.ipynb)** displays a sample chat. It includes feedback from the model so a developer can see what is going on.

The conversational intents needed to be defined. Each conversational intent contains:

* a tag (a unique name)
* patterns (sentence patterns for our neural network text classifier)
* responses (one will be used as a response)
* contextual elements.

From the above it is important to note that *only* the patterns go into the classifier, so creating the intents file those fields need the most care. Responses are simply selected randomly so it only matters that it makes sense, the neural network never sees it and tags are just handles, not processed, just for debugging or viewing probabilities, so the programmer can identify them. Importantly however, *unique*.

The intents file from the latest iteration can be viewed [here](https://github.com/aktivkohle/python-talks/blob/master/intents3_NO_CONTEXTS_MINIMALIST.json).

### notes on the intents file

The csv file with its 25 rows of products needed to be transformed into an intents file so the chatbot can communicate the necessary information in a conversational style. The natural split within the table is the category column, Phones&Tablets, Drones, Gaming&VR, Computing, Wearables, SmartHome. There are however several keyword ways to also split up the data, and given that the text in the *patterns* is going to be training the neural networks, it takes some thought. So 'Apple' is not a unique keyword here, it could be a phone, a notebook or a watch - so deliberately opt to use the keywords 'iPhone', 'MacBook', and 'Apple Watch' - all the exact terms Apple also likes to use. It was also helpful to create a tree diagram around the brand split. Of course if it were a human not a chatbot, or a much more sophisticated chatbot, this kind of discipline with the intents file would not be necessary, but to make it work within the confines of this model it has to be that way. Be careful not to use ambiguous keywords in the text. If the table were a lot bigger, there would need to be several intents to deal with ambigious responses, eg just the word 'polar' if there were several polar watches.

'Samsung' for example will be used only for the phones while the vacuum cleaner will always be called a POWERbot. There are plenty of rather unique keywords in there like 'Polar' which will always be the watch here. 

Actually, although the responses are not in any way being processed, their content will affect the language that the human interacting with the chatbot uses. We don't want to encourage the user to use language that would confuse the chatbot, so it is again important to restrict the responses to the relevant keywords, eg Samsung for the phone and POWERbot for the vacuum cleaner.

The model does not need to be rebuilt unless the intent patterns change. The article above mentions the possibility for several hundred intents and thousands of patterns which puts my intents file for this prototype in perspective.

### very big effect by removing stopwords on NN loss and accuracy

I haven't graphed the loss but could see it coming down but generally waving around. Accuracy went up but while loss started at about 2.8 and went down to 1.4, it was waving around so much in the middle it was not what you want to see. Accuracy had moved from 0.02 about 0.95 but was also moving around. [This stackoverflow answer](https://stackoverflow.com/questions/40910857/how-to-interpret-increase-in-both-loss-and-accuracy) communicates concisely how to interpret the various behaviours with loss and accuracy. After the three lines of code that removes stopwords were put in, the loss went right down to 0.04 and the accuracy was up at 0.9996 even 1.000 was seen, and was not waving around much. 

Nevertheless this stopwords measure introduced another problem: 

> Anyone there?

> We have a range of MacBooks, a Microsoft Surface or a Lenovo Yoga. Which kind would you like? 

The word "anyone" was within the stopwords so have to manually put that one back in. 'how' had to also go back in but not 'you' as that will ruin the results, it's there too many times in the intents file. 

### an iterative process
As a human with a knowledge of the language and style of conversing ideas to improve the intents file occur the more you interact with the chatbot. 

### keep it stateless
I removed all the contexts after the first attempt as it there is not really any need with this design to lock the chat into one track as there was with the original mopeds example *Do you want a moped today? Then here is our number*. Then made it more minimal, reducing the patterns more to keywords. Latest update is that the word 'how' has to go from everywhere except the greeting since this happened:

> How about something which flies?

> Hi there, how can I help?

> [('greeting', 0.50350749), ('wantsDrone', 0.48901734)] 

That is not good, it should be wantsDrone 0.90, that's why here for example, 'how' needed to be cleansed out of every part of the intents file except for the greeting. It's only running off bag of words, no sequences or character level things.. 

### sensible response when probabilities are low
After chatting with it for a while and pasting in some irrelevant texts could see that 0.60 would be a good cutoff for whether it has understood or not. I added some code so that the chatbot tells the user it cannot understand if no tag has a high enough probability. That way it reduces the chance of it giving answers that are completely out of tune with what the user said and is also more polite to interact with. 

from files import *
# things we need for NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# things we need for Tensorflow
import numpy as np


            
def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

# define a function which replaces lower for the lists of tuples and single words 
# resulting from of the ngrams:

def flexible_lower(textlike):
    if isinstance(textlike, str):
        return textlike.lower()
    elif isinstance(textlike, tuple):
        # turn tuple briefly into a list to run a list comprehension on it
        textlike_list = list(textlike)
        textlike_list = [e.lower() for e in textlike_list]
        return tuple(textlike_list)    
    

if __name__ == "__main__":    
    import tflearn
    import tensorflow as tf
    import random
    import pickle
    import json
    from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
    from nltk.corpus import stopwords
    STOPLIST = set(stopwords.words('english') + ["n't", "'s", "'m", "ca"] + list(ENGLISH_STOP_WORDS))

    # import our chat-bot intents file
    with open(INTENTSFILE) as json_data:
        intents = json.load(json_data)
        
    words = []
    classes = []
    documents = []
    ignore_words = list(STOPLIST)
    ignore_words.remove('anyone')
    ignore_words.remove('how')   # otherwise the greetings 'anyone there' and 'how are you' are not recognised


    # loop through each sentence in our intents patterns
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # tokenize each word in the sentence
            token = nltk.word_tokenize(pattern)
            twograms = nltk.bigrams(token)
            threegrams = nltk.trigrams(token)
            # ngrams includes 1-gram, 2-gram and 3-grams
            ngrams = token + list(twograms) + list(threegrams)
            # add to our words list
            words.extend(ngrams)
            # add to documents in our corpus
            documents.append((ngrams, intent['tag']))
            # add to our classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # stem and lower each word and remove duplicates
    words = [stemmer.stem(w.lower()) for w in words if w.lower() not in ignore_words]
    words = list(set(words))

    # remove duplicates
    classes = list(set(classes))

    print (len(documents), "documents")
    print (len(classes), "classes", classes)
    print (len(words), "unique stemmed words", words)   



    # create our training data
    training = []
    output = []
    # create an empty array for our output
    output_empty = [0] * len(classes)

    # training set, bag of words for each sentence
    for doc in documents:
        # initialize our bag of words
        bag = []
        # list of tokenized words for the pattern
        pattern_words = doc[0]
        # stem each word
        pattern_words = [stemmer.stem(word.lower()) for word in pattern_words if word.lower() not in ignore_words]
        # create our bag of words array
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        # output is a '0' for each tag and '1' for current tag
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append([bag, output_row])

    # shuffle our features and turn into np.array
    random.shuffle(training)
    training = np.array(training)

    # create train and test lists
    train_x = list(training[:,0])
    train_y = list(training[:,1])


    # reset underlying graph data
    tf.reset_default_graph()
    # Build neural network
    net = tflearn.input_data(shape=[None, len(train_x[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
    net = tflearn.regression(net)

    # Define model and setup tensorboard
    model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
    # Start training (apply gradient descent algorithm)
    model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
    model.save(MODELFILES)

    pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( TRAININGDATAFILE, "wb" ) )
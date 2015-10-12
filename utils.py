#! /usr/bin/env python

import csv
import itertools
import numpy as np
import nltk
import time
import sys
from datetime import datetime
from lstm_theano import LSTMTheano

sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
unknown_token = "UNKNOWN_TOKEN"
word_to_index = []
index_to_word = []


def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

def load_and_proprocess_data(vocabulary_size):

    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print "Reading CSV file..."
    with open('data/reddit-comments-2015-08.csv', 'rb') as f:
        reader = csv.reader(f, skipinitialspace=True)
        reader.next()
        # Split full comments into sentences
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print "Parsed %d sentences." % (len(sentences))

    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print "Found %d unique words tokens." % len(word_freq.items())

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size-1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    print "Using vocabulary size %d." % vocabulary_size
    print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

    return X_train, y_train, word_to_index, index_to_word


def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5, save_every=None):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (now, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5    
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
        # Optionally save model parameters
        if (save_every and epoch % save_every == 0):
          filename = "./data/lstm-%d-%d-%d.npz" % (model.hidden_dim, model.word_dim, epoch)
          save_model_parameters_theano(filename, model)
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1
    return losses

def generate_sentence(model):
    # We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str


def save_model_parameters_theano(outfile, model):
    U_i, U_f, U_o, U_g, W_i, W_f, W_o, W_g, V, b_i, b_f, b_o, b_g, b_V = [
        model.U_i.get_value(), model.U_f.get_value(), model.U_o.get_value(), model.U_g.get_value(),
        model.W_i.get_value(), model.W_f.get_value(), model.W_o.get_value(), model.W_g.get_value(),
        model.V.get_value(),
        model.b_i.get_value(), model.b_f.get_value(), model.b_o.get_value(), model.b_g.get_value(), model.b_V.get_value()]
    np.savez(outfile, U_i=U_i, U_f=U_f, U_o=U_o, U_g=U_g,
            W_i=W_i, W_f=W_f, W_o=W_o, W_g=W_g,
            V=V, b_V=b_V,
            b_i=b_i, b_f=b_f, b_o=b_o, b_g=b_g)
    print "Saved model parameters to %s." % outfile

def load_model_parameters_theano(path):
    npzfile = np.load(path)
    U_i, U_f, U_o, U_g = npzfile["U_i"], npzfile["U_f"], npzfile["U_o"], npzfile["U_g"]
    W_i, W_f, W_o, W_g = npzfile["W_i"], npzfile["W_f"], npzfile["W_o"], npzfile["W_g"]
    b_i, b_f, b_o, b_g = npzfile["b_i"], npzfile["b_f"], npzfile["b_o"], npzfile["b_g"]
    V, b_V = npzfile["V"], npzfile["b_V"]
    hidden_dim, word_dim = U_i.shape[0], U_i.shape[1]
    print "Building model model from %s with hidden_dim=%d word_dim=%d" % (path, hidden_dim, word_dim)
    sys.stdout.flush()
    model = LSTMTheano(word_dim, hidden_dim=hidden_dim)
    model.U_i.set_value(U_i); model.U_f.set_value(U_f); model.U_o.set_value(U_o); model.U_g.set_value(U_g)
    model.W_i.set_value(W_i); model.W_f.set_value(W_f); model.W_o.set_value(W_o); model.W_g.set_value(W_g)
    model.b_i.set_value(b_i); model.b_f.set_value(b_f); model.b_o.set_value(b_o); model.b_g.set_value(b_g)
    model.V.set_value(V); model.b_V.set_value(b_V)
    return model    

# EOF
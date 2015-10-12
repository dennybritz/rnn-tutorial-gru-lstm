#! /usr/bin/env python

import sys
import os
import time
from utils import *
from lstm_theano import LSTMTheano

VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '800'))
HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '80'))
LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
NEPOCH = int(os.environ.get('NEPOCH', '100'))
MODEL_FILE = os.environ.get('MODEL_FILE')

# Load and pre-process data
X_train, y_train, word_to_index, index_to_word = load_and_proprocess_data(VOCABULARY_SIZE)

# Create or load model instance
if MODEL_FILE != None:
    model = load_model_parameters_theano(MODEL_FILE)
else:
    model = LSTMTheano(VOCABULARY_SIZE, hidden_dim=HIDDEN_DIM)

# Print SGD step time
t1 = time.time()
model.sgd_step(X_train[10], y_train[10], 0.005)
t2 = time.time()
print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)
sys.stdout.flush()

# Train model
train_with_sgd(model, X_train[:100], y_train[:100], nepoch=NEPOCH, learning_rate=LEARNING_RATE,
  evaluate_loss_after=5, save_every=5)
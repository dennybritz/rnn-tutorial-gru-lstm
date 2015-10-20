#! /usr/bin/env python

import sys
import os
import time
from utils import *
from lstm_theano import LSTMTheano
from gru_theano import *

VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '8000'))
HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '100'))
LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.00001'))
REG_LAMBDA = float(os.environ.get('REG_LAMBDA', '0'))
DECAY = float(os.environ.get('DECAY', '0.99'))
NEPOCH = int(os.environ.get('NEPOCH', '100'))
LOSS_SUBSAMPLE = int(os.environ.get('LOSS_SUBSAMPLE', '8000'))
GLOVE_FILE = os.environ.get('GLOVE_FILE')
MODEL_FILE = os.environ.get('MODEL_FILE')

# Load and pre-process data
X_train, y_train, word_to_index, index_to_word = load_and_proprocess_data(VOCABULARY_SIZE)
wv = None
if GLOVE_FILE:
  wv = construct_wv_for_vocabulary(GLOVE_FILE, index_to_word)

# Create or load model instance
if MODEL_FILE != None:
    model = load_model_parameters_theano(MODEL_FILE)
else:
    model = GRUTheano(VOCABULARY_SIZE, hidden_dim=HIDDEN_DIM, reg_lambda=REG_LAMBDA, wordvec=wv, bptt_truncate=8)

# Print SGD step time
t1 = time.time()
model.sgd_step(X_train[10], y_train[10], LEARNING_RATE)
t2 = time.time()
print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)
sys.stdout.flush()

# Train model
train_with_sgd(model, X_train, y_train, nepoch=NEPOCH, learning_rate=LEARNING_RATE,
  evaluate_loss_after=1, subsample_loss=LOSS_SUBSAMPLE, save_every=1, decay=DECAY)
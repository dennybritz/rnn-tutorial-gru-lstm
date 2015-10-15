import numpy as np
import theano as theano
import theano.tensor as T
from theano.gradient import grad_clip
import time
import operator

class GRUTheano:
    
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (3, hidden_dim, word_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (3, hidden_dim, hidden_dim))
        b = np.zeros((3, hidden_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        b2 = np.zeros(word_dim)
        # Theano: Created shared variables
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        # Bias terms
        self.b = theano.shared(name='b_i', value=b.astype(theano.config.floatX))
        self.b2 = theano.shared(name='b_V', value=b2.astype(theano.config.floatX))
        # SGD: Initialize parameters
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.mb2 = theano.shared(name='mb2', value=np.zeros(b2.shape).astype(theano.config.floatX))
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()
    
    def __theano_build__(self):
        V, U, W, b, b2 = self.V, self.U, self.W, self.b, self.b2
        mV, mU, mW, mb, mb2 = self.mV, self.mU, self.mW, self.mb, self.mb2
        
        x = T.ivector('x')
        y = T.ivector('y')
        
        def forward_prop_step(x_t, s_t_prev):
            # This is how we calculated the hidden state in a simple RNN. No longer!
            # s_t = T.tanh(U[:,x_t] + W.dot(s_t_prev))
            
            # Clip the gradients
            W_clipped = grad_clip(W, -1, 1)
            U_clipped = grad_clip(U, -1, 1)
            V_clipped = grad_clip(V, -1, 1)
            b_clipped = grad_clip(b, -1, 1)
            b2_clipped = grad_clip(b2, -1, 1)
            
            # LRU hidden state calculation
            z_t = T.nnet.sigmoid(U_clipped[0][:,x_t] + W_clipped[0].dot(s_t_prev) + b_clipped[0])
            r_t = T.nnet.sigmoid(U_clipped[1][:,x_t] + W_clipped[1].dot(s_t_prev) + b_clipped[1])
            c_t = T.tanh(U_clipped[2][:,x_t] + W_clipped[2].dot(s_t_prev) * r_t + b_clipped[2])
            s_t = (1 - z_t) * c_t + z_t * s_t_prev
              
            # Final output calculation
            # Theano's softmax returns a matrix with one row, we only need the row
            o_t = T.nnet.softmax(V_clipped.dot(s_t) + b2_clipped)[0]

            return [o_t, s_t]
        
        [o,s], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            outputs_info=[None, dict(initial=T.zeros(self.hidden_dim))],
            truncate_gradient=self.bptt_truncate)
        
        prediction = T.argmax(o, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))
        
        # Gradients
        dU = T.grad(o_error, U)
        dW = T.grad(o_error, W)
        db = T.grad(o_error, b)
        dV = T.grad(o_error, V)
        db2 = T.grad(o_error, b2)
        
        # Assign functions
        self.forward_propagation = theano.function([x], o)
        self.predict = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], o_error)
        self.bptt = theano.function([x, y], [dU, dW, db, dV, db2])
        
        # SGD with Momentum
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')
        self.sgd_step = theano.function(
            [x, y, learning_rate, theano.Param(decay, default=0.9)],
            [], 
            updates=[(U, U - learning_rate * dU / T.sqrt(mU + 1e-8)),                     
                     (W, W - learning_rate * dW / T.sqrt(mW + 1e-8)),
                     (V, V - learning_rate * dV / T.sqrt(mV + 1e-8)),
                     (b, b - learning_rate * db / T.sqrt(mb + 1e-8)),
                     (b2, b2 - learning_rate * db2 / T.sqrt(mb2 + 1e-8)),
                     (mU, decay * mU + (1 - decay) * T.sqr(dU)),
                     (mW, decay * mW + (1 - decay) * T.sqr(dW)),
                     (mV, decay * mV + (1 - decay) * T.sqr(dV)),
                     (mb, decay * mb + (1 - decay) * T.sqr(db)),
                     (mb2, decay * mb2 + (1 - decay) * T.sqr(db2))
                    ])
    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])
    
    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)


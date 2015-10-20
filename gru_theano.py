import numpy as np
import theano as theano
import theano.tensor as T
from theano.gradient import grad_clip
import time
import operator

class GRUTheano:
    
    def __init__(self, word_dim, hidden_dim=100, reg_lambda=0, wordvec=None, bptt_truncate=-1):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.reg_lambda = reg_lambda
        self.bptt_truncate = bptt_truncate
        # Initialize the network parameters
        if wordvec != None:
            U = np.array([wordvec.T, wordvec.T, wordvec.T])
        else:
            U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (3, hidden_dim, word_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (9, hidden_dim, hidden_dim))
        b = np.zeros((6, hidden_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        
        c = np.zeros(word_dim)
        # Theano: Created shared variables
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        # Bias terms
        self.b = theano.shared(name='b_i', value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))
        # SGD: Initialize parameters
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()
    
    def __theano_build__(self):
        V, U, W, b, c = self.V, self.U, self.W, self.b, self.c
        
        x = T.ivector('x')
        y = T.ivector('y')
        
        def forward_prop_step(x_t, s_t_prev, s_t2_prev):
            # This is how we calculated the hidden state in a simple RNN. No longer!
            # s_t = T.tanh(U[:,x_t] + W.dot(s_t_prev))
            
            # Clip the gradients
            W_clipped = grad_clip(W, -1, 1)
            U_clipped = grad_clip(U, -1, 1)
            V_clipped = grad_clip(V, -1, 1)
            b_clipped = grad_clip(b, -1, 1)
            c_clipped = grad_clip(c, -1, 1)
            
            # Layer 1
            z_t = T.nnet.sigmoid(U_clipped[0][:,x_t] + W_clipped[0].dot(s_t_prev) + b_clipped[0])
            r_t = T.nnet.sigmoid(U_clipped[1][:,x_t] + W_clipped[1].dot(s_t_prev) + b_clipped[1])
            c_t = T.tanh(U_clipped[2][:,x_t] + W_clipped[2].dot(s_t_prev) * r_t + b_clipped[2])
            s_t = (1 - z_t) * c_t + z_t * s_t_prev
            
            # Layer 2
            z_t2 = T.nnet.sigmoid(W_clipped[3].dot(s_t) + W_clipped[6].dot(s_t2_prev) + b_clipped[3])
            r_t2 = T.nnet.sigmoid(W_clipped[4].dot(s_t) + W_clipped[7].dot(s_t2_prev) + b_clipped[4])
            c_t2 = T.tanh(W_clipped[5].dot(s_t) + W_clipped[8].dot(s_t2_prev) * r_t2 + b_clipped[5])
            s_t2 = (1 - z_t2) * c_t2 + z_t2 * s_t2_prev            
              
            # Final output calculation
            # Theano's softmax returns a matrix with one row, we only need the row
            o_t = T.nnet.softmax(V_clipped.dot(s_t2) + c_clipped)[0]

            return [o_t, s_t, s_t2]
        
        [o,s,s2], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            truncate_gradient=self.bptt_truncate,
            outputs_info=[None, dict(initial=T.zeros(self.hidden_dim)), dict(initial=T.zeros(self.hidden_dim))])
        
        prediction = T.argmax(o, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))
        
        # Regularization cost
        reg_cost = self.reg_lambda/2. * \
            (T.sum(T.sqr(V)) + T.sum(T.sqr(U)) + T.sum(T.sqr(W)) + T.sum(T.sqr(b)) + T.sum(T.sqr(c)))
        # Total cost
        cost = o_error + reg_cost
        
        # Gradients
        dU = T.grad(cost, U)
        dW = T.grad(cost, W)
        db = T.grad(cost, b)
        dV = T.grad(cost, V)
        dc = T.grad(cost, c)
        
        # Assign functions
        self.forward_propagation = theano.function([x], o)
        self.predict = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], cost)
        self.bptt = theano.function([x, y], [dU, dW, db, dV, dc])
        
        # SGD parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')
        
        # rmsprop cache updates
        mU = decay * self.mU + (1 - decay) * T.sqr(dU)
        mW = decay * self.mW + (1 - decay) * T.sqr(dW)
        mV = decay * self.mV + (1 - decay) * T.sqr(dV)
        mb = decay * self.mb + (1 - decay) * T.sqr(db)
        mc = decay * self.mc + (1 - decay) * T.sqr(dc)
        
        self.sgd_step = theano.function(
            [x, y, learning_rate, theano.Param(decay, default=0.99)],
            [], 
            updates=[(U, U - learning_rate * dU / T.sqrt(mU + 1e-8)),                     
                     (W, W - learning_rate * dW / T.sqrt(mW + 1e-8)),
                     (V, V - learning_rate * dV / T.sqrt(mV + 1e-8)),
                     (b, b - learning_rate * db / T.sqrt(mb + 1e-8)),
                     (c, c - learning_rate * dc / T.sqrt(mc + 1e-8)),
                     (self.mU, mU),
                     (self.mW, mW),
                     (self.mV, mV),
                     (self.mb, mb),
                     (self.mc, mc)
                    ])
    
    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])
    
    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)


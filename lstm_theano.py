import numpy as np
import theano as theano
import theano.tensor as T
import time
import operator

class LSTMTheano:
    
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        U_i = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        U_f = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        U_o = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        U_g = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        W_i = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        W_f = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        W_o = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        W_g = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        # Theano: Created shared variables
        self.U_i = theano.shared(name='U_i', value=U_i.astype(theano.config.floatX))
        self.U_f = theano.shared(name='U_f', value=U_f.astype(theano.config.floatX))
        self.U_o = theano.shared(name='U_o', value=U_o.astype(theano.config.floatX))
        self.U_g = theano.shared(name='U_g', value=U_g.astype(theano.config.floatX))
        self.W_i = theano.shared(name='W_i', value=W_i.astype(theano.config.floatX))
        self.W_f = theano.shared(name='W_f', value=W_f.astype(theano.config.floatX))
        self.W_o = theano.shared(name='W_o', value=W_o.astype(theano.config.floatX))
        self.W_g = theano.shared(name='W_g', value=W_g.astype(theano.config.floatX))        
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        # Bias terms
        self.b_i = theano.shared(name='b_i', value=np.zeros(hidden_dim).astype(theano.config.floatX))
        self.b_f = theano.shared(name='b_f', value=np.ones(hidden_dim).astype(theano.config.floatX))
        self.b_o = theano.shared(name='b_o', value=np.zeros(hidden_dim).astype(theano.config.floatX))
        self.b_g = theano.shared(name='b_g', value=np.zeros(hidden_dim).astype(theano.config.floatX))
        self.b_V = theano.shared(name='b_V', value=np.zeros(word_dim).astype(theano.config.floatX))        
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()
    
    def __theano_build__(self):
        V = self.V
        U_i, U_f, U_o, U_g = self.U_i, self.U_f, self.U_o, self.U_g
        W_i, W_f, W_o, W_g = self.W_i, self.W_f, self.W_o, self.W_g
        b_i, b_f, b_o, b_g, b_V = self.b_i, self.b_f, self.b_o, self.b_g, self.b_V
        x = T.ivector('x')
        y = T.ivector('y')
        
        def forward_prop_step(x_t, c_t_prev, s_t_prev):
            # This is how we calculated the hidden state in a simple RNN. No longer!
            # s_t = T.tanh(U[:,x_t] + W.dot(s_t_prev))
            
            # LSTM hidden state calculation
            i_t = T.nnet.sigmoid(U_i[:,x_t] + W_i.dot(s_t_prev) + b_i)
            f_t = T.nnet.sigmoid(U_f[:,x_t] + W_f.dot(s_t_prev) + b_f)
            o_t = T.nnet.sigmoid(U_o[:,x_t] + W_o.dot(s_t_prev) + b_o)
            g_t = T.tanh(U_g[:,x_t] + W_g.dot(s_t_prev) + b_g)
            c_t = c_t_prev * f_t + g_t * i_t
            s_t = T.tanh(c_t) * o_t
            
            # Final output calculation
            o_t = T.nnet.softmax(V.dot(s_t) + b_V)
            
            return [o_t[0], c_t, s_t]
        
        [o,c,s], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            outputs_info=[None, dict(initial=T.zeros(self.hidden_dim)), dict(initial=T.zeros(self.hidden_dim))],
            truncate_gradient=self.bptt_truncate)
        
        prediction = T.argmax(o, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))
        
        # Gradients
        dU_i, dU_f, dU_o, dU_g = T.grad(o_error, U_i), T.grad(o_error, U_f), T.grad(o_error, U_o), T.grad(o_error, U_g)
        dW_i, dW_f, dW_o, dW_g = T.grad(o_error, W_i), T.grad(o_error, W_f), T.grad(o_error, W_o), T.grad(o_error, W_g)
        db_i, db_f, db_o, db_g, db_V = T.grad(o_error, b_i), T.grad(o_error, b_f), T.grad(o_error, b_o), T.grad(o_error, b_g), T.grad(o_error, b_V)
        dV = T.grad(o_error, V)
        
        # Assign functions
        self.forward_propagation = theano.function([x], o)
        self.predict = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], o_error)
        self.bptt = theano.function([x, y], [dU_i, dU_f, dU_o, dU_g, dW_i, dW_f, dW_o, dW_g, dV, db_i, db_f, db_o, db_g, db_V])
        
        # SGD
        learning_rate = T.scalar('learning_rate')
        self.sgd_step = theano.function([x,y,learning_rate], [], 
                      updates=[(self.U_i, self.U_i - learning_rate * dU_i),
                               (self.U_f, self.U_f - learning_rate * dU_f),
                               (self.U_o, self.U_o - learning_rate * dU_o),
                               (self.U_g, self.U_g - learning_rate * dU_g),
                               (self.W_i, self.W_i - learning_rate * dW_i),
                               (self.W_f, self.W_f - learning_rate * dW_f),
                               (self.W_o, self.W_o - learning_rate * dW_o),
                               (self.W_g, self.W_g - learning_rate * dW_g),                              
                               (self.V, self.V - learning_rate * dV),
                               (self.b_i, self.b_i - learning_rate * db_i),
                               (self.b_f, self.b_f - learning_rate * db_f),
                               (self.b_o, self.b_o - learning_rate * db_o),
                               (self.b_g, self.b_g - learning_rate * db_g),
                               (self.b_V, self.b_V - learning_rate * db_V),
                              ])
    
    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])
    
    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words) 


def gradient_check_theano(model, x, y, h=0.001, error_threshold=0.01):
    # Overwrite the bptt attribute. We need to backpropagate all the way to get the correct gradient
    model.bptt_truncate = 1000
    # Calculate the gradients using backprop
    bptt_gradients = model.bptt(x, y)
    # List of all parameters we want to chec.
    model_parameters = ['U_i', 'U_f', 'U_o', 'U_g', 'W_i', 'W_f', 'W_o', 'W_g', 'V', 'b_i', 'b_f', 'b_o', 'b_g', 'b_V']
    # Gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        # Get the actual parameter value from the mode, e.g. model.W
        parameter_T = operator.attrgetter(pname)(model)
        parameter = parameter_T.get_value()
        print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
        # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # Save the original value so we can reset it later
            original_value = parameter[ix]
            # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
            parameter[ix] = original_value + h
            parameter_T.set_value(parameter)
            gradplus = model.calculate_total_loss([x],[y])
            parameter[ix] = original_value - h
            parameter_T.set_value(parameter)
            gradminus = model.calculate_total_loss([x],[y])
            estimated_gradient = (gradplus - gradminus)/(2*h)
            parameter[ix] = original_value
            parameter_T.set_value(parameter)
            # The gradient for this parameter calculated using backpropagation
            backprop_gradient = bptt_gradients[pidx][ix]
            # calculate The relative error: (|x - y|/(|x| + |y|))
            relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
            # If the error is to large fail the gradient check
            if relative_error > error_threshold:
                print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                print "+h Loss: %f" % gradplus
                print "-h Loss: %f" % gradminus
                print "Estimated_gradient: %f" % estimated_gradient
                print "Backpropagation gradient: %f" % backprop_gradient
                print "Relative Error: %f" % relative_error
                return 
            it.iternext()
        print "Gradient check for parameter %s passed." % (pname)

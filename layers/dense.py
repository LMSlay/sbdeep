
import theano.tensor as T
from util.nonlinearities import *
from util.init import *

__all__ = ["DenseLayer"]



class DenseLayer(object):


    def __init__(self, n_in, n_out, name,
                 W_init=Glorot, b=None, activation=tanh, learning_rate=0.001):


        self.input = input

        W = W_init(num_input=n_in, num_unit=n_out, name=name+"W")

        if b==None:
            b = theano.shared(
                value=numpy.zeros((n_out,), dtype=theano.config.floatX),
                name=name+"b",
                borrow=True
            )

        self.W = W
        self.b = b
        self.activation = activation
        self.learning_rate=learning_rate


        self.params = [self.W,self.b]

    def get_value(self, input):

        return self.activation(T.dot(input,self.W)+self.b)

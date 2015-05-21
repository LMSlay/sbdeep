import numpy
import theano
import theano.tensor as T

__all__ = ["DenseLayer"]



class DenseLayer(object):


    def __init__(self,input,n_in,n_out,W=None,b=None,activation=T.tanh):


        self.input = input


        if W==None:
            W_values = numpy.asarray(
                numpy.random.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in,n_out)),
                dtype=theano.config.floatX,
                )
            W = theano.shared(value=W_values,name="W",borrow=True)

        if b==None:
            b = theano.shared(
                value=numpy.zeros((n_out,), dtype=theano.config.floatX),
                name="b",
                borrow=True
            )

        self.W = W
        self.b = b

        dot_comput = T.dot(input,self.W)+self.b
        self.output = activation(dot_comput)

        self.params = [self.W,self.b]

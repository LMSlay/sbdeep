import numpy
import theano


class Glorot(object):


    def __init__(self,num_input,num_unit):

        param_values = numpy.asarray(
            numpy.random.uniform(
                low=-numpy.sqrt(6. / (num_input + num_unit)),
                high=numpy.sqrt(6. / (num_input + num_unit)),
                size=(num_input,num_unit)),
            dtype=theano.config.floatX,
            )

        self.params = theano.shared(value=param_values,name="W",borrow=True)
import numpy
import theano

def Glorot(num_input, num_unit ,name):

    param_values = numpy.asarray(
        numpy.random.uniform(
            low=-numpy.sqrt(6. / (num_input + num_unit)),
            high=numpy.sqrt(6. / (num_input + num_unit)),
            size=(num_input,num_unit)),
        dtype=theano.config.floatX,
        )

    return theano.shared(value=param_values, name=name, borrow=True)

def PreDefine(num_input, num_unit, name):
    pass



def defa(num_input, num_unit, name):

    param_values = numpy.asarray(
        numpy.random.uniform(size=(num_input, num_unit)),
        dtype=theano.config.floatX,
        )

    return theano.shared(value=param_values, name=name, borrow=True)


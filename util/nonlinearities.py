
import theano.tensor.nnet


#ref https://github.com/Lasagne/Lasagne/blob/master/lasagne/nonlinearities.py


def sigmoid(x):

    return theano.tensor.nnet.sigmoid(x)


# softmax (row-wise)
def softmax(x):

    return theano.tensor.nnet.softmax(x)


# tanh
def tanh(x):

    return theano.tensor.tanh(x)


# rectify
# https://github.com/Lasagne/Lasagne/pull/163#issuecomment-81765117
def rectify(x):

    return 0.5 * (x + abs(x))

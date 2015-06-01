import model

__author__ = 'slay'

from model import Classification

# sbdeep comp

from layers import dense
from util.nonlinearities import *
from util.init import *

# for test

import dutil

#theano.config.optimizer='None'


def test():

    #some data

    X_train, X_test, y_train, y_test, index_train, index_test = dutil.load_titanic()

    X_train = X_train.astype(numpy.float64)

    y_train = y_train.reshape(1,y_train.shape[0])[0].astype(numpy.int32)

    X_test = X_test.astype(numpy.float64)

    y_test = y_test.reshape(1,y_test.shape[0])[0].astype(numpy.int32)

    #train

    model = Classification()

    model.add(dense.DenseLayer(7, 20, name="hiddenLayer"))

    model.add(dense.DenseLayer(20, 2, name="outputLayer", W_init=defa, activation=softmax))

    model.fit(X_train, X_test, y_train, y_test,)


if __name__ == "__main__":
    test()
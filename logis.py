__author__ = 'slay'

import theano
import theano.tensor as T

import numpy

# sbdeep comp

from util import stopping
from layers import dense
from util.nonlinearities import *

# for test
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix,matthews_corrcoef
from sklearn import svm

import dutil

#theano.config.optimizer='None'

class LogisticRegression(object):

    def __init__(self, input, n_in, n_out,  W=None, b=None, activation=softmax):

        self.input = input

        self.W = theano.shared(
            value=numpy.random.uniform(size=(n_in,n_out)),
            name="w"
        )

        self.b = theano.shared(
            value=numpy.zeros((n_out,))
        )

        dot_comput = T.dot(input,self.W)+self.b
        self.output = activation(dot_comput)

        self.params = [self.W,self.b]




class MLP(object):


    def __init__(self,input,n_in,n_out):


        self.hiddenlayer = dense.DenseLayer(
            input=input,
            n_in=n_in,
            n_out=n_out,
            )

        self.outputlayer = LogisticRegression(
            input=self.hiddenlayer.output,
            n_in=n_out,
            n_out=2
        )



        self.output = self.outputlayer.output

        self.params = self.hiddenlayer.params + self.outputlayer.params

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.output)[T.arange(y.shape[0]), y])

def main():

    X_train, X_test, y_train, y_test, index_train, index_test = dutil.load_titanic()

    x = T.matrix("x")
    y = T.ivector('y')

    other_data = make_classification()

    learning_rate = 0.1

    classifier = MLP(
        input=x,
        n_in=7,
        n_out=20
    )

    cost = classifier.negative_log_likelihood(y)

    prediction = T.argmax(classifier.output, axis=1)

    gparams = [T.grad(cost, param) for param in classifier.params]

    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    test_model = theano.function(
        inputs=[x,y],
        outputs=cost,
    )

    train_model = theano.function(
        inputs=[x,y],
        outputs=cost,
        updates=updates,
    )

    weigth = range(len(classifier.params))

    stoping_condition = False



    best_validation_loss = 100

    iter = 0

    train_set_error = []
    vail_set_error = []

    while not stoping_condition:

        iter +=1

        train_error =  train_model(X_train.astype(numpy.float64),y_train.reshape(1,y_train.shape[0])[0].astype(numpy.int32))

        vail_error = test_model(X_test.astype(numpy.float64),y_test.reshape(1,y_test.shape[0])[0].astype(numpy.int32))

        print "iter:%i train error:%s vali error:%s" % (
            iter,
            str(train_error),
            str(vail_error),
        )

        train_set_error.append(train_error)
        vail_set_error.append(vail_error)

        stoping_condition = stopping.st3_early_stopping(train_error=train_set_error,vail_error=vail_set_error)

        if not stoping_condition:

            if vail_error<best_validation_loss:
                best_validation_loss = vail_error
                for i,p in enumerate(classifier.params):
                    weigth[i] = p.get_value()

        if (iter>1) and stoping_condition:
            print "Best vali:%s" % (best_validation_loss)

    for p,w in zip(classifier.params,weigth):
        p.set_value(w)

    vail_model = theano.function(
        inputs=[x,y],
        outputs=cost,
        )

    print "vail loss:%s" % (vail_model(X_test.astype(numpy.float64),y_test.reshape(1,y_test.shape[0])[0].astype(numpy.int32)))


if __name__ == "__main__":
    main()
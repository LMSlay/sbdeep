__author__ = 'slay'

import theano
import theano.tensor as T

import numpy

from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix,matthews_corrcoef
from sklearn import svm

import util

theano.config.optimizer='None'

class LogisticRegression(object):

    def __init__(self,input,n_in,n_out):

        self.input = input

        self.W = theano.shared(
            value=numpy.random.uniform(size=(n_in,n_out)),
            name="w"
        )

        self.b = theano.shared(
            value=numpy.zeros((n_out,))
        )

        self.p_1 =1 / (1 + T.exp(-T.dot(self.input, self.W) - self.b))

        self.params = [self.W,self.b]

class HiddenLayer(object):


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


class MLP(object):


    def __init__(self,input,n_in,n_out):


        self.hiddenlayer = HiddenLayer(
            input=input,
            n_in=n_in,
            n_out=n_out,
            )

        self.outputlayer = LogisticRegression(
            input=self.hiddenlayer.output,
            n_in=n_out,
            n_out=1
        )



        self.p_1 = self.outputlayer.p_1

        self.params = self.hiddenlayer.params + self.outputlayer.params

def main():

    X_train, X_test, y_train, y_test, index_train, index_test = util.load_titanic()

    x = T.matrix("x")
    y = T.matrix("y")

    other_data = make_classification()

    learning_rate = 0.1

    classifier = MLP(
        input=x,
        n_in=7,
        n_out=20
    )


    cost = (-y * T.log(classifier.p_1) - (1-y) * T.log(1-classifier.p_1)).mean() + 0.01 * (classifier.params[0] ** 2).sum()

    prediction = classifier.p_1 >0.5

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



    stoping_codition = True

    weigth = classifier.params

    best_validation_loss = 100

    iter = 0

    while stoping_codition:

        iter +=1

        train_set_error =  train_model(X_train.astype(numpy.float64),y_train)

        test_set_error = test_model(X_test.astype(numpy.float64),y_test)

        print "iter:%i train error:%s vali error:%s" % (
            iter,
            str(train_set_error),
            str(test_set_error),
        )

        if test_set_error < best_validation_loss:
            best_validation_loss = test_set_error
            weigth = classifier.params

        if (iter>1) and (test_set_error > best_validation_loss):
            print "Best vali:%s" % (best_validation_loss)
            stoping_codition=False



    classifier.params = weigth

    print "vail loss:%s" % (test_model(X_test.astype(numpy.float64),y_test))


if __name__ == "__main__":
    main()
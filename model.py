import theano.tensor as T
import theano
from util import stopping


class Classification(object):


    def __init__(self,):

        self.layers = []

        self.params = []


    def add(self, layer):
        self.layers.append(layer)

        [self.params.append(param) for param in layer.params]

    def get_value(self,):

        self.output = self.x

        for layer in self.layers:
            self.output = layer.get_value(self.output)

        return self.output


    def negative_log_likelihood(self,):

        return -T.mean(T.log(self.get_value())[T.arange(self.y.shape[0]), self.y])


    def fit(self, X_train, X_test, y_train, y_test,):

        self.x = T.matrix("x")

        self.y = T.ivector('y')

        cost = self.negative_log_likelihood()

        print self.params

        #fine tuning
        learning_rates = []
        for layer in self.layers:
            learning_rates.append(layer.learning_rate)
            learning_rates.append(layer.learning_rate)

        gparams = [T.grad(cost, param) for param in self.params]

        print zip(self.params, gparams, learning_rates)

        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam, learning_rate in zip(self.params, gparams, learning_rates)
        ]

        vail_model = theano.function(
            inputs=[self.x,self.y],
            outputs=cost,
            )

        train_model = theano.function(
            inputs=[self.x,self.y],
            outputs=cost,
            updates=updates,
            )

        weigth = range(len(self.params))

        stoping_condition = False

        best_validation_loss = 100

        iter = 0

        train_set_error = []

        vail_set_error = []

        while not stoping_condition:

            iter +=1

            train_error =  train_model(X_train, y_train)

            vail_error = vail_model(X_test, y_test)

            # fix to function later
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
                    for i,p in enumerate(self.params):
                        weigth[i] = p.get_value()

            if (iter>1) and stoping_condition:
                print "Best vali:%s" % (best_validation_loss)

        for p,w in zip(self.params,weigth):
            p.set_value(w)

        print "vail loss:%s" % (vail_model(X_test, y_test))
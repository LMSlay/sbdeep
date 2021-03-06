
import pandas as pd
import numpy
import theano.tensor as T
import os
import theano
import gzip
import cPickle
from sklearn.cross_validation import train_test_split

numpy.dtype(numpy.float64)

def load_titanic(dataset=pd.read_csv('/Users/slay/Downloads/train.csv')):

    dataset_y = dataset.as_matrix(['Survived'])
    dataset = dataset.drop("Name",axis=1)
    dataset = dataset.drop("PassengerId",axis=1)
    dataset = dataset.drop("Cabin",axis=1)
    dataset = dataset.drop("Ticket",axis=1)
    dataset = dataset.drop("Survived",axis=1)

    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].var())
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    dataset.ix[dataset['Embarked']=='S','Embarked']=1
    dataset.ix[dataset['Embarked']=='C','Embarked']=2
    dataset.ix[dataset['Embarked']=='Q','Embarked']=3

    dataset.ix[dataset['Sex']=='male','Sex']=1
    dataset.ix[dataset['Sex']=='female','Sex']=2

    X_train, X_test, y_train, y_test, index_train, index_test = train_test_split(dataset, dataset_y, dataset.index, test_size=0.2)

    return X_train, X_test, y_train, y_test, index_train, index_test

def load_data(dataset):

    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):

        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    def shared_dataset(data_xy, borrow=True):

        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)

        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval
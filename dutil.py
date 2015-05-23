
import pandas as pd
import numpy
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

import time
import pegasos
import numpy

from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits

def fit():
    raw_data = load_digits(2)
    X = raw_data['data']
    y = raw_data['target']

    datasets = train_test_split(X, y, random_state=12345)
    train_X, test_X, train_y, test_y = datasets

    model = pegasos.PegasosSVMClassifier(lambda_reg=1000000.000001, classes=train_y)
    start = time.clock()
#    model.fit(train_X, train_y)
    end = time.clock()
    #score = model.score(test_X, test_y)

    #print 'acc %.5f in %f seconds' % (score, end-start)

	# try partial fit now
    model = pegasos.PegasosSVMClassifier(lambda_reg=1000000.000001, classes=train_y)
    start = time.clock()
    print (test_y)
    print (train_y)
    print (numpy.asarray(train_y[0]).shape)
#    for i in range(0,train_X.shape[0]-1,2):
#		model.partial_fit(train_X[i:i+1,], train_y[i:i+1])
    model.partial_fit(train_X, train_y)
    end = time.clock()
    score = model.score(test_X, test_y)
    print 'acc %.5f in %f seconds' % (score, end-start)


if __name__ == '__main__':
    fit()


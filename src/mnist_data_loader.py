import cPickle
import gzip
import numpy as np
def load_data():
    mnist_file=gzip.open('../data/mnist_data.pkl.gz')
    training_data,validation_data,test_data = cPickle.load(mnist_file);
    mnist_file.close()
    # training_data: tuple 50000 element 2 dimension input, result
    # validation_data,test_data: tuple 10000 element 2 dimension input, result
    return training_data,validation_data,test_data
def create_data():
    tr_d,va_d,te_d=load_data()
    training_input=[np.reshape(x,(784,1)) for x in tr_d[0]]
    #training_input=[x for x in tr_d[0]]
    training_result=[change_result_fomat(y) for y in tr_d[1]]
    training_data=zip(training_input,training_result)
    validation_input=[np.reshape(x,(784,1)) for x in va_d[0]]
    validation_data=zip(validation_input,va_d[1])
    test_input=[np.reshape(x,(784,1)) for x in te_d[0]]
    test_data=zip(test_input,te_d[1])
    return (training_data, validation_data, test_data)
def change_result_fomat(index):
    # network has 10 output neuron, values 0 or 1
    e=np.zeros((10,1))
    e[index]=1.0
    return e

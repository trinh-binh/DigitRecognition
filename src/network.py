import random
import numpy as np

class Network(object):

    def __init__(self, layers):
        self.num_layers = len(layers)
        self.layers = layers
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(layers[:-1], layers[1:])]

    def feed_forward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def gradient_descent(self, training_data, loop, mini_pack_size, eta,
            test_data=None):
        print(" Computing...")
        n = len(training_data)
        for l in xrange(loop):
            random.shuffle(training_data)
            mini_packe_list = [
                training_data[k:k+mini_pack_size]
                for k in xrange(0, n, mini_pack_size)]
            for mini_pack in mini_packe_list:
                self.update(mini_pack, eta)
            if test_data:
		n_test = len(test_data)
                print("Loop {0} Pass: {1} / {2}").format(
                    l, self.evaluate(test_data), n_test)
    def update(self, mini_pack, eta):
        n_delta_b = [np.zeros(b.shape) for b in self.biases]
        n_delta_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_pack:
            delta_b, delta_w = self.back_propagation(x, y)
            n_delta_b = [n_b+d_b for n_b, d_b in zip(n_delta_b, delta_b)]
            n_delta_w = [n_w+d_w for n_w, d_w in zip(n_delta_w, delta_w)]
        self.biases = [b+(eta/len(mini_pack))*n_b
                       for b, n_b in zip(self.biases, n_delta_b)]
        self.weights = [w+(eta/len(mini_pack))*n_w
                        for w, n_w in zip(self.weights, n_delta_w)]

    def back_propagation(self, x, y):
        n_delta_b = [np.zeros(b.shape) for b in self.biases]
        n_delta_w = [np.zeros(w.shape) for w in self.weights]
        # feed forward
        activation = x
        activations = [x]
        nets = []
        for b, w in zip(self.biases, self.weights):
            n = np.dot(w, activation)+b
            nets.append(n)
            activation = sigmoid(n)
            activations.append(activation)
        # backward
	# last layer
        delta = self.error(y,activations[-1])*sigmoid_prime(nets[-1])
        n_delta_b[-1] = delta
        n_delta_w[-1] = np.dot(delta, activations[-2].transpose())
	#other layers
        for l in xrange(2, self.num_layers):
            n = nets[-l]
            fn = sigmoid_prime(n)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * fn
            n_delta_b[-l] = delta
            n_delta_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (n_delta_b, n_delta_w)
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for (x, y) in test_data]
        s=0
        for x,y in test_results:
            if x==y:
                s=s+1;
        return s
        #return sum(int(x == y) for (x, y) in test_results)
    def error(self, y, out):
        return (y-out)

# Active function
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

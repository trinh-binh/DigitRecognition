import mnist_data_loader
import network
training_data,validation_data,test_data=mnist_data_loader.create_data()
net = network.Network([784,15,10])
net.gradient_descent(training_data,50,10,1.0,test_data=test_data)

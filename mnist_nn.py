import cPickle, gzip
import numpy as np
import time
def sigmoid(a):
    """ Compute the sigmoid function
        Note: use the exponential function from the numpy library (i.e. np.exp()) 
              in your sigmoid function. This allows the function to compute the 
              sigmoid values (element-wise) for an array of numbers.
        
        Parameters: 
            a: the input value
        
        Output: 
            1 / (1 + e^(-a))
    """
    
    return 1.0 / (1 + np.exp(-a))

def sigmoid_derivative(a):
    """ Compute the value of the derivative of a sigmoid function
        Note: you can use the above function here
        
        Parameters: 
            a: the input value
        
        Output: 
            derivative of the sigmoid function applied to 'a'
    """
    return sigmoid(a) * (1.0 - sigmoid(a))

def predict(model, X):
    """ Takes a set of data points and predicts a label (0 or 1) for each. A data point has
        a predicted class of 1 if the final layer of the neural network outputs a value greater or equal
        to 0.5. Otherwise, the label is 0
        
        Parameters:
            model: a dictionary of parameters
            X: a NxK matrix, where each row represents a single data point.
            
        Output:
            A Nx1 Numpy vector, where the i'th value is the label (0 or 1) for the i'th datapoint in X
    """
    
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    z1 = X.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)
    z3 = a2.dot(W3) + b3
    a3 = sigmoid(z3)
    yHat = np.zeros(a3.shape[0])
    #print "a3 ", a3
    for x in range(a3.shape[0]):
        yHat[x] = classify(a3[x])
    #print "yHat ", yHat
    return yHat

def calculate_loss(model, data):    
    """ Compute the loss for the data provided, given the model.
        
        Parameters:
            model: a dictionary of parameters
            data:  a tuple containing the X and y data points, where X is 
                   an NxK matrix and y is a Nx1 vector of labels
            
        Output:
            A value for the loss function.
    """
    
    X, y = data[0], data[1]
    n = X.shape[0]
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    yHat = predict(model, X)
    sum = 0.0
    for x in range(n):
        if y[x] != yHat[x]:
            sum += 1
    return sum / n

def train_nn(data, h1_dim, h2_dim, test_data, learning_rate=0.01, num_epochs=50000, verbose=True):
    """ Train the parameters of the neural network by performing backpropagation 
        on a training set.
    
        Parameters:
            data          : a tuple containing the X and y data points, where X is 
                            an NxK matrix and y is a Nx1 vector of labels
            h1_dim        : number of neurons in the first hidden layer
            h2_dim        : number of neurons in the second hidden layer
            learning_rate : the learning rate for gradient descent
            num_epochs    : number of total iterations of gradient descent. One 
                            iteration means updating the parameters once based on the
                            entire dataset.
            verbose       : if set to True, the loss function value is printed every 
                            100 iterations
        
        Output:
            A dictionary containing all parameters of the neural network. The keys 
            of the dictionary are: W1, b1, W2, b2, W3, b3
    """
    verbose1 = False
    X, y = data[0], data[1]
    
    num_examples = len(X)        # training set size
    input_dim = 784              # number of neurons in the input layer
    output_dim = 10              # number of neurons in the output layer
      
    # Initialize the parameters to random values. We need to learn these.
    W1 = np.random.randn(input_dim, h1_dim) / np.sqrt(input_dim)
    b1 = np.zeros((1, h1_dim))
    W2 = np.random.randn(h1_dim, h2_dim) / np.sqrt(h2_dim)
    b2 = np.zeros((1, h2_dim))
    W3 = np.random.randn(h2_dim, output_dim) / np.sqrt(output_dim)
    b3 = np.zeros((1, output_dim))

    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3' : W3, 'b3' : b3}

    ynew = y.reshape([len(y),1])#np.expand_dims(y, axis = 1)
    yn = np.zeros((len(y), 10))
    for x in range(len(y)):
        yn[x][ynew[x][0]] = 1
    for k in range(num_epochs):
        print calculate_loss(model, test_data)
        for xe, ye in zip(X,yn):
            xe = xe.reshape((1,784))
            ye = ye.reshape((1,10))
            z1 = xe.dot(W1) + b1
            a1 = sigmoid(z1)
            z2 = a1.dot(W2) + b2
            a2 = sigmoid(z2)
            z3 = a2.dot(W3) + b3
            a3 = sigmoid(z3)
            d4 = (a3 - ye) * (sigmoid_derivative(z3))
            d3 = (d4.dot(W3.T)) * (sigmoid_derivative(z2))
            d2 = (d3.dot(W2.T)) * (sigmoid_derivative(z1))
            
            dW3 = a2.T.dot(d4)
            db3 = d4

            dW2 = a1.T.dot(d3)
            db2 = d3
            dW1 = xe.T.dot(d2)
            db1 = d2
            W1 = W1 - (learning_rate * dW1)
            b1 = b1 - (learning_rate * db1)
            W2 = W2 - (learning_rate * dW2)
            b2 = b2 - (learning_rate * db2)
            W3 = W3 - (learning_rate * dW3)
            b3 = b3 - (learning_rate * db3)
            model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3' : W3, 'b3' : b3}

    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3' : W3, 'b3' : b3}
    return model

def classify(Y):
    maxx = Y[0]
    maxind = 0
    for x in range(10):
        if Y[x] > maxx:
            maxx = Y[x]
            maxind = x
    return maxind

def pd(X):
    for i in range(28):
        for j in range(28):
            if X[i*28+j]> 0:
                print "1",
            else:
                print "0",
        print " "


if __name__=="__main__":
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    idx = np.arange(50000)
    aa = np.zeros
    t1 = (np.take(train_set[0], idx, axis=0), np.take(train_set[1], idx, axis=0))
    model = train_nn(t1, 16, 16, test_set, verbose=True)

    #print "Error rate on test set: ", calculate_loss(model, test_set)
    #NNmodel1=NNModel()
    #NNmodel1.test()

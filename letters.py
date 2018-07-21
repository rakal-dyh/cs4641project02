import cPickle, gzip
import pickle
import numpy as np
import time
import scipy.io

def load_data(mat_file_path, width=28, height=28, max_=None, verbose=True):
    ''' Load data in from .mat file as specified by the paper.
        Arguments:
            mat_file_path: path to the .mat, should be in sample/
        Optional Arguments:
            width: specified width
            height: specified height
            max_: the max number of samples to load
            verbose: enable verbose printing
        Returns:
            A tuple of training and test data, and the mapping for class code to ascii value,
            in the following format:
                - ((training_images, training_labels), (testing_images, testing_labels), mapping)
    '''
    # Local functions
    def rotate(img):
        # Used to rotate images (for some reason they are transposed on read-in)
        flipped = np.fliplr(img)
        return np.rot90(flipped)

    def display(img, threshold=0.5):
        # Debugging only
        render = ''
        for row in img:
            for col in row:
                if col > threshold:
                    render += '@'
                else:
                    render += '.'
            render += '\n'
        return render

    # Load convoluted list structure form loadmat
    mat = scipy.io.loadmat(mat_file_path)

    # Load char mapping
    mapping = {kv[0]:kv[1:][0] for kv in mat['dataset'][0][0][2]}
    pickle.dump(mapping, open('mapping.p', 'wb' ))

    # Load training data
    if max_ == None:
        max_ = len(mat['dataset'][0][0][0][0][0][0])
    training_images = mat['dataset'][0][0][0][0][0][0][:max_].reshape(max_, height, width, 1)
    training_labels = mat['dataset'][0][0][0][0][0][1][:max_]

    # Load testing data
    if max_ == None:
        max_ = len(mat['dataset'][0][0][1][0][0][0])
    else:
        max_ = int(max_ / 6)
    testing_images = mat['dataset'][0][0][1][0][0][0][:max_].reshape(max_, height, width, 1)
    testing_labels = mat['dataset'][0][0][1][0][0][1][:max_]

    # Reshape training data to be valid
    if verbose == True:
        _len = len(training_images)
    for i in range(len(training_images)):
        if verbose == True:
            print '%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100),
        training_images[i] = rotate(training_images[i])
    if verbose == True:
        print ''

    # Reshape testing data to be valid
    if verbose == True:
        _len = len(testing_images)
    for i in range(len(testing_images)):
        if verbose == True:
            print '%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100),
        testing_images[i] = rotate(testing_images[i])
    if verbose == True:
        print ''

    # Convert type to float32
    training_images = training_images.astype('float32')
    testing_images = testing_images.astype('float32')

    # Normalize to prevent issues with model
    training_images /= 255
    testing_images /= 255

    nb_classes = len(mapping)

    return ((training_images, training_labels), (testing_images, testing_labels), mapping, nb_classes)



def sigmoid(a):
    return 1.0 / (1 + np.exp(-a))

def sigmoid_derivative(a):
    return sigmoid(a) * (1.0 - sigmoid(a))

def reLU(a):
    n = np.zeros(a.shape)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i][j] <= 0:
                n[i][j] = 0
            else:
                n[i][j] = a[i][j]
    return n

def reLU_derivative(a):
    n = np.zeros(a.shape)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i][j] <= 0:
                n[i][j] = 0
            else:
                n[i][j] = 1
    return n

def predict(model, X, activation):
    """ Takes a set of data points and predicts a label (0 or 1) for each. A data point has
        a predicted class of 1 if the final layer of the neural network outputs a value greater or equal
        to 0.5. Otherwise, the label is 0
        
        Parameters:
            model: a dictionary of parameters
            X: a NxK matrix, where each row represents a single data point.
            
        Output:
            A Nx1 Numpy vector, where the i'th value is the label (0 or 1) for the i'th datapoint in X
    """
    if activation == "reLU":
        act = reLU
        act_der = reLU_derivative
    else:
        act = sigmoid
        act_der = sigmoid_derivative
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

def calculate_loss(model, data, activation):    
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
    yHat = predict(model, X, activation)
    sum = 0.0
    for x in range(n):
        if classify(y[x]) != yHat[x]:
            #pd(X[x])
            #print "labeled as ", yHat[x], ", should be ", y[x]
            sum += 1
    return sum / n

def train_nn(data, h1_dim, h2_dim, test_data, activation, learning_rate=0.02, num_epochs=5000, verbose=True):
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
    if activation == "reLU":
        act = reLU
        act_der = reLU_derivative
    else:
        act = sigmoid
        act_der = sigmoid_derivative
    verbose1 = False
    X, yn = data[0], data[1]
    lowest_error = 1.0
    count = 0
    num_examples = len(X)        # training set size
    input_dim = 784              # number of neurons in the input layer
    output_dim = 26              # number of neurons in the output layer


    W1 = np.random.randn(input_dim, h1_dim) / np.sqrt(input_dim)
    b1 = np.zeros((1, h1_dim))
    W2 = np.random.randn(h1_dim, h2_dim) / np.sqrt(h2_dim)
    b2 = np.zeros((1, h2_dim))
    W3 = np.random.randn(h2_dim, output_dim) / np.sqrt(output_dim)
    b3 = np.zeros((1, output_dim))

    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3' : W3, 'b3' : b3}

    while count < 20:
        count += 1
        loss = 0
        loss = calculate_loss(model, data, activation)
        if loss < lowest_error:
            lowest_error = loss
            count = 0
        if count == 5:
            learning_rate *= 0.5
        print " "
        print "train loss: ", loss
        print "test loss:  ", calculate_loss(model, test_data, activation)
        for xe, ye in zip(X,yn):
            xe = xe.reshape((1,784))
            ye = ye.reshape((1,26))
            z1 = xe.dot(W1) + b1
            a1 = act(z1)
            z2 = a1.dot(W2) + b2
            a2 = act(z2)
            z3 = a2.dot(W3) + b3
            a3 = act(z3)
            d4 = (a3 - ye) * (act_der(z3))
            d3 = (d4.dot(W3.T)) * (act_der(z2))
            d2 = (d3.dot(W2.T)) * (act_der(z1))
            
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
            if verbose1:
                for key in model:
                    print key, " ", model[key].shape
                print "X ", xe.shape
                print "yn ", ye.shape
                print " "
                print "a3 ", a3.shape
                print "yn ", ye.shape
                print " "
                print "z1 ", z1.shape
                print "a1 ", a1.shape
                print "z2 ", z2.shape
                print "a2 ", a2.shape
                print "z3 ", z3.shape
                print "a3 ", a3.shape
                print " "
                print "d4 ", d4.shape
                print "d3 ", d3.shape
                print "d2 ", d2.shape
                print " "
                print "dW3 ", dW3.shape
                print "db3 ", db3.shape
                print "dW2 ", dW2.shape
                print "db2 ", db2.shape
                print "dW1 ", dW1.shape
                print "db1 ", db1.shape
                print " "
                print " "
                print " "
                print " "

    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3' : W3, 'b3' : b3}
    return model

def classify(Y):
    maxx = Y[0]
    maxind = 0
    for x in range(26):
        if Y[x] > maxx:
            maxx = Y[x]
            maxind = x
    return maxind

def pd(X):
    for i in range(28):
        for j in range(28):
            if X[i*28+j] > 0.5:
                print "1",
            else:
                print "0",
        print " "

def pL(X):
    for i in range(28):
        for j in range(28):
            if X[i][j] > 0.5:
                print "1",
            else:
                print "0",
        print " "

def genetic_optimization():
    return 0

if __name__=="__main__":

    train_set2, test_set2, mapping, nb_classes = load_data('emnist_letters', width=28, height=28, max_=None, verbose=False)
    train_set = (np.zeros((len(train_set2[0]), 784)), np.zeros((len(train_set2[1]), 26)))
    test_set = (np.zeros((len(test_set2[0]), 784)), np.zeros((len(test_set2[1]), 26)))
    for x in range(len(train_set2[0])):
        train_set[0][x] = train_set2[0][x].flatten()
        train_set[1][x][train_set2[1][x][0] - 1] = 1.0
    for x in range(len(test_set2[0])):
        test_set[0][x] = test_set2[0][x].flatten()
        test_set[1][x][test_set2[1][x][0] - 1] = 1.0

    for x in range(10):
        pd(train_set[0][x])
        print "len ", len(train_set[0])
        print train_set[1][x]
        print train_set[1][x][0]
        print train_set[1][x].shape

    idx = np.arange(50000)
    N = 50000
    shuffle = np.random.permutation(N)


    # TODO: shuffle is an array of shuffled indices. Break it into consecutive proportions (folds), 
    # and assign the first partition to fold 0, second partition to fold 1, etc...
    cross_val_folds = 5
    fold_idxs = dict()
    for x in range(cross_val_folds):
        fold_idxs[x] = shuffle[x*N/cross_val_folds:(x+1)*N/cross_val_folds]

    t1 = (np.take(train_set[0], idx, axis=0), np.take(train_set[1], idx, axis=0))
    #model = train_nn(t1, 16, 16, test_set, verbose=True)

    print "RUNNING MAIN"
    model = train_nn(train_set, 64, 32, test_set, "reaLU", num_epochs=80)
    for x in range(cross_val_folds):
        print "RUNNING FOLD ", x + 1
        train1 = (np.delete(train_set[0], fold_idxs[x], axis=0), np.delete(train_set[1], fold_idxs[x], axis=0))
        test1 = (np.take(train_set[0], fold_idxs[x], axis=0), np.take(train_set[1], fold_idxs[x], axis=0))
        model = train_nn(train1, 32, 16, test1, "reaLU", num_epochs=80)
        print " "
        print " "
        print "train error:      ", calculate_loss(model, train1)
        print "validation error: ", calculate_loss(model, test1)
        print " "
        print " "


    #print "Error rate on test set: ", calculate_loss(model, test_set)
    #NNmodel1=NNModel()
    #NNmodel1.test()

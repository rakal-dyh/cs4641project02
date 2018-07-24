from  scipy import io as spio
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import matplotlib.pyplot as plt
K.set_image_dim_ordering('th')

emnist = spio.loadmat('matlab/emnist_letters.mat')# load training dataset

x_train = emnist["dataset"][0][0][0][0][0][0]
x_train = x_train.astype(np.float32)
# load training labels
y_train = emnist["dataset"][0][0][0][0][0][1]


# load test dataset
x_test = emnist["dataset"][0][0][1][0][0][0]
x_test = x_test.astype(np.float32)

# load test labels
y_test = emnist["dataset"][0][0][1][0][0][1]

# normalize
x_train /= 255
x_test /= 255

# print('0th dimension of X_train: ',x_train.shape[0])
# print('1st dimension of X_train: ',x_train.shape[1])

# print(y_train.shape[0])
# print(y_train.shape[1])
# print('Unique y_train: ',np.unique(y_train))
# print('Total Unique y_train: ',np.unique(y_train).shape[0])

# reshape using matlab order
# reshape to be [samples][pixels][width][height]
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28, order="A")
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28, order="A")

# labels should be onehot encoded
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# print(y_train)

# print(type(y_train[0]))
y_train1 = y_train.tolist()
y_test1 = y_test.tolist()
#print(y_train)
y_train = np.zeros((x_train.shape[0],26))
y_test = np.zeros((x_test.shape[0],26))


for i in range(len(y_train1)):
    ind = y_train1[i][0]
    y_train[i][ind-1] = 1

for i in range(len(y_test1)):
    ind = y_test1[i][0]
    y_test[i][ind-1] = 1

# print(y_train.shape[0])
# print(y_train.shape[1])

# print('0th dimension of X_train: ',x_train.shape[0])
# print('1st dimension of X_train: ',x_train.shape[1])
# print('2nd dimension of X_train: ',x_train.shape[2])
# print('3rd dimension of X_train: ',x_train.shape[3])

# print(y_train.shape[0])
# print(y_train.shape[1])
num_classes = y_test.shape[1]
def baseline_model():
    # create model
    # Model has Convolutional layer, Pooling layer, Regularization layer,
    # Flatten layer, Fully connected layer (rectifier activation), output layer
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build the model
model = baseline_model()

idx = np.arange(124800)
N = 124800
shuffle = np.random.permutation(N)


# TODO: shuffle is an array of shuffled indices. Break it into consecutive proportions (folds), 
# and assign the first partition to fold 0, second partition to fold 1, etc...
cross_val_folds = 5
fold_idxs = dict()
for x in range(cross_val_folds):
    fold_idxs[x] = shuffle[(int)(x*N/cross_val_folds):(int)((x+1)*N/cross_val_folds)]

# Fit the model
# store fitted model on history
for ind in range(cross_val_folds):
    print(" ")
    print("RUNNING FOLD ", ind)
    print(" ")
    history = model.fit(np.delete(x_train, fold_idxs[ind], axis=0), np.delete(y_train, fold_idxs[ind], axis=0), validation_data=(np.take(x_train, fold_idxs[ind], axis=0), np.take(y_train, fold_idxs[ind], axis=0)), epochs=12, batch_size=200, verbose=2)
    # Final evaluation of the model
    print("final")
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("CNN Error: %.2f%%" % (100-scores[1]*100))
    print(history.history.keys())

    # evaluate using 10-fold cross validation
    # kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    # results = cross_val_score(model, X, Y, cv=kfold)
    # print(results)

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()




# idx = np.arange(50000)
# N = 50000
# shuffle = np.random.permutation(N)


# # TODO: shuffle is an array of shuffled indices. Break it into consecutive proportions (folds), 
# # and assign the first partition to fold 0, second partition to fold 1, etc...
# cross_val_folds = 5
# fold_idxs = dict()
# for x in range(cross_val_folds):
#     fold_idxs[x] = shuffle[x*N/cross_val_folds:(x+1)*N/cross_val_folds]

# t1 = (np.take(train_set[0], idx, axis=0), np.take(train_set[1], idx, axis=0))
# #model = train_nn(t1, 16, 16, test_set, verbose=True)

# print "RUNNING MAIN"
# model = train_nn(train_set, 64, 32, test_set, "reaLU", num_epochs=80)
# for x in range(cross_val_folds):
#     print "RUNNING FOLD ", x + 1
#     train1 = (np.delete(train_set[0], fold_idxs[x], axis=0), np.delete(train_set[1], fold_idxs[x], axis=0))
#     test1 = (np.take(train_set[0], fold_idxs[x], axis=0), np.take(train_set[1], fold_idxs[x], axis=0))
#     model = train_nn(train1, 32, 16, test1, "reaLU", num_epochs=80)
#     print " "
#     print " "
#     print "train error:      ", calculate_loss(model, train1)
#     print "validation error: ", calculate_loss(model, test1)
#     print " "
#     print " "
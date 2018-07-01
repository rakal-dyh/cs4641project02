#this code read the first n number of nodes with random label


import os, struct
from array import array
from cvxopt.base import matrix
import numpy as np
from PIL import Image as im
from matplotlib import pyplot as plt

def read(num, dataset = "training", path = "MNIST_data"):
    """
    Python function for importing the MNIST data set.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = array("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = array("B", fimg.read())
    fimg.close()

    #ind = [ k for k in xrange(size) if lbl[k] in digits ]
    images =  matrix(0, (num, rows*cols))
    labels = matrix(0, (num, 1))
    for i in xrange(num):
        images[i, :] = img[ i*rows*cols : (i+1)*rows*cols ]
        labels[i] = lbl[i]

    return images, labels


if __name__=="__main__":
    [images,label]=read(3)
    images1=images[1,0:784]
    #label1=label[0:2]
    print(images1)
    #print(label1)
    data=np.array(images1, dtype='float')
    data=255-data
    data=data.reshape((28,28))
    print data
    img = im.fromarray(np.uint8(data * 255) , 'L')
    img.show()

    pass

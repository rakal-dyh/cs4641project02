from loadMNIST import read
import numpy as np

class NNModel():
    data=[]
    label=[]
    labelFormatted[]

    testdata=[]
    testlabel=[]
    testlabelFormatted[]

    #this method read the images and labels
    def readData(self,numberOfRows):
        self.data=read(3)
        print data
        pass

    #use the data to train the model
    def trainModel(self):
        pass

    #predict new data points with model
    #retrun a vector with 10 columns
    '''
    ex: (new format)
        1,0,0,0,0,0,0,0,0,0
        0,0,0,0,1,0,0,0,0,0
        ...
    it is (old format)
        0
        4
    '''
    def predict(self):
        pass

    #calculate_loss
    def calculate_loss(self):
        pass

    #using test data to check the accuracy
    def validation(self):
        pass

    #change from origin label to new formatted label
    '''
        origin label is a vector like:
        [0,5,2,4,...]
        new formatted label is:
        1,0,0,0,0,0,0,0,0,0
        0,0,0,0,0,1,0,0,0,0
        0,0,1,0,0,0,0,0,0,0
        0,0,0,0,1,0,0,0,0,0
    '''
    def changeLabelFormatToNew(self):
        pass

#Util:

    @staticmethod
    def sigmoid(a):
        return 1/(1+np.exp(-a))

    @staticmethod
    def sigmoid_derivative(a):
        out=np.multiply(sigmoid(a),(1-sigmoid(a)))
    return out



if __name__=="__main__":
    NNmodel1=NNModel()

    pass

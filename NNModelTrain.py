from loadMNISTinNum import read
import numpy as np

class NNModel():
    data=np.zeros(0)
    label=np.zeros(0)
    labelFormatted=np.zeros(0)

    testdata=np.zeros(0)
    testlabel=np.zeros(0)
    testlabelFormatted=np.zeros(0)


    #for test
    def test(self):
        num=50
        self.readData(num,num)
        X=self.data
        y=self.labelFormatted
        #print y
        data = [X,y]
        model = self.train_nn(data, 30, 30, verbose=True)
        pre=self.predict(model,X)
        #print pre
        print self.check_accuracy(y,pre,num)
        pass


    #this method read the images and labels
    def readData(self,numberOfRows,numberOfRows_test):
        data,label=read(numberOfRows)
        self.data=np.array(data)
        self.label=np.array(label)
        self.labelFormatted=self.changeLabelFormatToNew(self.label)
        data2,label2=read(numberOfRows_test,"testing","MNIST_data")
        self.testdata=np.array(data2)
        self.testlabel=np.array(label2)
        self.testlabelFormatted=self.changeLabelFormatToNew(self.testlabel)
        pass


    #use the data to train the model
    def train_nn(self,data, h1_dim, h2_dim, learning_rate=0.01, num_epochs=2000, verbose=False):
        X, y = data[0], data[1]

        num_examples = len(X)        # training set size
        input_dim = 784                # number of neurons in the input layer
        output_dim = 10               # number of neurons in the output layer

        # Initialize the parameters to random values. We need to learn these.
        W1 = np.random.randn(input_dim, h1_dim) / np.sqrt(input_dim)
        b1 = np.zeros((1, h1_dim))
        W2 = np.random.randn(h1_dim, h2_dim) / np.sqrt(h2_dim)
        b2 = np.zeros((1, h2_dim))
        W3 = np.random.randn(h2_dim, output_dim) / np.sqrt(output_dim)
        b3 = np.zeros((1, output_dim))

        # This is what we return at the end. UPDATE THIS AFTER EACH ITERATION
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3' : W3, 'b3' : b3}

        # TODO: Your implementation goes here
        for i in range(num_epochs):
            print "epoch:"
            print i
            z1=np.matrix(X)*np.matrix(W1)+b1
            a1=self.sigmoid(z1)
            z2=a1*W2+b2
            a2=self.sigmoid(z2)
            z3=a2*W3+b3
            a3=self.sigmoid(z3)


            preY=self.predict(model,X)
            delta4=np.multiply(np.matrix(preY-y),self.sigmoid_derivative(z3))
            delta3=np.multiply(self.sigmoid_derivative(z2),delta4*np.transpose(W3))
            delta2=np.multiply(self.sigmoid_derivative(z1),delta3*np.transpose(W2))



            parW3=np.transpose(a2)*delta4
            parb3=np.sum(delta4)
            parW2=np.transpose(a1)*delta3
            parb2=np.sum(delta3)
            parW1=np.transpose(X)*delta2
            parb1=np.sum(delta2)

            #print parW1
            #print parW2
            #print parW3


            model['W1']=model['W1']-learning_rate*parW1
            model['W2']=model['W2']-learning_rate*parW2
            model['W3']=model['W3']-learning_rate*parW3

            model['b1']=model['b1']-learning_rate*parb1
            model['b2']=model['b2']-learning_rate*parb2
            model['b3']=model['b3']-learning_rate*parb3

        return model
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
    def predict(self,model,X):
        W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
        N=np.size(X,0)

        output=np.zeros(N)

        z1=np.matrix(X)*np.matrix(W1)+b1
        a1=self.sigmoid(z1)

        z2=a1*W2+b2
        a2=self.sigmoid(z2)

        z3=a2*W3+b3
        a3=self.sigmoid(z3)

        # output=[0,0,0,0,0,0,0,0,0,0,0]
        # pos=0
        # max=-1
        # for i in range(10):
        #     print a3[i]
        #     if a3[i]>max:
        #         pos=i
        #         max=a3[i]
        # output[pos]=1
        output=[]

        for i in range(N):
            pos=0
            max=-1
            for j in range(10):
                if a3[i,j]>max:
                    pos=j
                    max=a3[i,j]
            tmp=[0,0,0,0,0,0,0,0,0,0]
            tmp[pos]=1
            output.append(tmp)

        out=np.array(output)
        return out
        #        0.8,0.6,0.12,0,0,0,0,0,0,0
        #        1,0,0,0,0,0,0,0,0,0


    #calculate_loss
    def calculate_loss(self,model,data):
        X, y = data[0], data[1]
        W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']

        N=np.size(y)
        yPre=np.array(predict(model,X))


        sum=0
        for i in range(N):
            sum=sum+self.calculate_loss_two_points(yPre[i],y[i])
            sum=sum/N
            return sum
        pass


    #using test data to check the accuracy
    def validation(self):
        #print self.predict()

        pass

    def check_accuracy(self,y,pre,num):
        right=0
        for i in range(num):
            for j in range(10):
                if (y[i,j]==1 and pre[i,j]==1):
                    right=right+1
        num=num+0.0
        right=right+0.0
        print right
        print num
        return right/num

    #change from origin label to new formatted label
    #return new formatted label
    '''
        origin label is a vector like:
        [0,5,2,4,...]
        new formatted label is:
        1,0,0,0,0,0,0,0,0,0
        0,0,0,0,0,1,0,0,0,0
        0,0,1,0,0,0,0,0,0,0
        0,0,0,0,1,0,0,0,0,0
    '''
    def changeLabelFormatToNew(self,oldlabel):
        datasize=oldlabel.size
        #newlabel=[]
        tmplabel=[]
        for i in range(datasize):
            tmp=[0,0,0,0,0,0,0,0,0,0]
            pos=oldlabel[i]
            tmp[pos[0]]=1
            tmplabel.append(tmp)
        #print tmplabel
        newlabel=np.array(tmplabel)
        return newlabel
        pass

#Util:

    @staticmethod
    def sigmoid(a):
        return 1/(1+np.exp(-a))


    def sigmoid_derivative(self,a):
        out=np.multiply(self.sigmoid(a),(1-self.sigmoid(a)))
        return out

    @staticmethod
    def calculate_loss_two_points(yPre,yReal):
        pos=0
        for i in range (10):
            if yReal[i]==1:
                pos=i
        return abs(1-yPre[pos])
        pass



if __name__=="__main__":
    NNmodel1=NNModel()
    NNmodel1.test()
    pass

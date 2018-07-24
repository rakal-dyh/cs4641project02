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
        num_train=1000
        num_test=10000
        self.readData(num_train,num_test)
        X=self.data
        y=self.labelFormatted
        X_test=self.testdata
        y_test=self.testlabelFormatted
        #print y
        data = [X,y]
        data_pre=[self.testdata,self.testlabelFormatted]
        model = self.train_nn(data, 40, 10, verbose=True)


        '''
        if validate with train data, use X, y, num_train bellow
        if validate with test data, use X_test, y_test, num_test
        '''
        pre=self.predict(model,X_test)
        print self.check_accuracy(y_test,pre,num_test)

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
    def train_nn(self,data, h1_dim, h2_dim, learning_rate=0.01, num_epochs=300, verbose=False):
        X, y = data[0], data[1]

        num_examples = len(X)        # training set size
        input_dim = 784                # number of neurons in the input layer
        output_dim = 1               # number of neurons in the output layer

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
            W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
            z1=np.matrix(X)*np.matrix(W1)+b1
            a1=self.sigmoid(z1)
            z2=a1*W2+b2
            a2=self.sigmoid(z2)
            z3=a2*W3+b3
            a3=self.sigmoid(z3)

            #print np.transpose(a3)
            #print a3[1]


            preY=self.predict(model,X)
            # if i==0:
            #     print np.size(np.transpose(a3)-np.matrix(y))



            #delta4=np.multiply(np.transpose(np.matrix(preY-y)),self.sigmoid_derivative(z3))
            delta4=np.multiply(np.transpose(np.transpose(a3)-np.matrix(y)),self.sigmoid_derivative(z3))
            delta3=np.multiply(self.sigmoid_derivative(z2),delta4*np.transpose(W3))
            delta2=np.multiply(self.sigmoid_derivative(z1),delta3*np.transpose(W2))

            # print type(np.transpose(np.matrix(preY-y)))
            # print type(np.transpose(a3)-np.matrix(y))
            # print np.size(np.transpose(np.matrix(preY-y)),0)
            # print np.size(np.transpose(a3)-np.matrix(y),1)

            parW3=np.transpose(a2)*delta4
            #print np.size(parW3,0)
            #print np.size(parW3,1)
            parb3=np.sum(delta4)
            #print np.size(parb3)
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

    '''
    def predict(self,model,X):
        W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
        N=np.size(X,0)

        #output=np.zeros(N)

        z1=np.matrix(X)*np.matrix(W1)+b1
        a1=self.sigmoid(z1)

        z2=a1*W2+b2
        a2=self.sigmoid(z2)

        z3=a2*W3+b3
        a3=self.sigmoid(z3)

        out=[]
        print a3[1]
        for i in range(N):
            if a3[i]>0.5:
                out.append(1)
            else:
                out.append(0)

        #print out
        #print '----'
        #print np.transpose(a3)
        return out


    #calculate_loss
    def calculate_loss(self,model,data):
        X, y = data[0], data[1]
        W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']

        N=np.size(y)
        yPre=np.array(predict(model,X))


        sum=0
        for i in range(N):
            sum=sum+abs(yPre[i]-y[i])
            sum=sum/N
        return sum
        pass


    #using test data to check the accuracy
    def validation(self):
        #print self.predict()

        pass

    def check_accuracy(self,y,pre,num):
        right=0
        num_0=0
        for i in range(num):
            if (y[i]==1) and (pre[i]==1):
                right=right+1
                if y[i]==1:
                    num_0=num_0+1
        num=num
        right=right+0.0
        print 'accuacy on identify 0'
        print right
        print num_0
        print right/(num_0)

        print 'accuacy for all'
        right=0
        for i in range(num):
            if y[i]==pre[i]:
                right=right+1
        num=num+0.0
        right=right+0.0
        print right
        print num
        return right/num

    #change from origin label to new formatted label
    #return new formatted label
    '''
        if label is 0 : 1
        if label != 0 : 0
    '''
    def changeLabelFormatToNew(self,oldlabel):
        datasize=oldlabel.size
        #newlabel=[]
        # tmplabel=[]
        # for i in range(datasize):
        #     tmp=[0,0,0,0,0,0,0,0,0,0]
        #     pos=oldlabel[i]
        #     tmp[pos[0]]=1
        #     tmplabel.append(tmp)
        # #print tmplabel
        # newlabel=np.array(tmplabel)
        # return newlabel
        # pass
        newlabel=[]
        for i in range(datasize):
            if oldlabel[i]==0:
                newlabel.append(1)
            else:
                newlabel.append(0)
        newlabel=np.array(newlabel)
        return newlabel

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

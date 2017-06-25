import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# create neuron class
class Neuron:

    # define attributes
    def __init__(self, input, weight, bias):
        self.input = input
        self.weight = weight
        self.bias = bias

    # methods

    # calculate net input n = Wp + b
    def net_input(self):
        n = self.weight * self.input + self.bias
        return n

    # hardlimit transfer function a = hardlim(n)
    def hardlim(self):
        if self.net_input() >= 0:
            a = 1
        else:
            a = 0
        return a

    # log-sigmoid transfer function a = logsig(n)
    def logsig(self):
        a = 1/(1 + np.exp(-self.net_input()))
        return a

    # linear transfer function
    def purelin(self):
        a = self.net_input()
        return a

    # softmax transfer function a = e^n/sum(e^n)
    def softmax(self):
        a = np.exp(self.net_input())/np.exp(self.net_input()).sum()
        return a

    # softmax derivative jacobian matrix
    def j_softmax(self,a,num):
        jacobian = np.empty([num, num])
        for i in range(num):
            for j in range(num):
                if i == j:
                    jacobian[i, j] = a[i] * np.sum(a - a[i])
                else:
                    jacobian[i, j] = -a[i] * a[j]
        return jacobian

    # hyperbolic tangent sigmoid transfer function
    def tangsig(self):
        a = 2/(1 + np.exp(-2 * self.net_input())) - 1
        return a

    # tansig derivative jacobian matrix
    def j_tansig(self,a,num):
        jacobian = np.diag(np.ones(num)) - np.diagflat(np.power(a, 2))
        return jacobian

    # classify output
    def classify(self,a):
       c = np.where(a[0]>=0.5,1,0)
       c = np.asscalar(c)
       return c

# define cross entropy error calculation function for softmax (assuming targets of 1,0)
def cross_entropy(a,t):
    i = np.where(a==0)
    a[i] = 1e-15 # replace 0s so ln doesn't produce infinity
    e = -t * np.log(a)
    #e = - np.multiply(np.transpose(t), np.log(a))
    return e

# calculate hidden layer sensitivity
def senseh(F_prime,W,s):
    s = F_prime * np.transpose(W) * s
    return s

# calculate output layer sensitivity
def senseo(F_prime,e):
    s = -2 * F_prime * e
    return s

# define learning rule function
def learn(weight_old, bias_old, sensitivity, input, learning_rate):
    weight_new = weight_old - learning_rate * sensitivity * np.transpose(input)
    bias_new = bias_old - learning_rate * sensitivity
    return  weight_new, bias_new

# normailize data and split dataset into training, validation, testing sets
def split(dataset,t,v):
    # normalize dataset into range between -1 and 1
    norm = dataset.ix[:, :-2].apply(lambda x: -1 + 2 * (x - x.min()) / (x.max() - x.min()), axis=0)
    dataset = pd.concat([norm,dataset.ix[:,-2:]],axis=1)

    # split dataset by target class
    dataset0 = dataset.ix[dataset.ix[:,'window']==0,:]
    dataset1 = dataset.ix[dataset.ix[:,'window']==1,:]

    # build datasets by target class
    train0, validate0, test0 = np.split(dataset0.sample(frac=1),
                                        [int(t * len(dataset0)), int((1-v) * len(dataset0))])
    train1, validate1, test1 = np.split(dataset1.sample(frac=1),
                                        [int(t * len(dataset1)), int((1-v) * len(dataset1))])

    # re-combine by target class
    train = pd.concat([train0,train1])
    validate = pd.concat([validate0, validate1])
    test = pd.concat([test0, test1])

    # shuffle order and reset index numbers
    train = train.sample(frac=1)
    train.reset_index(inplace=True)
    validate = validate.sample(frac=1)
    validate.reset_index(inplace=True)
    test = test.sample(frac=1)
    test.reset_index(inplace=True)

    return train.iloc[:,1:], validate.iloc[:,1:], test.iloc[:,1:]

def main():

    # import dataset
    cols = ['id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'type']
    glass = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data', names=cols)
    # create window target column (1 for true; 0 for false)
    glass.ix[glass.ix[:, 'type'] <= 4, 'window'] = 1
    glass.ix[glass.ix[:, 'type'] > 4, 'window'] = 0
    # create non-window target column (1 for true; 0 for false)
    glass.ix[glass.ix[:, 'type'] > 4, 'non_window'] = 1
    glass.ix[glass.ix[:, 'type'] <= 4, 'non_window'] = 0
    # drop id and class columns
    glass.drop(['id', 'type'], inplace=True, axis=1)

    # split into training, validate, testing sets
    train, validate, test = split(glass, t=0.7, v=0.15)

    # specify number of neurons in each layer and the learning rate
    num_neurons1 = 10
    num_neurons2 = 2
    alpha = 0.1
    iterations = 100

    # initialize weight and bias randomly for each layer from -0.5 to 0.5
    # W1 number of columns matches training set columns, less final two
    W1 = np.matrix(np.random.rand(num_neurons1,train.iloc[0,:-2].count()) - 0.5)
    b1 = np.matrix(np.random.rand(num_neurons1,1) - 0.5)
    W2 = np.matrix(np.random.rand(num_neurons2,num_neurons1) - 0.5)
    b2 = np.matrix(np.random.rand(num_neurons2,1) - 0.5)

    # initialize cross entropy lists for training and validation sets
    ce_t = []
    ce_v = []
    # initialize validation set indexer
    index_v = 0

    # training & validation
    for epoch in range(iterations):

        # if number of iterations is longer than dataset, loop back to the beginning
        index = epoch
        if index >= len(train):
            index = epoch - len(train)

        # create input matrix from training dataset
        input = np.matrix(train.iloc[index,:-2])
        input = np.transpose(input)

        # first layer
        neuron1 = Neuron(input=input,weight=W1,bias=b1)
        a1 = neuron1.tangsig()
        # second layer
        neuron2 = Neuron(input=a1,weight=W2,bias=b2)
        a2 = neuron2.softmax()

        # calculate error of each iteration and update cost total
        target = np.matrix(train.iloc[index,-2:])
        e = cross_entropy(a2,target)
        ce_t.append(np.asscalar(e))
        e = np.concatenate([e,e])

        # calculate layer 2 sensitivity
        s2 = senseo(F_prime=neuron2.j_softmax(a=a2,num=num_neurons2),e=e)

        # calculate layer 1 sensitivity
        s1 = senseh(F_prime=neuron1.j_tansig(a=a1,num=num_neurons1),W=W2,s=s2)

        # calculate new weight and bias for layer 2
        W2, b2 = learn(weight_old=W2,bias_old=b2,sensitivity=s2,input=a1,learning_rate=alpha)
        # calculate new weight and bias for layer 1
        W1, b1 = learn(weight_old=W1,bias_old=b1,sensitivity=s1,input=input,learning_rate=alpha)

        # run validation check every only on iterations where training set % validation set == 0
        if index % (len(train)/len(validate)) == 0:
            if index_v >= len(validate):
                index_v = 0
            # create input matrix from validation dataset
            input = np.matrix(validate.iloc[index_v, :-2])
            input = np.transpose(input)
            neuron1 = Neuron(input=input, weight=W1, bias=b1)
            a1 = neuron1.tangsig()
            # second layer
            neuron2 = Neuron(input=a1, weight=W2, bias=b2)
            a2 = neuron2.softmax()
            # calculate error and add to cost list
            target = np.matrix(validate.iloc[index_v, -2:])
            e = cross_entropy(a2, target)
            ce_v.append(np.asscalar(e))

            # increment validation set indexer
            index_v += 1
        else:
            ce_v.append(np.nan)


    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=[8, 8])
    ax.plot(np.arange(0,iterations),ce_t)
    ax.plot(np.arange(0,len(ce_v)),ce_v,'k.')
    plt.xlabel('epochs')
    plt.ylabel('cross entropy')
    plt.grid()
    plt.show()

    # initialize confusion matrix series
    actual = pd.Series(test.iloc[:,-2],name='Actual')
    predict = []

    # test network
    for i in range(len(test)):
        # create input matrix from test dataset
        input = np.matrix(test.iloc[i, :-2])
        input = np.transpose(input)
        neuron1 = Neuron(input=input,weight=W1,bias=b1)
        a1 = neuron1.tangsig()
        neuron2 = Neuron(input=a1,weight=W2,bias=b2)
        a2 = neuron2.softmax()
        c = neuron2.classify(a2)
        predict.append(c)

    # generate confusion matrix of test results
    predict = pd.Series(predict,name='Predicted')
    confusion = pd.crosstab(actual,predict,margins=True)
    print confusion

if __name__ == '__main__':
    main()
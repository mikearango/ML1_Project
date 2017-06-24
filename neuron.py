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
    def j_softmax(self,a,n):
        jacobian = np.empty([n, n])
        for i in range(n):
            for j in range(n):
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
    def j_tansig(self,a,n):
        jacobian = np.diag(np.ones(n)) - np.diagflat(np.power(a, 2))
        return jacobian

# define error calculation function
def error_calc(output, target):
    e = target - output
    return e

# define cross entropy error calculation function for softmax (assuming targets of 1,0)
def cross_entropy(a):
    i = np.where(a == 0)
    a[i] = 1e-15 # replace 0s so ln doesn't produce infinity
    e = -np.log(a).sum()
    return e

# calculate hidden layer sensitivity
def senseh(F_prime,W_1,s_1):
    s = F_prime * np.transpose(W_1) * s_1
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

# split dataset into training, validation, testing sets
def split(dataset,t,v):
    # split dataset by target class
    dataset0 = dataset.ix[dataset.ix[:,'target']==0,:]
    dataset1 = dataset.ix[dataset.ix[:,'target']==1,:]

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

    return train, validate, test

def main():

    # import dataset
    cols = ['id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'type']
    glass = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data', names=cols)
    glass.ix[glass.ix[:, 'type'] <= 4, 'target'] = 0 # assign 0 to window glass
    glass.ix[glass.ix[:, 'type'] > 4, 'target'] = 1 # assign 1 to non-window glass
    glass.drop(['id', 'type'], inplace=True, axis=1)

    # split into training, validate, testing sets
    train, validate, test = split(glass, t=0.7, v=0.15)

    # specify number of neurons in each layer and the learning rate
    num_neurons1 = 10
    num_neurons2 = 2
    alpha = 0.2
    iterations = 100

    # initialize weight and bias randomly for each layer from -0.5 to 0.5
    W1 = np.matrix(np.random.rand(num_neurons1,1) - 0.5)
    b1 = np.matrix(np.random.rand(num_neurons1,1) - 0.5)
    W2 = np.matrix(np.random.rand(num_neurons2,num_neurons1) - 0.5)
    b2 = np.matrix(np.random.rand(num_neurons2,1) - 0.5)

    # initialize cost lists for training and validation sets
    cost_t = []
    cost_v = []

    # training & validation
    for epoch in range(iterations):

        # first layer
        neuron1 = Neuron(input=train[epoch],weight=W1,bias=b1)
        a1 = neuron1.tangsig()
        # second layer
        neuron2 = Neuron(input=a1,weight=W2,bias=b2)
        a2 = neuron2.softmax()

        # calculate error of each iteration and update cost total
        cost_t.append(cross_entropy(a2))

        # calculate layer 2 sensitivity
        s2 = senseo(F_prime=neuron2.j_softmax(a=a2,n=num_neurons2),e=e)

        # calculate layer 1 sensitivity
        s1 = senseh(F_prime=neuron1.j_tansig(a=a1,n=num_neurons1),w=W2,s=s2)

        # calculate new weight and bias for layer 2
        W2, b2 = learn(weight_old=W2,bias_old=b2,sensitivity=s2,input=a1,learning_rate=alpha)
        # calculate new weight and bias for layer 1
        W1, b1 = learn(weight_old=W1,bias_old=b1,sensitivity=s1,input=p_sample,learning_rate=alpha)

    # test network
    neuron1 = Neuron(input=test,weight=W1,bias=b1)
    a1 = neuron1.tangsig()
    neuron2 = Neuron(input=a1,weight=W2,bias=b2)
    a2 = neuron2.softmax()
    

if __name__ == '__main__':
    main()
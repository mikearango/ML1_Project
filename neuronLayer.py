import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# create neuron class
class NeuronLayer:
    # define attributes
    def __init__(self, r, s, f, j):
        self.r = r # Number of inputs
        self.s = s # number of neurons
        self.f = f # Transfer function
        self.j = j # derivatve transfer function

    def setWeightBias(self, w, b):
        self.w = w  # set weights
        self.b = b # set biases
        self.r = w.shape[1]
        self.s = w.shape[0]
        return

    def processInput(self, p):
        self.p = p
        self.n = self.w * self.p + self.b
        self.a = self.f(self.n)
        return


# hardlimit transfer function a = hardlim(n)
def hardlim(n):
    if n >= 0:
        a = 1
    else:
        a = 0
    return a


# log-sigmoid transfer function a = logsig(n)
def logsig(n):
    return 1 / (1 + np.exp(-n))


# linear transfer function
def purelin(n):
    return n

# hyperbolic tangent sigmoid transfer function
def tansig(n):
    return (2 / (1 + np.exp(-2 * n)) - 1)


# softmax transfer function a = e^n/sum(e^n)
def softmax(n):
    # a = np.exp(self.net_input())/np.exp(self.net_input()).sum()
    return np.exp(n / np.exp(n).sum(axis=0))

# classify output
def classify(a):
    c = np.where(a[0] >= 0.5, 1, 0)
    return c


# tansig derivative jacobian matrix
def j_tansig(a):
    jacobian = np.diag(np.ones(a.shape[0])) - np.diagflat(np.power(a, 2))
    return jacobian


# softmax derivative jacobian matrix
def j_softmax(a):
    jacobian = np.empty([a.shape[0], a.shape[0]])
    for i in range(jacobian.shape[0]):
        for j in range(jacobian.shape[1]):
            if i == j:
                jacobian[i, j] = a[i]*np.sum(a-a[i])
            else:
                jacobian[i, j] = -a[i] * a[j]
    return jacobian


# define cross entropy error calculation function for softmax (assuming targets of 1,0)
def cross_entropy(a, t):
    i = np.where(a == 0)
    a[i] = 1e-15  # replace 0s so ln doesn't produce infinity
    e = -t * np.log(a)
    return e


# calculate hidden layer sensitivity
def senseh(F_prime, W, s):
    s = F_prime * np.transpose(W) * s
    return s


# calculate output layer sensitivity
def senseo(F_prime, e):
    s = -2 * F_prime * e
    return s


# define learning rule function
def learn(weight_old, bias_old, sensitivity, input, learning_rate):
    weight_new = weight_old - learning_rate * sensitivity * np.transpose(input)
    bias_new = bias_old - learning_rate * sensitivity
    return weight_new, bias_new


# normailize data and split dataset into training, validation, testing sets
def split(dataset, num_inputs, t, v):
    # normalize dataset into range between -1 and 1

    norm = dataset.ix[:, slice(0,num_inputs)].apply(lambda x: -1 + 2 * (x - x.min()) / (x.max() - x.min()), axis=0)
    dataset = pd.concat([norm, dataset.ix[:,'window']], axis=1)

    # split dataset by target class
    dataset0 = dataset.ix[dataset.ix[:, 'window'] == 0, :]
    dataset1 = dataset.ix[dataset.ix[:, 'window'] == 1, :]

    # build datasets by target class
    train0, validate0, test0 = np.split(dataset0.sample(frac=1),
                                        [int(t * len(dataset0)), int((1 - v) * len(dataset0))])
    train1, validate1, test1 = np.split(dataset1.sample(frac=1),
                                        [int(t * len(dataset1)), int((1 - v) * len(dataset1))])

    # re-combine by target class
    train = pd.concat([train0, train1])
    validate = pd.concat([validate0, validate1])
    test = pd.concat([test0, test1])

    # shuffle order and reset index numbers
    train = train.sample(frac=1)
    train.reset_index(inplace=True)
    validate = validate.sample(frac=1)
    validate.reset_index(inplace=True)
    test = test.sample(frac=1)
    test.reset_index(inplace=True)

    return train.iloc[:, 1:], validate.iloc[:, 1:], test.iloc[:, 1:]


def main():
    # import dataset

    cols = ['id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'type']
    glass = pd.read_csv('glass.txt', names=cols)
    # create window target column (1 for true; 0 for false)
    glass.ix[glass.ix[:, 'type'] <= 4, 'window'] = 1
    glass.ix[glass.ix[:, 'type'] > 4, 'window'] = 0
    # create non-window target column (1 for true; 0 for false)
    glass.ix[glass.ix[:, 'type'] > 4, 'non_window'] = 1
    glass.ix[glass.ix[:, 'type'] <= 4, 'non_window'] = 0
    # drop id and class columns
    glass.drop(['id', 'type'], inplace=True, axis=1)
    num_inputs = glass.values.shape[1] - 2

    # split into training, validate, testing sets
    train, validate, test = split(glass, num_inputs, t=0.7, v=0.15)

    # specify number of neurons in each layer and the learning rate
    num_neurons1 = 10
    num_neurons2 = 2
    alpha = 0.1
    epoch = 30
    neuron1 = NeuronLayer(num_neurons1, num_inputs , tansig, j_tansig)
    neuron2 = NeuronLayer(num_neurons2, num_neurons1, softmax, j_softmax)

    # initialize weight and bias randomly for each layer from -0.5 to 0.5
    # W1 number of columns matches training set columns, less final two
    W1 = np.matrix(np.random.rand(num_neurons1, num_inputs) - 0.5)
    b1 = np.matrix(np.random.rand(num_neurons1, 1) - 0.5)
    W2 = np.matrix(np.random.rand(num_neurons2, num_neurons1) - 0.5)
    b2 = np.matrix(np.random.rand(num_neurons2, 1) - 0.5)

    # initialize cross entropy lists for training and validation sets
    ce_t = []
    ce_v = []
    # initialize validation set indexer
    index_v = 0

    # training & validation
    for j in range(epoch):

        for i in range(len(train)):

            #set weights and biases
            neuron1.setWeightBias(w=W1, b=b1)
            neuron2.setWeightBias(w=W2, b=b2)

            # create input matrix from training dataset
            input = np.matrix(train.iloc[i, slice(0,num_inputs)]).transpose()

            # process network
            neuron1.processInput(input)
            neuron2.processInput(neuron1.a)


            # calculate error of each iteration and update cost total
            target = np.matrix(train.iloc[i, slice(0,num_inputs)])
            e = cross_entropy(neuron2.a, target)
            if i == 0:
                e_all = e
            else:
                e_all = np.concatenate((e_all, e), axis=1)
            e = np.concatenate([e, e])

            # calculate layer 2 sensitivity
            s2 = senseo(F_prime=j_softmax(a=neuron2.a), e=e)
            if i == 0:
                s2_all = s2
            else:
                s2_all = np.concatenate((s2_all, s2), axis=1)

            # calculate layer 1 sensitivity
            s1 = senseh(F_prime=j_tansig(a=neuron1.a), W=W2, s=s2)
            if i == 0:
                s1_all = s1
            else:
                s1_all = np.concatenate((s1_all, s1), axis=1)

        # append average cross entropy to list
        ce_t.append(e_all.mean())

        # calculate new weight and bias for layer 2
        W2, b2 = learn(weight_old=W2, bias_old=b2, sensitivity=s2_all.mean(axis=1), input=neuron1.a, learning_rate=alpha)
        # calculate new weight and bias for layer 1
        W1, b1 = learn(weight_old=W1, bias_old=b1, sensitivity=s1_all.mean(axis=1), input=neuron1.p, learning_rate=alpha)

        # validation
        input = np.matrix(validate.iloc[:, slice(0,num_inputs)]).transpose()


        # first layer
        neuron1.processInput(input)
        neuron2.processInput(neuron1.a)

        target = np.matrix(validate.iloc[:, slice(0,num_inputs)])
        # compute errors
        for i in range(len(target)):
            e = cross_entropy(neuron2.a[:, i], target[i])
            if i == 0:
                e_all = e
            else:
                e_all = np.concatenate((e_all, e), axis=1)

        # append average validation cross entropy to list
        ce_v.append(e_all.mean())
        if j >= 5 and ce_v[j] > ce_v[j - 1]:
            break

    # initialize confusion matrix series
    actual = pd.Series(test.iloc[:,slice(0,num_inputs)], name='Actual')

    # test network

    # create input matrix from test dataset
    input = np.matrix(test.iloc[:, slice(0,num_inputs)]).transpose()
    neuron1.processInput(input)
    neuron2.processInput(neuron1.a)

    predict = classify(neuron2.a)

    # generate confusion matrix of test results
    predict = pd.Series(predict[0, :], name='Predicted')
    confusion = pd.crosstab(actual, predict, margins=True)
    print confusion

    # plot error
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=[8, 8])
    ax.plot(np.arange(0, len(ce_t)), ce_t, label='Training')
    ax.plot(np.arange(0, len(ce_v)), ce_v, label='Validation')
    ax.set_ylabel('Average Cross Entropy Error')
    ax.set_xlabel('Number of Batch Iterations')
    ax.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()

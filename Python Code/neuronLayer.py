# ----------------------------------------------------------------------------------------------------------------------
# import statements
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import NLObjects as nl
import DataPrep as dp
import Split as sp
# ----------------------------------------------------------------------------------------------------------------------
# Data pre-processing
# ----------------------------------------------------------------------------------------------------------------------
# normalize data and split dataset into training, validation, testing sets
def main():
    # import dataset
    glass = dp.GlassImport()
    num_inputs = glass.shape[1] - 2
    # split into training, validate, testing sets
    train, validate, test = sp.split(glass, num_inputs, t=0.7, v=0.15)

    # specify number of neurons in each layer and the learning rate
    num_neurons1 = 10
    num_neurons2 = 2
    alpha = 0.1
    epoch = 30
    nlayer1 = nl.NeuronLayer(num_neurons1, num_inputs, nl.tansig, nl.j_tansig)
    nlayer2 = nl.NeuronLayer(num_neurons2, num_neurons1, nl.softmax, nl.j_softmax)
    np.random.seed(0)
    # initialize weight and bias randomly for each layer from -0.5 to 0.5
    # W1 number of columns matches training set columns, less final two
    W1 = np.matrix(np.random.rand(num_neurons1, num_inputs) - 0.5)
    b1 = np.matrix(np.random.rand(num_neurons1, 1) - 0.5)
    W2 = np.matrix(np.random.rand(num_neurons2, num_neurons1) - 0.5)
    b2 = np.matrix(np.random.rand(num_neurons2, 1) - 0.5)

    # initialize cross entropy lists for training and validation sets
    ce_t = []
    ce_v = []
    # Initialize weight and bias
    nlayer1.setWeightBias(w=W1, b=b1)
    nlayer2.setWeightBias(w=W2, b=b2)
# ----------------------------------------------------------------------------------------------------------------------
# training the network
# ----------------------------------------------------------------------------------------------------------------------
    for j in range(epoch):
        for i in range(len(train)):
            # create input matrix from training dataset
            input = np.matrix(train.iloc[i, slice(0,num_inputs)]).transpose()
            # process network
            nlayer1.FP(input)
            nlayer2.FP(nlayer1.a)
            # calculate error of each iteration and update cost total
            target = np.matrix(train.iloc[i, -2:])
            e = nl.cross_entropy(nlayer2.a, target)
            if i == 0:
                e_all = e
            else:
                e_all = np.concatenate((e_all, e), axis=1)
            e = np.concatenate([e, e])

            # calculate layer 2 sensitivity
            s2 = nl.senseo(t=target, a=nlayer2.f(nlayer2.n))
            if i == 0:
                s2_all = s2
            else:
                s2_all = np.concatenate((s2_all, s2), axis=1)

            # calculate layer 1 sensitivity
            s1 = nl.senseh(F_prime=nlayer1.j(nlayer1.a), W=W2, s=s2)
            if i == 0:
                s1_all = s1
            else:
                s1_all = np.concatenate((s1_all, s1), axis=1)

        # append average cross entropy to list
        ce_t.append(e_all.mean())

        # update weight and bias for layer 2
        nlayer2.update(sensitivity=s2_all.mean(axis=1), learning_rate=alpha)
        # update weight and bias for layer 1
        nlayer1.update(sensitivity=s1_all.mean(axis=1), learning_rate=alpha)

        # validation
        input = np.matrix(validate.iloc[:, slice(0,num_inputs)]).transpose()
        # first layer
        nlayer1.FP(input)
        nlayer2.FP(nlayer1.a)
        target = np.matrix(validate.iloc[:, -2:])
        # compute errors
        for i in range(len(target)):
            e = nl.cross_entropy(nlayer2.a[:, i], target[i])
            if i == 0:
                e_all = e
            else:
                e_all = np.concatenate((e_all, e), axis=1)

        # append average validation cross entropy to list
        ce_v.append(e_all.mean())
        if j >= 5 and ce_v[j] > ce_v[j - 1]:
            break
# ----------------------------------------------------------------------------------------------------------------------
# Test and evaluate the network
# ----------------------------------------------------------------------------------------------------------------------
    # initialize confusion matrix series
    actual = pd.Series(test.iloc[:, -2], name='Actual')

    # create input matrix from test dataset
    input = np.matrix(test.iloc[:, slice(0,num_inputs)]).transpose()
    nlayer1.FP(input)
    nlayer2.FP(nlayer1.a)

    predict = nl.classify(nlayer2.a)
    predict = np.array(predict).flatten()
    # generate confusion matrix of test results
    predict = pd.Series(predict, name='Predicted')
    confusion = pd.crosstab(actual, predict, margins=True)
    print confusion

    # plot error
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=[8, 8])
    ax.plot((np.arange(0, len(ce_t))), ce_t, label='Training')
    ax.plot((np.arange(0, len(ce_v))), ce_v, label='Validation')
    ax.set_ylabel('Average Cross Entropy Error')
    ax.set_xlabel('Number of Batch Iterations')
    ax.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()

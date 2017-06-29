# ----------------------------------------------------------------------------------------------------------------------
# Import statements
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import NLObjects as MLP
from DataPrep import GlassImport
from Split import split
# ----------------------------------------------------------------------------------------------------------------------
# Data pre-processing
# ----------------------------------------------------------------------------------------------------------------------
# import the glass dataset
glass = GlassImport()
# return the dimension of each input, R
dim_inputs = glass.shape[1] - 2
# split into training, validation, & testing sets
train, validate, test = split(glass, dim_inputs, t=0.7, v=0.15)
# number of neurons in layer 1
num_neurons1 = 10
# number of neurons in layer 2
num_neurons2 = 2
# learning rate
alpha = 0.1
# num_iter
epoch = 150
# instantiate layer 1
nlayer1 = MLP.NeuronLayer(num_neurons1, dim_inputs, np.tanh, MLP.j_tansig)
# instantiate layer 2
nlayer2 = MLP.NeuronLayer(num_neurons2, num_neurons1, MLP.softmax, MLP.j_softmax)
# set seed for reproducible results
np.random.seed(150)
# randomly initialize weight and bias on the interval [-0.5, 0.5]
W1 = np.matrix(np.random.rand(num_neurons1, dim_inputs) - 0.5)
b1 = np.matrix(np.random.rand(num_neurons1, 1) - 0.5)
W2 = np.matrix(np.random.rand(num_neurons2, num_neurons1) - 0.5)
b2 = np.matrix(np.random.rand(num_neurons2, 1) - 0.5)
# pass the randomly initialized weight and bias onto the network object
nlayer1.setWeightBias(w=W1, b=b1)
nlayer2.setWeightBias(w=W2, b=b2)
# ------------------- training set -------------------------
# matrix of inputs
p_train = train.iloc[:, range(dim_inputs)].T
p_train = np.asmatrix(p_train)
# matrix of targets
target = train.iloc[:, -2:]
target = np.asmatrix(target)
# ------------------- validation set -------------------------
# matrix of inputs
p_val = validate.iloc[:, range(dim_inputs)].T
p_val = np.asmatrix(p_val)
# matrix of targets
target_val = validate.iloc[:, -2:]
target_val = np.asmatrix(target_val)
# initialize cross entropy lists for training and validation sets
ce_t = []
ce_v = []
# ----------------------------------------------------------------------------------------------------------------------
# training the network
# ----------------------------------------------------------------------------------------------------------------------
for i in range(epoch):
    # ------------------- training set --------------------
    for q in range(len(train)):
        # propagate the inputs forward ---------------
        nlayer1.FP(p_train[:, q])
        nlayer2.FP(nlayer1.a)
        # calculate the error ------------
        ce = MLP.cross_entropy(nlayer2.a, target[q, :])
        if i == 0:
            e_all = ce
        else:
            e_all = np.concatenate((e_all, ce), axis=1)
        ce = np.concatenate([ce, ce])
        # backprop ----------------
        # layer 2 sensitivity
        s2 = MLP.senseo(t=target[q, :], a=nlayer2.f(nlayer2.n))
        if i == 0:
            s2_all = s2
        else:
            s2_all = np.concatenate((s2_all, s2), axis=1)
        # layer 1 sensitivity
        s1 = MLP.senseh(F_prime=nlayer1.j(nlayer1.a), W=W2, s=s2)
        if i == 0:
            s1_all = s1
        else:
            s1_all = np.concatenate((s1_all, s1), axis=1)
    # ------------------- update weight and bias ----------------
    nlayer2.update(sensitivity=s2_all.mean(axis=1), learning_rate=alpha)  # layer 1
    nlayer1.update(sensitivity=s1_all.mean(axis=1), learning_rate=alpha)  # layer 2
    ce_t.append(e_all.mean())
    # ------------------- validation set --------------------
    for q in range(len(validate)):
        # propagate the inputs forward
        nlayer1.FP(p_val[:, q])
        nlayer2.FP(nlayer1.a)
        # calculate the error ------------
        ce = MLP.cross_entropy(nlayer2.a, target_val[q, :])
        if i == 0:
            e_all = ce
        else:
            e_all = np.concatenate((e_all, ce), axis=1)
        ce = np.concatenate([ce, ce])
    ce_v.append(e_all.mean())
    # early stopping
    # if i == 0:
    #     val_fail = []
    # elif ce_v[i] > ce_v[i-1]:
    #     val_fail.append(1)
    #     if len(val_fail) == 5:
    #         print 'Validation error has increased for 5 consecutive epochs. Early stopping at epoch {}'.format(i)
    #         break
    # else:
    #     val_fail = []
# ----------------------------------------------------------------------------------------------------------------------
# Test and evaluate the network
# ----------------------------------------------------------------------------------------------------------------------
# matrix of inputs
p_test = np.matrix(test.iloc[:, range(dim_inputs)]).T
# ------------------- Test -------------------
# propagate inputs forward
nlayer1.FP(p_test)
nlayer2.FP(nlayer1.a)
# classify inputs based on network output probabilities
predict = MLP.classify(nlayer2.a)
predict = np.array(predict).flatten()
# ------------------- Generating a Confusion Matrix -------------
# initialize confusion matrix series
actual = pd.Series(test.iloc[:, -2], name='Actual')
predict = pd.Series(predict, name='Predicted')
confusion = pd.crosstab(actual, predict, margins=True)
print confusion
# ------------------- Plot Cross Entropy Loss for Training and Validation Sets -------------
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=[8, 8])
ax.plot((np.arange(0, len(ce_t))), np.log(ce_t), label='Training')
ax.plot((np.arange(0, len(ce_v))), np.log(ce_v), label='Validation')
ax.set_ylabel('Log Cross-Entropy Loss')
ax.set_xlabel('Epochs')
ax.legend()
plt.grid()
plt.show()

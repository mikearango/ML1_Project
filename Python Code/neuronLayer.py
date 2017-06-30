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


def main():
    # --------------------- import data ---------------------
    glass = dp.GlassImport()
    num_inputs = glass.shape[1] - 2
    # split into training, validation & testing data
    train, validate, test = sp.split(glass, num_inputs, t=0.7, v=0.15)
    # --------------------- specify network architecture ---------------------
    num_neurons1 = 10  # layer 1
    num_neurons2 = 2   # layer 2
    alpha = 0.01       # learning rate
    epoch = 50        # number of iterations
    nlayer1 = nl.NeuronLayer(num_neurons1, num_inputs, nl.tansig, nl.j_tansig)      # instantiate layer 1
    nlayer2 = nl.NeuronLayer(num_neurons2, num_neurons1, nl.softmax, nl.j_softmax)  # instantiate layer 2
    np.random.seed(0)
    # randomly initialize weight and bias on the interval [-0.5, 0.5]
    W1 = np.matrix(np.random.rand(num_neurons1, num_inputs) - 0.5)
    b1 = np.matrix(np.random.rand(num_neurons1, 1) - 0.5)
    W2 = np.matrix(np.random.rand(num_neurons2, num_neurons1) - 0.5)
    b2 = np.matrix(np.random.rand(num_neurons2, 1) - 0.5)
    # initialize cross entropy loss (training & validation)
    ce_t = []
    ce_v = []
    global s1_all, s2_all
    # pass on randomly initialized weights and biases to the network
    nlayer1.setWeightBias(w=W1, b=b1)
    nlayer2.setWeightBias(w=W2, b=b2)
# ----------------------------------------------------------------------------------------------------------------------
# training the network
# ----------------------------------------------------------------------------------------------------------------------
    for j in range(epoch):
        for i in range(len(train)):
            # create input matrix from training dataset
            input = np.matrix(train.iloc[i, slice(0, num_inputs)]).transpose()
            # --------------------- propagate the inputs forward ---------------------
            nlayer1.FP(input)
            nlayer2.FP(nlayer1.a)
            # --------------------- calculate errors ---------------------
            target = np.matrix(train.iloc[i, -2:])
            e = nl.cross_entropy(nlayer2.a, target)
            if i == 0:
                e_all = e
            else:
                e_all = np.concatenate((e_all, e), axis=1)
            # --------------------- backpropagate sensitivities ---------------------
            s2 = nl.senseo(t=target, a=nlayer2.f(nlayer2.n))  # layer 2 sensitivity
            if i == 0:
                s2_all = s2
            else:
                s2_all = np.concatenate((s2_all, s2), axis=1)
            s1 = nl.senseh(F_prime=nlayer1.j(nlayer1.a), W=W2, s=s2)  # layer 1 sensitivity
            if i == 0:
                s1_all = s1
            else:
                s1_all = np.concatenate((s1_all, s1), axis=1)
        # --------------------- cross-entropy loss ---------------------
        ce_t.append(e_all.mean())
        # --------------------- update weights and biases ---------------------
        nlayer2.update(sensitivity=s2_all.mean(axis=1), learning_rate=alpha)  # layer 2 update
        nlayer1.update(sensitivity=s1_all.mean(axis=1), learning_rate=alpha)  # layer 1 update
# ----------------------------------------------------------------------------------------------------------------------
# Validating the network
# ----------------------------------------------------------------------------------------------------------------------
        input = np.matrix(validate.iloc[:, slice(0, num_inputs)]).transpose()
        # --------------------- propagate inputs forward ---------------------
        nlayer1.FP(input)
        nlayer2.FP(nlayer1.a)
        target = np.matrix(validate.iloc[:, -2:])
        # --------------------- compute errors ---------------------
        for i in range(len(target)):
            e = nl.cross_entropy(nlayer2.a[:, i], target[i])
            if i == 0:
                e_all = e
            else:
                e_all = np.concatenate((e_all, e), axis=1)
        ce_v.append(e_all.mean())
        # --------------------- Early Stopping Condition ---------------------
        if j == 0:
            val_fail = []
        elif ce_v[j] > ce_v[j-1]:
            val_fail.append(1)
            if len(val_fail) == 5:
                print 'Validation error has increased for 5 consecutive epochs. Early stopping at epoch {}'.format(j)
                break
        else:
            val_fail = []
# ----------------------------------------------------------------------------------------------------------------------
# Test and evaluate the network
# ----------------------------------------------------------------------------------------------------------------------
    # --------------------- confusion matrix ---------------------
    actual = pd.Series(test.iloc[:, -2], name='Actual')  # actual values (targets)
    input = np.matrix(test.iloc[:, slice(0, num_inputs)]).transpose()  # network inputs, p
    nlayer1.FP(input)  # layer 1 net-input
    nlayer2.FP(nlayer1.a)  # layer 2 net-input
    predict = nl.classify(nlayer2.a)  # predicted values from network
    predict = np.array(predict).flatten()
    predict = pd.Series(predict, name='Predicted')
    confusion = pd.crosstab(actual, predict, margins=False)  # create confusion matrix
    confusion = confusion.astype(float) # convert values to floats
    print confusion  # output confusion matrix to the console
    # --------------------- accuracy metrics ---------------------
    ERR = (confusion.loc[0, 1] + confusion.loc[1, 0]) / len(predict)
    ACC = 1 - ERR
    FPR = confusion.iloc[0, 1] / (confusion.iloc[0, 1] + confusion.iloc[0, 0])
    TPR = confusion.iloc[1, 1] / (confusion.iloc[1, 0] + confusion.iloc[1, 1])
    print 'Accuracy: %.2f' % ACC
    print 'Error: %.2f' % ERR
    print 'False Positive Rate: %.2f' % FPR
    print 'True Positive Rate: %.2f' % TPR
    # --------------------- plot confusion matrix ---------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(confusion, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(x=j, y=i,
                    s=confusion.iloc[i, j].astype(int),
                    va='center', ha='center')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix of Test Set Predictions')
    # --------------------- plot log(cross-entropy loss) ---------------------
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=[8, 8])
    ax.plot((np.arange(0, len(ce_t))), np.log(ce_t), label='Training', linewidth=2.0, color='blue')
    ax.plot((np.arange(0, len(ce_v))), np.log(ce_v), label='Validation', linewidth=2.0, color='green')
    ax.set_ylabel('Log Cross Entropy Error')
    ax.set_xlabel('Epochs')
    ax.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()

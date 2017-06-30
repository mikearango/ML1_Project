import numpy as np
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

    def FP(self, p):
        self.p = p
        self.n = self.w * self.p + self.b
        self.a = self.f(self.n)
        return

    def update(self, learning_rate, sensitivity):
        self.w = self.w - learning_rate * sensitivity * self.p.T
        self.b = self.b - learning_rate * sensitivity
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
    e_pos = np.exp(n)
    e_neg = np.exp(-n)
    return (e_pos - e_neg) / (e_pos + e_neg)

# softmax transfer function a = e^n/sum(e^n)
def softmax(n):
    return np.exp(n) / np.sum(np.exp(n), axis=0)

# classify output
def classify(a):
    c = np.where(a[0] >= 0.5, 0, 1)
    return c

# tansig derivative jacobian matrix
def j_tansig(a):
    tg_diags = 1 - np.power(np.tanh(a),2)
    tg = np.zeros((a.shape[0], a.shape[0]))
    np.fill_diagonal(tg, tg_diags)
    return tg

# softmax derivative jacobian matrix
def j_softmax(a):
    jacobian = np.zeros((a.shape[0], a.shape[0]))
    for i in range(jacobian.shape[0]):
        for j in range(jacobian.shape[0]):
            if i == j:
                jacobian[i,j] = a[i] * (1 - a[i])
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
    s = F_prime * W.T * s
    return s


# calculate output layer sensitivity
def senseo(t, a):
    s = (a - t)
    return s

# update weights and biases
def update(weight_old, bias_old, sensitivity, input, learning_rate):
    weight_new = weight_old - learning_rate * sensitivity * input.T
    bias_new = bias_old - learning_rate * sensitivity
    return weight_new, bias_new


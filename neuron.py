import numpy as np
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

    # hyperbolic tangent sigmoid transfer function
    def tangsig(self):
        a = 2/(1 + np.exp(-2 * self.net_input())) - 1
        return a


# define target function to approximate
def target(p):
    g = 1 + np.sin(np.pi/2*p)
    return g

# define error calculation function
def error_calc(output, target):
    e = target - output
    return e

# define learning rule function
def learn(weight_old, bias_old, sensitivity, input, learning_rate):
    weight_new = weight_old - learning_rate * sensitivity * np.transpose(input)
    bias_new = bias_old - learning_rate * sensitivity
    return  weight_new, bias_new


def main():
    # range to sample
    p = np.arange(-2,2.2,0.1)

    # specify number of neurons in each layer and the learning rate
    num_neurons1 = 10
    num_neurons2 = 1
    alpha = 0.2

    # initialize weight and bias randomly for each layer from -0.5 to 0.5
    W1 = np.matrix(np.random.rand(num_neurons1,1) - 0.5)
    b1 = np.matrix(np.random.rand(num_neurons1,1) - 0.5)
    W2 = np.matrix(np.random.rand(num_neurons2,num_neurons1) - 0.5)
    b2 = np.matrix(np.random.rand(num_neurons2,1) - 0.5)

    # first layer, inputting random sample from p
    for epoch in range(5000):
        p_sample = p[np.random.randint(0,len(p))]

        neuron1 = Neuron(input=p_sample,weight=W1,bias=b1)
        a1 = neuron1.logsig()
        # second layer
        neuron2 = Neuron(input=a1,weight=W2,bias=b2)
        a2 = neuron2.purelin()

        # calculate error of each iteration (e = t - a)
        e = target(p_sample) - a2

        # create jacobian matrix of layer 2 transfer function derivative
        # softmax uses off-diagonral values so iteration required to create matrix
        F2_prime = np.empty([num_neurons2, num_neurons2])
        for i in range(num_neurons2):
            for j in range(num_neurons2):
                if i == j:
                    F2_prime[i, j] = a2[i] * np.sum(a2 - a2[i])
                else:
                    F2_prime[i, j] = -a2[i] * a2[j]
        # calculate layer 2 sensitivity s2 = -2*F2'(n2)*(e)
        s2 = -2*F2_prime*e

        # create jacobian matrix of first layer transfer function derivative
        F1_prime = np.diag(np.ones(num_neurons1)) - np.diagflat(np.power(a1, 2))
        # calculate layer 1 sensitivity s1 = F1'(n1)*W2t*s2
        s1 = F1_prime * np.transpose(W2) * s2

        # calculate new weight and bias for layer 2
        W2, b2 = learn(weight_old=W2,bias_old=b2,sensitivity=s2,input=a1,learning_rate=alpha)
        # calculate new weight and bias for layer 1
        W1, b1 = learn(weight_old=W1,bias_old=b1,sensitivity=s1,input=p_sample,learning_rate=alpha)

    # calculate neuron outputs for whole dataset with trained network
    neuron1 = Neuron(input=p,weight=W1,bias=b1)
    a1 = neuron1.logsig()
    neuron2 = Neuron(input=a1,weight=W2,bias=b2)
    a2 = neuron2.purelin()


if __name__ == '__main__':
    main()
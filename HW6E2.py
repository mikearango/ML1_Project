import numpy as np
import math
import matplotlib.pyplot as plt

def trainNetwork(p, t, S, eta, _iter):
    def logsigDerivative(x):
        return (1 - x) * (x);

    def purelinDerivative(x):
        return x;

    W = (.5 - -.5) * np.random.random((2, S)) - .5
    # W = np.array([[-0.27,-0.41],[0.09,-0.17]])
    b_l1 = np.random.random((S))
    # b_l1 = np.array([-0.48, -0.13])
    b_l2 = np.random.random()
    # b_l2 = 0.48
    pShuffle = np.random.randint(0,p.shape[0],p.shape[0])
    errors = []
    for i in range(0,_iter):
        err = 0
        for pI in pShuffle:
            temp = (W[0,:] * p[pI]) + b_l1

            a1 =1/(1 + np.exp(-1 * temp))

            a2 = np.dot(W[1,:],a1)+b_l2
            e = t[pI] - a2
            s2 = -2 * purelinDerivative(e)

            Temp = np.zeros((a1.shape[0],a1.shape[0]))
            for i in range(0, a1.shape[0]):
                Temp[i,i] = logsigDerivative(a1[i])

            s1 = (np.dot(Temp,np.transpose(W[1,:]))* s2)

            W[1,:] = W[1,:] - eta * s2 * a1
            b_l2 = b_l2 - eta * s2

            W[0,:]= W[0,:]- (eta * s1 * p[pI])
            b_l1 = b_l1 - (eta * s1)
            err += e
        errors.append(err / p.shape[0])
    return (W, b_l1, b_l2, errors)

def testNetwork(p, t, S, W, b_l1, b_l2):
    err = 0
    a = []
    for i in range(0,p.shape[0]):
        temp = (W[0,:] * p[i]) + b_l1

        a1 =1/(1 + np.exp(-1 * temp))

        a2 = np.dot(W[1,:],a1)+b_l2
        e = t[i] - a2
        err += e
        a.append(a2)
    return (a, err / p.shape[0])

def hideXticks(r, c):
    plt.setp(ax[r,c].get_xticklabels(), visible=False)
    plt.setp(ax[r,c].get_xticklines(), visible=False)
    plt.setp(ax[r,c].get_xticklines()[1::2], visible=False)

def testPhase(phase, p_training, t_training, sNeurons, learningRate, iTerations):
    (Weights, BiasL1, BiasL2, Error) = trainNetwork(p_training, t_training, sNeurons, learningRate, iTerations)
    ax[phase,0].plot(range(1, len(Error) + 1), Error, marker='o')
    ax[phase,0].set_title("Convergence for S=" + str(sNeurons) + " lr=" + str(learningRate))
    hideXticks(phase,0)
    (Actual,Error)=testNetwork(p_training, t_training,sNeurons,Weights,BiasL1,BiasL2)
    ax[phase,1].plot(p_training, Actual, color='red', marker='o')
    ax[phase,1].plot(p_training, t_training, color='blue',marker='x')
    ax[phase,1].set_title("Mapping for S=" + str(sNeurons) + " lr=" + str(learningRate) + "E=" + str(Error))
    hideXticks(phase, 1)
    return Error

np.random.seed(1)
learningRate = 0.1
sNeurons = 2
iTerations = 100
p_training = np.arange(-2, 2, .1)
t_training = 1 + np.sin(math.pi / 2 * p_training)

fig, ax = plt.subplots(nrows=8, ncols=2, figsize=(20,20))
error = testPhase(0, p_training,t_training,2,.0001,iTerations)
error = testPhase(1, p_training,t_training,10,.0001,iTerations)
error = testPhase(2, p_training,t_training,2,.001,iTerations)
error = testPhase(3, p_training,t_training,10,.001,iTerations)
error = testPhase(4, p_training,t_training,2,.01,iTerations)
error = testPhase(5, p_training,t_training,10,.01,iTerations)
error = testPhase(6, p_training,t_training,2,.1,iTerations)
error = testPhase(7, p_training,t_training,10,.1,iTerations)
plt.show()



print('stop')


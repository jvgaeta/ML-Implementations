from __future__ import division  # floating point division
import math
import numpy as np
from scipy.optimize import fmin,fmin_bfgs
import random

# File with useful utility functions

def mean(numbers):
    return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def calculateprob(x, mean, stdev):
    if stdev < 1e-3:
        if math.fabs(x-mean) < 1e-2:
            return 1.0
        else:
            return 0
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
    
def sigmoid(xvec):
    """ Compute the sigmoid function """
    # Cap -xvec, to avoid overflow
    # Undeflow is okay, since it get set to zero
    xvec[xvec < -100] = -100
   
    vecsig = 1.0 / (1.0 + np.exp(np.negative(xvec)))
 
    return vecsig

def single_val_sigmoid(value):
    if value < -100:
        value = -100

    return 1.0 / (1.0 + np.exp(value * -1))

def dsigmoid(xvec):
    """ Gradient of standard sigmoid 1/(1+e^-x) """
    vecsig = sigmoid(xvec)
    return vecsig * (1 - vecsig)


def threshold_probs(probs):
    """ Converts probabilities to hard classification """
    classes = np.ones(len(probs),)
    classes[probs < 0.5] = 0
    return classes
            
def fmin_simple(loss, grad, initparams):
    """ Lets just call fmin_bfgs, so we can better trust the optimizer """
    return fmin_bfgs(loss,initparams,fprime=grad)                

def logsumexp(a):
    """
    Compute the log of the sum of exponentials of input elements.
    Modified scipys logsumpexp implemenation for this specific situation
    """

    awithzero = np.hstack((a, np.zeros((len(a),1))))
    maxvals = np.amax(awithzero, axis=1)
    aminusmax = np.exp((awithzero.transpose() - maxvals).transpose())

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        out = np.log(np.sum(aminusmax, axis=1))

    out = np.add(out,maxvals)

    return out

def custom_probs(xvec):
    prob = (1/2) * (1 + (xvec / np.sqrt(1 + np.square(xvec))))
    return prob

def custom_probs_single_val(x):
    prob = (1 / 2) * (1 + (x / np.sqrt(1 + x**2)))
    return prob

def custom_sign(xvec):
    new_vec = xvec
    d = {1: 1, -1:-1}
    for i in range(0, len(xvec)):
        if new_vec[i] == 0:
            new_vec[i] = random.choice(d.keys())

    new_vec = np.sign(new_vec)

    return new_vec



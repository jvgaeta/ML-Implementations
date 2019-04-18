from __future__ import division  # floating point division
import numpy as np
import utilities as utils
import random as rand

# generic classifier others will inherit from
class Classifier:  
    def __init__( self, params=None ):
        pass

    def learn(self, Xtrain, ytrain):
        pass
        
    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, params=None ):
        self.weights = None
        if params is not None and 'regwgt' in params:
            self.regwgt = params['regwgt']
        else:
            self.regwgt = .01
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1
        
        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.inv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.regwgt*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples
        
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1     
        ytest[ytest < 0] = 0    
        return ytest
        
class NaiveBayes(Classifier):
    
    def __init__( self, params=None ):
        self.usecolumnones = True
        if params is not None:
            self.usecolumnones = params['usecolumnones']
        self.mv0 = []
        self.mv1 = []
        self.freq0 = 0
        self.freq1 = 0


    def split(self, Xtrain, ytrain):
        zero = []
        one = []
        # split into two separate arrays for whether it evaluates to 0 or 1
        for i in range(0, len(ytrain)):
            if (ytrain[i] == 0):
                zero.append(Xtrain[i])
            else:
                one.append(Xtrain[i])
        return zero, one


    # compute the mean and variance for each feature
    def calculate_mv(self, zero, one):
        feature_array = []
        for i in range(0, len(zero[0])):
            feature_array = []
            for j in range(0, len(zero)):
                feature_array.append(zero[j][i])
            self.mv0.append([utils.mean(feature_array), utils.stdev(feature_array)])

        for i in range(0, len(one[0])):
            feature_array = []
            for j in range(0, len(one)):
                feature_array.append(one[j][i])
            self.mv1.append([utils.mean(feature_array), utils.stdev(feature_array)])

    # computes p(y) for both one and zero
    def compute_freq(self, ytrain):
        zero_count = 0
        one_count = 0
        for i in range(0, len(ytrain)):
            if (ytrain[i] == 0):
                zero_count += 1
            else:
                one_count += 1
        self.freq0 = zero_count / len(ytrain)
        self.freq1 = one_count / len(ytrain)




    def learn(self, Xtrain, ytrain):
        # split the training data
        zero, one = self.split(Xtrain, ytrain)
        # compute mean and variances
        self.calculate_mv(zero, one)
        # compute freq0, freq1 (p(y))
        self.compute_freq(ytrain)

    # compute the better probability and select either 0 or 1
    def calculate_best(self, x):
        
        probability1 = 1
        probability0 = 1
        # computing product of all p(x_i|y)
        for i in range(0, len(x)):
            # probability of zero happening, ignoring the denom and the 1/2 at the beginning since they are the same for both
            probability0 *= utils.calculateprob(x[i], self.mv0[i][0], self.mv0[i][1])
            # probability of one happening, ignoring the denom and the 1/2 at the beginning since they are the same for both
            probability1 *= utils.calculateprob(x[i], self.mv1[i][0], self.mv1[i][1])

        # now multiplty by p(y)
        probability0 *= self.freq0
        probability1 *= self.freq1
        # pick the larger of the two probabilities
        if (probability0 > probability1):
            return 0
        else:
            return 1


    def predict(self, Xtest):
        ytest = []

        # exclude the column of ones in our prediction, it doesn't change anything
        if self.usecolumnones is False:
            Xtest = np.delete(Xtest, -1, 1)

        # just append either a 0 or 1 depending on the computed probability comparison
        for i in range(0, len(Xtest)):
            ytest.append(self.calculate_best(Xtest[i]))
        return ytest
            
    
class LogitReg(Classifier):
    """ Logistic regression; need to complete the inherited learn and predict functions """

    def __init__( self, params=None ):
        self.weights = None
        self.regwgt = 50


    def learn(self, Xtrain, ytrain):
        tolerance = .001
        w = np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T, Xtrain)), Xtrain.T), ytrain)
        w1 = [i for i in range(10, Xtrain.shape[1] + 10)]
        probs = []
        P = [[0 for i in range(0, len(Xtrain))] for j in range(0, len(Xtrain))]

        while np.linalg.norm(w - w1) > tolerance:
            probs = []
            for i in range(0, len(Xtrain)):
                probs.append(utils.single_val_sigmoid(np.dot(w.T, Xtrain[i])))
            for i in range(0, len(probs)):
                P[i][i] = probs[i]
            w1 = w
            # weight update rule, vanilla
            w = w + np.dot(np.dot(np.linalg.pinv(np.dot(np.dot(np.dot(Xtrain.T, P), np.subtract(np.identity(len(probs)), P)), Xtrain)), Xtrain.T), np.subtract(ytrain, probs))
        self.weights = w


    def predict(self, Xtest):
        probabilities = utils.sigmoid(np.dot(self.weights, Xtest.T))
        return utils.threshold_probs(probabilities) 



class NeuralNet(Classifier):
    """Two-layer neural network"""
    
    def __init__(self, params=None):
        # Number of input, hidden, and output nodes
        # Hard-coding sigmoid transfer for this implementation for simplicity
        self.ni = params['ni']
        self.nh = params['nh']
        self.no = params['no']
        self.transfer = utils.sigmoid
        self.dtransfer = utils.dsigmoid

        # Set step-size
        self.stepsize = .001

        # Number of repetitions over the dataset
        self.reps = 100
        
        # Create random {0,1} weights to define features
        self.wi = np.random.randint(2, size=(self.nh, self.ni))
        self.wo = np.random.randint(2, size=(self.no, self.nh))

    def learn(self, Xtrain, ytrain):
        """ Incrementally update neural network using stochastic gradient descent """        
        for reps in range(self.reps):
            for samp in range(Xtrain.shape[0]):
                self.update(Xtrain[samp,:],ytrain[samp])

    def predict(self, Xtest):
        ytest = []
        for i in range(0, len(Xtest)):
            ytest.append(self.evaluate(Xtest[i]))
        ytest = np.array(ytest)
        return utils.threshold_probs(ytest)

    def evaluate(self, inputs):
        if inputs.shape[0] != self.ni:
            raise ValueError('NeuralNet:evaluate -> Wrong number of inputs')

        # hidden activations
        ah = np.ones(self.nh)
        ah = self.transfer(np.dot(self.wi,inputs))  

        # output activations
        ao = np.ones(self.no)
        ao = self.transfer(np.dot(self.wo,ah))
        
        return ao[0]

    def update(self, inp, out):
        nput = np.array(inp).reshape(self.ni, self.no)
        h = utils.sigmoid(np.dot(self.wi, nput))
        yhat = utils.sigmoid(np.dot(self.wo, h))
        # y_i = np.array(out).reshape(1, 1)
        # if we fail, then the result is not accurate 
        delta_i_first = -out / yhat + (1 - out) / (1 - yhat) 
        delta_i = np.dot(np.dot(delta_i_first, yhat), 1 - yhat)
        # delta_i = np.dot(np.dot(self.dtransfer(np.dot(y_i, yhat)), yhat), 1 - yhat)
        self.wo = self.wo - self.stepsize * np.dot(delta_i, h.T)
        hadamard = self.wo.T * h* (1 - h)
        self.wi = self.wi - self.stepsize * np.dot(delta_i*hadamard, nput.T)

class LinearClass(Classifier):
    def __init__( self, params=None ):
        self.weights = None

    def learn(self, Xtrain, ytrain):
        tolerance = .0001
        w = np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T, Xtrain)), Xtrain.T), ytrain)
        w1 = [i for i in range(10, Xtrain.shape[1] + 10)]
        probs = []
        P = [[0 for i in range(0, len(Xtrain))] for j in range(0, len(Xtrain))]
        while np.linalg.norm(w - w1) > tolerance:
            probs = []
            for i in range(0, len(Xtrain)):
                probs.append(utils.custom_probs_single_val(np.dot(w.T, Xtrain[i])))
            for i in range(0, len(probs)):
                P[i][i] = probs[i]
            w1 = w
            # weight update rule, why does this fail at times and very inaccurate with pinv
            w = w + np.dot(np.dot(np.linalg.pinv(np.dot(np.dot(np.dot(Xtrain.T, P), np.subtract(np.identity(len(probs)), P)), Xtrain)), Xtrain.T), np.subtract(ytrain, probs))
        self.weights = w


    def predict(self, Xtest):
        probabilities = utils.custom_probs(np.dot(self.weights, Xtest.T))
        return utils.threshold_probs(probabilities) 

        
            


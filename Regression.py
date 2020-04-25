'''
Created on 24 apr. 2020

@author: George
'''

import random
import numpy as np
from theano.sandbox.cuda.basic_ops import row

class MySGDRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []

    # simple stochastic GD
    def fit(self, x, y, learningRate = 0.001, noEpochs = 1000):
        self.coef_ = [0.0 for _ in range(len(x[0]) + 1)]    #beta or w coefficients y = w0 + w1 * x1 + w2 * x2 + ...
        # self.coef_ = [random.random() for _ in range(len(x[0]) + 1)]    #beta or w coefficients 
        for epoch in range(noEpochs):
            # TBA: shuffle the trainind examples in order to prevent cycles
            for i in range(len(x)): # for each sample from the training data
                ycomputed = self.eval(x[i])     # estimate the output
                crtError = ycomputed - y[i]     # compute the error for the current sample
                for j in range(0, len(x[0])):   # update the coefficients
                    self.coef_[j] = self.coef_[j] - learningRate * crtError * x[i][j]
                self.coef_[len(x[0])] = self.coef_[len(x[0])] - learningRate * crtError * 1

        self.intercept_ = self.coef_[-1]
        self.coef_ = self.coef_[:-1]

    def eval(self, xi):
        yi = self.coef_[-1]
        # intrebare: cand chemam predict dupa fit, nu cumva ultimul coeficient din coef_ nu este cel liber?
        # De vreme ce am facut slice-ul coeficientilor la finalul fit-ului... 
        for j in range(len(xi)):
            yi += self.coef_[j] * xi[j]
        return yi 

    def predict(self, x):
        yComputed = [self.eval(xi) for xi in x]
        return yComputed




class MyBatchRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []
        self.noFeatures = None
    
    def loss(self, theta,X,y): # not used right now (could be useful when plotting the loss function over time)
        '''
        Computes the loss for X and y given.
        @param theta: column vector of equation's parameters 
        @param X: matrix of features - number of entries x number of features
        @param y: column vector of outputs for each entry in the dataset - number of entries x 1
        '''
        m = len(y)
        predictions = X.dot(theta)
        cost = (1/2*m) * np.sum(np.square(predictions-y))
        return cost


    def fit(self, X, y, theta=None, eta=0.2, epochs=1000, batchSize=None):
        self.noFeatures = len(X[0])
        X = np.asarray(X).reshape((-1,self.noFeatures))
        y = np.asarray(y).reshape((-1,1))
        '''
        @param X: matrix of features - number of entries x number of features
        @param y: column vector of outputs for each entry in the dataset - number of entries x 1
        @param theta: column vector of equation parameters used to determine the regression
        @param eta: learning rate used in the process (smaller means better chance to find the optima, but slower convergence)
        @param epochs: number of epochs for the training
        @param batchSize: dimension of a batch - used to divide the dataset into smaller groups and train them simultaneously
        '''
        
        if batchSize is None:
            batchSize = len(y)
        
        if theta is None:
            theta = np.random.randn(3,1)
        m = len(y)
        #n_batches = int(m/batchSize)
        
        for _ in range(epochs):
            indices = np.random.permutation(m) # shuffle the data to avoid cycles
            X = X[indices]
            y = y[indices]
            for i in range(0,m,batchSize): # split the data in batches
                X_i = X[i:i+batchSize]
                y_i = y[i:i+batchSize]
                
                X_i = np.c_[np.ones(len(X_i)),X_i] # append columnwise column vector of ones to X batch (=> the shape of matrix we desire)
               
                prediction = np.dot(X_i,theta)
    
                theta = theta -(1/m)*eta*( X_i.T.dot((prediction - y_i)))

        
        self.intercept_ = theta[-1][0]
        self.coef_ = [coef[0] for coef in theta[:-1]]
    
    def eval(self, xi):
        yi = self.coef_[-1]
        # intrebare: cand chemam predict dupa fit, nu cumva ultimul coeficient din coef_ nu este cel liber?
        # De vreme ce am facut slice-ul coeficientilor la finalul fit-ului...
        for j in range(len(xi)):
            yi += self.coef_[j] * xi[j]
        return yi

    def predict(self, X):
        X = np.asarray(X).reshape((-1,self.noFeatures))
        yComputed = [self.eval(xi) for xi in X]
        return yComputed





def dot(A,B):
    return sum([i*j for (i, j) in zip([K[0] for K in A], B)])

def T(A):
    t_matrix = zip(*A)
    t = []
    for row in t_matrix: 
        t.append(row)
    return t


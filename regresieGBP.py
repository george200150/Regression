'''
Created on 24 apr. 2020

@author: George
'''

#REGRESIE UNIVARIATA


import Regression
from sklearn.metrics.regression import mean_squared_error
from normalizare import statisticalNormalisation

def plotData(x1, y1, x2 = None, y2 = None, x3 = None, y3 = None, title = None):
    plt.plot(x1, y1, 'ro', label = 'train data')
    if (x2):
        plt.plot(x2, y2, 'b-', label = 'learnt model')
    if (x3):
        plt.plot(x3, y3, 'g^', label = 'test data')
    plt.title(title)
    plt.legend()
    plt.show()

from random import seed
import os, numpy as np
from utils import *
from random import seed

# *two hundred plots later*
def fun():
    seed(1)
    crtDir =  os.getcwd()
    filePath = os.path.join(crtDir, 'date.txt')
    
    inputs, outputs = loadDataSingleFeature(filePath, 'Economy..GDP.per.Capita.', 'Happiness.Score')
    print('in:  ', inputs[:5])
    print('out: ', outputs[:5])
    
        
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace = False)
    testSample = [i for i in indexes  if not i in trainSample]
    
    #NORMALIZATION OF TRAIN DATA
    trainInputs = [inputs[i] for i in trainSample]
    trainInputs = statisticalNormalisation(trainInputs)
    trainOutputs = [outputs[i] for i in trainSample]
    trainOutputs = statisticalNormalisation(trainOutputs)
    
    #NORMALIZATION OF TEST DATA
    testInputs = [inputs[i] for i in testSample]
    testInputs = statisticalNormalisation(testInputs)
    testOutputs = [outputs[i] for i in testSample]
    testOutputs = statisticalNormalisation(testOutputs)
    
    
    plotDataHistogram(trainInputs+testInputs, 'Capita GDP')
    
    plotDataHistogram(trainOutputs+testOutputs, 'Happiness score')
    
    plotData2D(trainInputs+testInputs, trainOutputs+testOutputs)
    
    
    xx = [[el] for el in trainInputs]
    
    #regressor = linear_model.LinearRegression()
    #regressor = regression.MyLinearUnivariateRegression()
    #regressor = Regression.MySGDRegression()
    regressor = Regression.MyBatchRegression()
    regressor.fit(xx, trainOutputs) # FIT SINGLE MATRIX of noSamples x noFeatures
    
    w0, w1 = regressor.intercept_, regressor.coef_[0]
    
    feature1 = [el for el in trainInputs]
    feature1train = trainInputs
    
    noOfPoints = 50
    xref1 = []
    val = min(feature1)
    step1 = (max(feature1) - min(feature1)) / noOfPoints
    for _ in range(1, noOfPoints):
        for _ in range(1, noOfPoints):
            xref1.append(val)
        val += step1
            
    yref = [w0 + w1 * el1 for el1 in xref1]
    
    plot2DModel(feature1train, trainOutputs, xref1, yref)
    
    
    
    xx = [[el] for el in testInputs]
    computedTestOutputs = regressor.predict(xx)
    #computedTestOutputs = [w0 + w1 * el for el in testInputs]
    
    noOfPoints = 50
    xref1 = []
    val = min(testInputs)
    step1 = (max(testInputs) - min(testInputs)) / noOfPoints
    for _ in range(1, noOfPoints):
        for _ in range(1, noOfPoints):
            xref1.append(val)
        val += step1
    
    #plot2DModel(feature1test, computedTestOutputs, xref1, yref) # "predictions vs real test data"
    plot2DModel(testInputs, testOutputs, xref1, yref) # "predictions vs real test data"
    #plotData(inputs, outputs, testInputs, computedTestOutputs, testInputs, testOutputs, "predictions vs real test data")
    
    #compute the differences between the predictions and real outputs
    error = 0.0
    for t1, t2 in zip(computedTestOutputs, testOutputs):
        error += (t1 - t2) ** 2
    error = error / len(testOutputs)
    print("prediction error (manual): ", error)
    
    error = mean_squared_error(testOutputs, computedTestOutputs)
    print("prediction error (tool): ", error)


if __name__ == '__main__':
    fun()


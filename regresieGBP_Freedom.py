'''
Created on 24 apr. 2020

@author: George
'''

#REGRESIE  BIVARIATA


import Regression
from sklearn.metrics.regression import mean_squared_error
from Regression import MyBatchRegression

def plot3Ddata(x1Train, x2Train, yTrain, x1Model = None, x2Model = None, yModel = None, x1Test = None, x2Test = None, yTest = None, title = None):
    from mpl_toolkits import mplot3d
    ax = plt.axes(projection = '3d')
    if (x1Train):
        plt.scatter(x1Train, x2Train, yTrain, c = 'r', marker = 'o', label = 'train data') 
    if (x1Model):
        plt.scatter(x1Model, x2Model, yModel, c = 'b', marker = '_', label = 'learnt model') 
    if (x1Test):
        plt.scatter(x1Test, x2Test, yTest, c = 'g', marker = '^', label = 'test data')  
    plt.title(title)
    ax.set_xlabel("capita")
    ax.set_ylabel("freedom")
    ax.set_zlabel("happiness")
    plt.legend()
    plt.show()




from utils import *
from random import seed
import os, numpy as np
from normalizare import statisticalNormalisation



def inputRead():
    crtDir =  os.getcwd()
    filePath = os.path.join(crtDir, 'date.txt')
    
    inputs, outputs = loadData(filePath, ['Economy..GDP.per.Capita.', 'Freedom'], 'Happiness.Score')
    print('in:  ', inputs[:5])
    print('out: ', outputs[:5])
    
    #firsts = [i for i,_ in inputs]
    #seconds = [j for _,j in inputs]
    
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace = False)
    
    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]
    
    testSample = [i for i in indexes  if not i in trainSample]
    testInputs = [inputs[i] for i in testSample]
    testOutputs = [outputs[i] for i in testSample]
    
    #NORMALIZATION OF TEST DATA
    feature1test = [el for el,_ in testInputs]
    feature2test = [el for _,el in testInputs]
    feature1test = statisticalNormalisation(feature1test)
    feature2test = statisticalNormalisation(feature2test)
    testInputs = [[el,el2] for el,el2 in zip(feature1test,feature2test)]
    testOutputs = statisticalNormalisation(testOutputs)
    
    #NORMALIZATION OF TRAIN DATA
    feature1train = [el for el,_ in trainInputs]
    feature2train = [el for _,el in trainInputs]
    feature1train = statisticalNormalisation(feature1train)
    feature2train = statisticalNormalisation(feature2train)
    trainInputs = [[el,el2] for el,el2 in zip(feature1train,feature2train)]
    trainOutputs = statisticalNormalisation(trainOutputs)
    
    return trainInputs, trainOutputs, feature1train, feature2train, testInputs, testOutputs, feature1test, feature2test



# *two hundreds plots later*
def fun():
    seed(1)
    
    trainInputs, trainOutputs, feature1train, feature2train, testInputs, testOutputs, feature1test, feature2test = inputRead()
    
    #plotDataHistogram(feature1train+feature1test, 'Capita GDP')
    #plotDataHistogram(feature2train+feature2test, 'Freedom')
    #plotDataHistogram(trainOutputs+testOutputs, 'Happiness score')
    #plotData3D(trainInputs+testInputs, trainOutputs+testOutputs)
    # uncomment these when visualising data
    
    
    #regressor = linear_model.LinearRegression()
    #regressor = regression.MyLinearUnivariateRegression()
    #regressor = Regression.MySGDRegression()
    regressor = MyBatchRegression()
    regressor.fit(trainInputs, trainOutputs) # FIT SINGLE MATRIX of noSamples x noFeatures
    
    w0, w1, w2 = regressor.intercept_, regressor.coef_[0], regressor.coef_[1] 
    
    #nope. again. nope.
    #feature1 = [el for el,_ in xx]
    #feature2 = [el2 for _,el2 in xx]
    
    
    
    
    noOfPoints = 50
    xref1 = []
    val = min(feature1train)
    step1 = (max(feature1train) - min(feature1train)) / noOfPoints
    for _ in range(1, noOfPoints):
        for _ in range(1, noOfPoints):
            xref1.append(val)
        val += step1
    
    
    xref2 = []
    val = min(feature2train)
    step2 = (max(feature2train) - min(feature2train)) / noOfPoints
    for _ in range(1, noOfPoints):
        for _ in range(1, noOfPoints):
            xref2.append(val)
        val += step2
            
    yref = [w0 + w1 * el1 + w2 * el2 for el1, el2 in zip(xref1, xref2)]
    
    plotModel(feature1train, feature2train, trainOutputs, xref1, xref2, yref)
    
    
    
    
    computedTestOutputs = regressor.predict([[x,y] for x,y in testInputs])

    noOfPoints = 50
    xref1 = []
    val = min(feature1test)
    step1 = (max(feature1test) - min(feature1test)) / noOfPoints
    for _ in range(1, noOfPoints):
        for _ in range(1, noOfPoints):
            xref1.append(val)
        val += step1
    
    
    xref2 = []
    val = min(feature2test)
    step2 = (max(feature2test) - min(feature2test)) / noOfPoints
    for _ in range(1, noOfPoints):
        for _ in range(1, noOfPoints):
            xref2.append(val)
        val += step2
    
    
    plotModel(feature1test, feature2test, computedTestOutputs, xref1, xref2, yref) # "predictions vs real test data"
    
    
    
    noOfPoints = 50
    xref1 = []
    val = min(feature1train)
    step1 = (max(feature1train) - min(feature1train)) / noOfPoints
    for _ in range(1, noOfPoints):
        for _ in range(1, noOfPoints):
            xref1.append(val)
        val += step1
    
    xref2 = []
    val = min(feature2train)
    step2 = (max(feature2train) - min(feature2train)) / noOfPoints
    for _ in range(1, noOfPoints):
        aux = val
        for _ in range(1, noOfPoints):
            xref2.append(aux)
            aux += step2
    yref = [w0 + w1 * el1 + w2 * el2 for el1, el2 in zip(xref1, xref2)]
    plot3Ddata(feature1train, feature2train, trainOutputs, xref1, xref2, yref, [], [], [], 'train data and the learnt model')
    
    
    
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
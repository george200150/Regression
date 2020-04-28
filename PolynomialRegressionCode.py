'''
Created on 25 apr. 2020
@author: George
'''
import pandas as pd
#import Mynumpy as np
#from Mynumpy import linspace,\
#c_ as columnAppend, ones, square, sum, power
from numpy import linspace,\
c_ as columnAppend, ones, square, sum, power
from numpy.linalg import inv
#from Mynumpy import inv
import matplotlib.pyplot as plt


def loadData():
    """
    # Load and plot data files
    """
    # Load the training data
    trainInputFeatures = pd.read_csv('xTrain.dat',  header = None) 
    trainOutputs = pd.read_csv('yTrain.dat', header = None) 
    
    # Load the test data
    testFeatures = pd.read_csv('xTest.dat',  header = None) 
    testRealOutputs = pd.read_csv('yTest.dat', header = None) 
    
    # # Plot training data and real outputs
    trainXFeatures = trainInputFeatures.values
    trainYOutputs = trainOutputs.values
    plt.scatter(trainXFeatures, trainYOutputs, color = 'g', marker = '*', s = 30)
    
    # # Plot training data and real outputs
    plt.scatter(testFeatures.values, testRealOutputs.values, color = 'b', marker = 'o', s = 30)
    plt.title('Training and Testing Data')
    plt.show()
    
    return trainInputFeatures, trainOutputs, testFeatures, testRealOutputs



def trainLinearRegression(trainInputFeatures, trainOutputs):
    """
    # Train linear regression model on training set
    """
    trainXFeatures = trainInputFeatures.values
    trainYOutputs = trainOutputs.values
    
    N = len(trainInputFeatures)
    X = columnAppend[ones(N), trainXFeatures]
    A = inv(X.T@X)
    D = A@X.T
    result = D@trainYOutputs
    
    # Plot scatter plot
    plt.scatter(trainXFeatures, trainYOutputs, color = 'm', marker = 'o', s = 30)
    plotRegressionCurve1(trainXFeatures, trainYOutputs, result)
    
    # Find average error on the training set
    A = square(yPredicted - trainYOutputs)
    error = sum(A) / N
    #print('MSE:')
    print('Train 1st-order regression model on training set')
    print('Average error on the training set: ', error)
    
    return result

yPredicted = []
def plotRegressionCurve1(x, y, b): 
    # Plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "b", marker = "o", s = 30) 

    # Predicted response vector 
    global yPredicted
    yPredicted = b[0] + b[1] * x 
    
    # Plotting the regression line 
    plt.plot(x, yPredicted, color = "orange") 

    # putting labels 
    plt.xlabel('x') 
    plt.ylabel('y') 

    # Function to show plot 
    plt.show() 


def testLR(testFeatures, testRealOutputs, result):
    """
    # Test linear regression model on testing set
    """
    testXFeatures = testFeatures.values
    testYOutputs = testRealOutputs.values
    
    plt.scatter(testXFeatures, testRealOutputs, color = 'b', marker = 'o', s = 30)
    
    plotRegressionCurve2(testXFeatures, testYOutputs, result)
    
    # Find average error on the training set
    A = square(testYPredicted - testYOutputs)
    error = sum(A) / testXFeatures.shape[0]
    print('Average error on the testing set: ', error)
    print('1st-order linear regression')

testYPredicted = []
def plotRegressionCurve2(x, y, b): 
    # Plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "b", marker = "o", s = 30) 

    # Predicted response vector 
    global testYPredicted
    testYPredicted = b[0] + b[1] * x 
    
    # plotting the regression line 
    plt.plot(x, testYPredicted, color = "orange") 

    # Putting labels 
    plt.xlabel('x') 
    plt.ylabel('y') 

    # Function to show plot 
    plt.show() 



def train2OrderPolynomialRegression(trainInputFeatures, trainOutputs):
    """
    # Train 2nd-order regression model on training set
    """
    trainXFeatures = trainInputFeatures.values
    trainYOutputs = trainOutputs.values
    
    # # Plot scatter plot
    # plt.scatter(trainXFeatures, trainYOutputs, color = 'm', marker = 'o', s = 30)
    
    N = len(trainInputFeatures)
    X = columnAppend[ones(N), trainXFeatures, square(trainInputFeatures)]
    A = inv(X.T@X)
    D = A@X.T
    result = D@trainYOutputs
    
    # Plot scatter plot
    plt.scatter(trainXFeatures, trainYOutputs, color = 'm', marker = 'o', s = 30)
    plotRegressionCurve3(trainXFeatures, trainYOutputs, result)
    plt.show() 
    
    # Find average error on the training set
    A = square(result[2] * square(trainXFeatures) + result[1] * trainXFeatures + result[0]  - trainYOutputs)
    error = sum(A) / N
    print('\n\nTrain 2nd-order regression model on training set')
    print('Average error on the training set: ', error)
    return result

yPredicted = []
def plotRegressionCurve3(x, y, b): 
    # plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "b", marker = "o", s = 30) 

    xLine = linspace(x.min(), x.max(), 100)
    global yPredicted
    yPredicted = b[2] * square(xLine) + b[1] * xLine + b[0] 
    regressionCurve = yPredicted
    # Plotting the regression line 
    plt.plot(xLine, regressionCurve, color = "orange") 
    # Putting labels 
    plt.xlabel('x') 
    plt.ylabel('y')     



def test2PR(testFeatures, testRealOutputs, result):
    """
    # Test 2nd-order regression model on testing set
    """
    testXFeatures = testFeatures.values
    testYOutputs = testRealOutputs.values
    
    plt.scatter(testXFeatures, testRealOutputs, color = 'b', marker = 'o', s = 30)
    
    plt.scatter(testXFeatures, testRealOutputs, color = 'b', marker = 'o', s = 30)
    plotRegressionCurve4(testXFeatures, testYOutputs, result)
    plt.show() 
    
    # Find average error on the training set
    A = square(result[2] * square(testXFeatures) + result[1] * testXFeatures + result[0]  - testYOutputs)
    error = sum(A) / testXFeatures.shape[0]
    print('Average error on the testing set: ', error)
    print('2nd-order polynomial regression')

yPredicted = []
def plotRegressionCurve4(x, y, b): 
    # Plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "b", marker = "o", s = 30) 

    xLine = linspace(x.min(), x.max(), 100)
    global yPredicted
    yPredicted = b[2] * square(xLine) + b[1] * xLine + b[0] 
    regressionCurve = yPredicted
    # Plotting the regression line 
    plt.plot(xLine, regressionCurve, color = "orange") 

    # Putting labels 
    plt.xlabel('x') 
    plt.ylabel('y') 



def train3OrderPolynomialRegression(trainInputFeatures, trainOutputs):
    """
    # Train 3rd-order regression model on training set
    """
    trainXFeatures = trainInputFeatures.values
    trainYOutputs = trainOutputs.values
    
    # Plot scatter plot
    plt.scatter(trainXFeatures, trainYOutputs, color = 'm', marker = 'o', s = 30)
    
    N = len(trainInputFeatures)
    X = columnAppend[ones(N), trainXFeatures, square(trainInputFeatures), power(trainInputFeatures, 3)]
    A = inv(X.T@X)
    D = A@X.T
    result = D@trainYOutputs
    
    plotRegressionCurve5(trainXFeatures, trainYOutputs, result)
    plt.show() 
    
    # Find average error on the training set
    A = square(result[3] * power(trainXFeatures, 3) + result[2] * square(trainXFeatures) + result[1] * trainXFeatures + result[0]  - trainYOutputs)
    error = sum(A) / N
    print('\n\nTrained the 3nd-order polynomial model on the training set')
    print('Average error on the training set: ', error)
    
    return result

yPredicted = []
def plotRegressionCurve5(x, y, b):
    # Plotting the actual points as scatter plot
    plt.scatter(x, y, color = "b", marker = "o", s = 30)
    
    xLine = linspace(x.min(), x.max(), 100)
    global yPredicted
    yPredicted = b[3] * power(xLine, 3) + b[2] * square(xLine) + b[1] * xLine + b[0]
    regressionCurve = yPredicted
    # Plotting the regression line
    plt.plot(xLine, regressionCurve, color = "orange")
    # Putting labels
    plt.xlabel('x')
    plt.ylabel('y')




def test3PR(testFeatures, testRealOutputs, result):
    """
    # Test 3rd-order regression model on testing set
    """
    testXFeatures = testFeatures.values
    testYOutputs = testRealOutputs.values
    
    plt.scatter(testXFeatures, testRealOutputs, color = 'b', marker = 'o', s = 30)
    
    
    
    plotRegressionCurve6(testXFeatures, testYOutputs, result)
    plt.show() 
    
    # Find average error on the training set
    A = square(result[3] * power(testXFeatures, 3) + result[2] * square(testXFeatures) + result[1] * testXFeatures + result[0]  - testYOutputs)
    error = sum(A) / testXFeatures.shape[0]
    print('Average error on the testing set: ', error)
    print('3rd-order polynomial regression')

yPredicted = []
def plotRegressionCurve6(x, y, b): 
    # Plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "b", marker = "o", s = 30) 

    xLine = linspace(x.min(), x.max(), 100)
    global yPredicted
    yPredicted = b[3] * power(xLine, 3) + b[2] * square(xLine) + b[1] * xLine + b[0] 
    regressionCurve = yPredicted
    # Plotting the regression line 
    plt.plot(xLine, regressionCurve, color = "orange") 
    # Putting labels 
    plt.xlabel('x') 
    plt.ylabel('y') 




def train4OrderPolynomialRegression(trainInputFeatures, trainOutputs):
    """
    # Train 4th-order regression model on training set
    """
    trainXFeatures = trainInputFeatures.values
    trainYOutputs = trainOutputs.values
    
    # Plot scatter plot
    plt.scatter(trainXFeatures, trainYOutputs, color = 'm', marker = 'o', s = 30)
    
    N = len(trainInputFeatures)
    X = columnAppend[ones(N), trainXFeatures, square(trainInputFeatures), power(trainInputFeatures, 3), power(trainInputFeatures, 4)]
    A = inv(X.T@X)
    D = A@X.T
    result = D@trainYOutputs
    
    plotRegressionCurve7(trainXFeatures, trainYOutputs, result)
    plt.show() 
    
    # Find average error on the training set
    A = square(result[4] * power(trainXFeatures, 4) + result[3] * power(trainXFeatures, 3) + result[2] * square(trainXFeatures) + result[1] * trainXFeatures + result[0]  - trainYOutputs)
    error = sum(A) / N
    print('\n\nTrain 4th-order regression model on training set')
    print('Average error on the training set: ', error)
    
    return result

yPredicted = []
def plotRegressionCurve7(x, y, b): 
    # Plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "b", marker = "o", s = 30) 

    xLine = linspace(x.min(), x.max(), 100)
    global yPredicted
    yPredicted = b[4] *  power(xLine, 4) + b[3] * power(xLine, 3) + b[2] * square(xLine) + b[1] * xLine + b[0] 
    regressionCurve = yPredicted
    # Plotting the regression line 
    plt.plot(xLine, regressionCurve, color = "orange") 
    # Putting labels 
    plt.xlabel('x') 
    plt.ylabel('y') 



def test4PR(testFeatures, testRealOutputs, result):
    """
    # Test 4th-order regression model on testing set
    """
    testXFeatures = testFeatures.values
    testYOutputs = testRealOutputs.values
    
    plt.scatter(testXFeatures, testRealOutputs, color = 'b', marker = 'o', s = 30)
    plotRegressionCurve8(testXFeatures, testYOutputs, result)
    plt.show() 
    
    # Find average error on the training set
    A = square(result[4] * power(testXFeatures, 4) + result[3] * power(testXFeatures, 3) + result[2] * square(testXFeatures) + result[1] * testXFeatures + result[0]  - testYOutputs)
    error = sum(A) / testXFeatures.shape[0]
    print('Average error on the testing set: ', error)
    print('4th-order polynomial regression')

yPredicted = []
def plotRegressionCurve8(x, y, b): 
    # Plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "b", marker = "o", s = 30) 

    xLine = linspace(x.min(), x.max(), 100)
    global yPredicted
    yPredicted = b[4] * power(xLine, 4) + b[3] * power(xLine, 3) + b[2] * square(xLine) + b[1] * xLine + b[0] 
    regressionCurve = yPredicted
    # Plotting the regression line 
    plt.plot(xLine, regressionCurve, color = "orange") 

    # Putting labels 
    plt.xlabel('x') 
    plt.ylabel('y')     




if __name__ == '__main__':
    trainInputFeatures, trainOutputs, testFeatures, testRealOutputs = loadData()
    
    result = trainLinearRegression(trainInputFeatures, trainOutputs)
    testLR(testFeatures, testRealOutputs, result)
    
    result = train2OrderPolynomialRegression(trainInputFeatures, trainOutputs)
    test2PR(testFeatures, testRealOutputs, result)
    
    result = train3OrderPolynomialRegression(trainInputFeatures, trainOutputs)
    test3PR(testFeatures, testRealOutputs, result)
    
    result = train4OrderPolynomialRegression(trainInputFeatures, trainOutputs)
    test4PR(testFeatures, testRealOutputs, result)
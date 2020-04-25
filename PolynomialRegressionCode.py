'''
Created on 25 apr. 2020

@author: George
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
# Load and plot data files
"""
# Load the training data hw1xtr.dat and hw1ytr.dat into the memory
trainInputFeatures = pd.read_csv('xTrain.dat',  header = None) 
trainOutputs = pd.read_csv('yTrain.dat', header = None) 

# Load the test data hw1xte.dat and hw1yte.dat into the memory
test_features = pd.read_csv('xTest.dat',  header = None) 
test_desired_outputs = pd.read_csv('yTest.dat', header = None) 

# # Plot training_data and desired_outputs
trainXFeatures = trainInputFeatures.values
trainYOutputs = trainOutputs.values
plt.scatter(trainXFeatures, trainYOutputs, color = 'g', marker = '*', s = 30)

# # Plot training_data and desired_outputs
plt.scatter(test_features.values, test_desired_outputs.values, color = 'b', marker = 'o', s = 30)
plt.title('Training and Testing Data')
plt.show()






"""
# Train linear regression model on training set
"""
trainXFeatures = trainInputFeatures.values
trainYOutputs = trainOutputs.values

N = len(trainInputFeatures)
X = np.c_[np.ones(N), trainXFeatures]
A = np.linalg.inv(X.T@X)
D = A@X.T
result = D@trainYOutputs

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

# Plot scatter plot
plt.scatter(trainXFeatures, trainYOutputs, color = 'm', marker = 'o', s = 30)
plotRegressionCurve1(trainXFeatures, trainYOutputs, result)

# Find average error on the training set
A = np.square(yPredicted - trainYOutputs)
error = np.sum(A) / N
#print('MSE:')
print('Train 1st-order regression model on training set')
print('Average error on the training set: ', error)





"""
# Test linear regression model on testing set
"""
testXFeatures = test_features.values
testYOutputs = test_desired_outputs.values

plt.scatter(testXFeatures, test_desired_outputs, color = 'b', marker = 'o', s = 30)

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

plotRegressionCurve2(testXFeatures, testYOutputs, result)

# Find average error on the training set
A = np.square(testYPredicted - testYOutputs)
error = np.sum(A) / testXFeatures.shape[0]
print('Average error on the testing set: ', error)
print('1st-order linear regression')




"""
# Train 2nd-order regression model on training set
"""
trainXFeatures = trainInputFeatures.values
trainYOutputs = trainOutputs.values

# # Plot scatter plot
# plt.scatter(trainXFeatures, trainYOutputs, color = 'm', marker = 'o', s = 30)

N = len(trainInputFeatures)
X = np.c_[np.ones(N), trainXFeatures, np.square(trainInputFeatures)]
A = np.linalg.inv(X.T@X)
D = A@X.T
result = D@trainYOutputs

yPredicted = []
def plotRegressionCurve3(x, y, b): 
    # plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "b", marker = "o", s = 30) 

    xLine = np.linspace(trainXFeatures.min(), trainXFeatures.max(), 100)
    global yPredicted
    yPredicted = b[2] * np.square(xLine) + b[1] * xLine + b[0] 
    regressionCurve = yPredicted
    # Plotting the regression line 
    plt.plot(xLine, regressionCurve, color = "orange") 
    # Putting labels 
    plt.xlabel('x') 
    plt.ylabel('y')     

# Plot scatter plot
plt.scatter(trainXFeatures, trainYOutputs, color = 'm', marker = 'o', s = 30)
plotRegressionCurve3(trainXFeatures, trainYOutputs, result)
plt.show() 

# Find average error on the training set
A = np.square(result[2] * np.square(trainXFeatures) + result[1] * trainXFeatures + result[0]  - trainYOutputs)
error = np.sum(A) / N
print('\n\nTrain 2nd-order regression model on training set')
print('Average error on the training set: ', error)





"""
# Test 2nd-order regression model on testing set
"""
testXFeatures = test_features.values
testYOutputs = test_desired_outputs.values

plt.scatter(testXFeatures, test_desired_outputs, color = 'b', marker = 'o', s = 30)

yPredicted = []
def plotRegressionCurve4(x, y, b): 
    # Plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "b", marker = "o", s = 30) 

    xLine = np.linspace(trainXFeatures.min(), trainXFeatures.max(), 100)
    global yPredicted
    yPredicted = b[2] * np.square(xLine) + b[1] * xLine + b[0] 
    regressionCurve = yPredicted
    # Plotting the regression line 
    plt.plot(xLine, regressionCurve, color = "orange") 

    # Putting labels 
    plt.xlabel('x') 
    plt.ylabel('y') 

plt.scatter(testXFeatures, test_desired_outputs, color = 'b', marker = 'o', s = 30)
plotRegressionCurve4(testXFeatures, testYOutputs, result)
plt.show() 

# Find average error on the training set
A = np.square(result[2] * np.square(testXFeatures) + result[1] * testXFeatures + result[0]  - testYOutputs)
error = np.sum(A) / testXFeatures.shape[0]
print('Average error on the testing set: ', error)
print('2nd-order polynomial regression')





"""
# Train 3rd-order regression model on training set
"""
trainXFeatures = trainInputFeatures.values
trainYOutputs = trainOutputs.values

# Plot scatter plot
plt.scatter(trainXFeatures, trainYOutputs, color = 'm', marker = 'o', s = 30)

N = len(trainInputFeatures)
X = np.c_[np.ones(N), trainXFeatures, np.square(trainInputFeatures), np.power(trainInputFeatures, 3)]
A = np.linalg.inv(X.T@X)
D = A@X.T
result = D@trainYOutputs

yPredicted = []
def plotRegressionCurve5(x, y, b): 
    # Plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "b", marker = "o", s = 30) 

    xLine = np.linspace(trainXFeatures.min(), trainXFeatures.max(), 100)
    global yPredicted
    yPredicted = b[3] * np.power(xLine, 3) + b[2] * np.square(xLine) + b[1] * xLine + b[0]     
    regressionCurve = yPredicted
    # Plotting the regression line 
    plt.plot(xLine, regressionCurve, color = "orange") 

    # Putting labels 
    plt.xlabel('x') 
    plt.ylabel('y') 

plotRegressionCurve5(trainXFeatures, trainYOutputs, result)
plt.show() 

# Find average error on the training set
A = np.square(result[3] * np.power(trainXFeatures, 3) + result[2] * np.square(trainXFeatures) + result[1] * trainXFeatures + result[0]  - trainYOutputs)
error = np.sum(A) / N
print('\n\nTrained the 3nd-order polynomial model on the training set')
print('Average error on the training set: ', error)





"""
# Test 3rd-order regression model on testing set
"""
testXFeatures = test_features.values
testYOutputs = test_desired_outputs.values

plt.scatter(testXFeatures, test_desired_outputs, color = 'b', marker = 'o', s = 30)

yPredicted = []
def plotRegressionCurve6(x, y, b): 
    # Plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "b", marker = "o", s = 30) 

    xLine = np.linspace(testXFeatures.min(), testXFeatures.max(), 100)
    global yPredicted
    yPredicted = b[3] * np.power(xLine, 3) + b[2] * np.square(xLine) + b[1] * xLine + b[0] 
    regressionCurve = yPredicted
    # Plotting the regression line 
    plt.plot(xLine, regressionCurve, color = "orange") 
    # Putting labels 
    plt.xlabel('x') 
    plt.ylabel('y') 

plotRegressionCurve6(testXFeatures, testYOutputs, result)
plt.show() 

# Find average error on the training set
A = np.square(result[3] * np.power(testXFeatures, 3) + result[2] * np.square(testXFeatures) + result[1] * testXFeatures + result[0]  - testYOutputs)
error = np.sum(A) / testXFeatures.shape[0]
print('Average error on the testing set: ', error)
print('3rd-order polynomial regression')






"""
# Train 4th-order regression model on training set
"""
trainXFeatures = trainInputFeatures.values
trainYOutputs = trainOutputs.values

# Plot scatter plot
plt.scatter(trainXFeatures, trainYOutputs, color = 'm', marker = 'o', s = 30)

N = len(trainInputFeatures)
X = np.c_[np.ones(N), trainXFeatures, np.square(trainInputFeatures), np.power(trainInputFeatures, 3), np.power(trainInputFeatures, 4)]
A = np.linalg.inv(X.T@X)
D = A@X.T
result = D@trainYOutputs

yPredicted = []
def plotRegressionCurve7(x, y, b): 
    # Plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "b", marker = "o", s = 30) 

    xLine = np.linspace(trainXFeatures.min(), trainXFeatures.max(), 100)
    global yPredicted
    yPredicted = b[4] *  np.power(xLine, 4) + b[3] * np.power(xLine, 3) + b[2] * np.square(xLine) + b[1] * xLine + b[0] 
    regressionCurve = yPredicted
    # Plotting the regression line 
    plt.plot(xLine, regressionCurve, color = "orange") 
    # Putting labels 
    plt.xlabel('x') 
    plt.ylabel('y') 

plotRegressionCurve7(trainXFeatures, trainYOutputs, result)
plt.show() 

# Find average error on the training set
A = np.square(result[4] * np.power(trainXFeatures, 4) + result[3] * np.power(trainXFeatures, 3) + result[2] * np.square(trainXFeatures) + result[1] * trainXFeatures + result[0]  - trainYOutputs)
error = np.sum(A) / N
print('\n\nTrain 4th-order regression model on training set')
print('Average error on the training set: ', error)





"""
# Test 4th-order regression model on testing set
"""
testXFeatures = test_features.values
testYOutputs = test_desired_outputs.values

plt.scatter(testXFeatures, test_desired_outputs, color = 'b', marker = 'o', s = 30)

yPredicted = []
def plotRegressionCurve8(x, y, b): 
    # Plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "b", marker = "o", s = 30) 

    xLine = np.linspace(testXFeatures.min(), testXFeatures.max(), 100)
    global yPredicted
    yPredicted = b[4] * np.power(xLine, 4) + b[3] * np.power(xLine, 3) + b[2] * np.square(xLine) + b[1] * xLine + b[0] 
    regressionCurve = yPredicted
    # Plotting the regression line 
    plt.plot(xLine, regressionCurve, color = "orange") 

    # Putting labels 
    plt.xlabel('x') 
    plt.ylabel('y')     

plotRegressionCurve8(testXFeatures, testYOutputs, result)
plt.show() 

# Find average error on the training set
A = np.square(result[4] * np.power(testXFeatures, 4) + result[3] * np.power(testXFeatures, 3) + result[2] * np.square(testXFeatures) + result[1] * testXFeatures + result[0]  - testYOutputs)
error = np.sum(A) / testXFeatures.shape[0]
print('Average error on the testing set: ', error)
print('4th-order polynomial regression')







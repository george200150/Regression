'''
Created on 25 apr. 2020

@author: George
'''
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#import  as np
import random

from regresieGBP_Freedom import inputRead


trainInputs, trainOutputs, feature1train, feature2train, testInputs, testOutputs, feature1test, feature2test = inputRead() # datele sunt deja normalizate ( 
Y=trainOutputs+testOutputs
X=trainInputs+testInputs

X = np.asarray(X).reshape([-1,2])
Y = np.asarray(Y).reshape([-1,1])

trainInput, testInput, trainOutput, testOutput = train_test_split(X,Y,test_size=0.2) # splits the data 80/20

print("X Shape: ",X.shape)
print("Y Shape: ",Y.shape)
print("train Input Shape: ",trainInput.shape)
print("test Input Shape: ",testInput.shape)
print("train Output Shape: ",trainOutput.shape)
print("test Output Shape: ",testOutput.shape)

# standardizing data
scaler = preprocessing.StandardScaler().fit(trainInput)
trainInput = scaler.transform(trainInput)
testInput=scaler.transform(testInput)



testInput=np.array(testInput)
testOutput=np.array(testOutput)





# SkLearn SGD classifier (batch sgd)
numtraining = len(X)

def createBatches(sizeB):
    # Provide chunks one by one
    batchSeparator = 0
    while batchSeparator < numtraining:
        chunkrows = range(batchSeparator, batchSeparator+sizeB)
        X_chunk = X[chunkrows] 
        y_chunk = Y[chunkrows]
        yield X_chunk, y_chunk
        batchSeparator += sizeB

batches = createBatches(sizeB=1)


Model = SGDRegressor(learning_rate = 'constant', alpha = 0, eta0 = 0.01, shuffle=True)

chunks = list(batches)
for _ in range(4):
    random.shuffle(chunks)
    for X_chunk, y_chunk in chunks:
        Model.partial_fit(X_chunk, y_chunk) # partially fit the model using the current batch

#y_predicted = Model.predict(X)





# SkLearn SGD classifier (normal sgd)
"""n_iter=1000
Model = SGDRegressor(max_iter=n_iter)
Model.fit(trainInput, trainOutput)"""
yPredicted_SK_SGD=Model.predict(testInput)
plt.scatter(testOutput,yPredicted_SK_SGD)
plt.grid()
plt.xlabel('Actual y')
plt.ylabel('Predicted y')
plt.title('Scatter plot from actual y and predicted y') # must form a diagonal line (best case)
plt.show()

print('Mean Squared Error :',mean_squared_error(testOutput, yPredicted_SK_SGD))


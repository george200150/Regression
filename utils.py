'''
Created on 24 apr. 2020

@author: George
'''


import csv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



def loadData(fileName, inputVariabName, outputVariabName):
    size = len(inputVariabName)
    data = []
    dataNames = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1
    selectedVariables = [dataNames.index(inputVariabName[i]) for i in range(size)]
    
    inputs = [[float(data[i][selectedVariables[j]]) for j in range(size)] for i in range(len(data))]
    selectedOutput = dataNames.index(outputVariabName)
    outputs = [float(data[i][selectedOutput]) for i in range(len(data))]
    
    return inputs, outputs

def loadDataSingleFeature(fileName, inputVariabName, outputVariabName):
    data = []
    dataNames = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1
    selectedVariable = dataNames.index(inputVariabName)
    inputs = [float(data[i][selectedVariable]) for i in range(len(data))]
    selectedOutput = dataNames.index(outputVariabName)
    outputs = [float(data[i][selectedOutput]) for i in range(len(data))]
    
    return inputs, outputs


def plotData2D(inputs, outputs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for i,o in zip(inputs,outputs):
        xs = i
        ys = o
        ax.scatter(xs, ys, marker='o')
    
    ax.set_xlabel('GDP capita')
    ax.set_ylabel('happiness')
    
    plt.title('GDP capita vs. happiness')
    plt.show()

def plotData3D(inputs, outputs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for i,o in zip(inputs,outputs):
        xs = i[0]
        ys = i[1]
        zs = o
        ax.scatter(xs, ys, zs, marker='o')
    
    ax.set_xlabel('GDP capita')
    ax.set_ylabel('Freedom')
    ax.set_zlabel('happiness')
    
    plt.title('GDP capita vs. Freedom vs. happiness')
    plt.show()





def plotDataHistogram(x, variableName):
    n, bins, patches = plt.hist(x, 10)
    plt.title('Histogram of ' + variableName)
    plt.show()



def plot2DModel(feature1train, trainOutputs, xref1, yref):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for i1,o in zip(feature1train,trainOutputs):
        xs = i1
        ys = o
        ax.scatter(xs, ys, marker='o')
    
    ax.set_xlabel('GDP capita')
    ax.set_ylabel('happiness')
    
    
    ax.plot(xref1, yref, label='parametric curve')
    plt.title('GDP capita vs. happiness')
    plt.show()


def plotModel(feature1train, feature2train, trainOutputs, xref1, xref2, yref):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for i1,i2,o in zip(feature1train,feature2train,trainOutputs):
        xs = i1
        ys = i2
        zs = o
        ax.scatter(xs, ys, zs, marker='o')
    
    ax.set_xlabel('GDP capita')
    ax.set_ylabel('Freedom')
    ax.set_zlabel('happiness')
    
    
    ax.plot(xref1, xref2, yref, label='parametric curve')
    plt.title('GDP capita vs. Freedom vs. happiness')
    plt.show()
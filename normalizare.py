'''
Created on 24 apr. 2020

@author: George
'''

from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
#from statistics import mean, stdev
from MyStatistics import mean, stdev
import csv


#tema (75 puncte):
#implementare proprie pentru normalizarea statistica a datelor 
#din problema de regresie bivariata ($GDP$, $Freedom$ si $Happiness$)


# load all the data from a csv file
def loadDataMoreInputs(fileName):
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
    return dataNames, data

# extract a particular feature (column)
def extractFeature(allData, names, featureName):
    pos = names.index(featureName)
    return [float(data[pos]) for data in allData]

# plot a histogram for some data x
def plotDataHistogram(x, variableName):
    n, bins, patches = plt.hist(x, 20)
    plt.title('Histogram of ' + variableName)
    plt.show()


class stdNorm:
    def __init__(self):
        self.mean = None
        self.stdev = None
    
    def statisticalNormalisation(self, features):
        if self.mean is None or self.stdev is None:
            self.mean = mean(features)
            self.stdev = stdev(features)
        normalisedFeatures = [(feat - self.mean) / self.stdev for feat in features]
        return normalisedFeatures
        
        
# statistical normalisation (centered around meand and standardisation)
'''def statisticalNormalisation(features):
    # meanValue = sum(features) / len(features)
    meanValue = mean(features)
    # stdDevValue = (1 / len(features) * sum([ (feat - meanValue) ** 2 for feat in features])) ** 0.5 
    stdDevValue = stdev(features)
    normalisedFeatures = [(feat - meanValue) / stdDevValue for feat in features]
    return normalisedFeatures'''



if __name__ == '__main__':
    # load some data 
    data = load_wine()
    features = data['data']
    target = data['target']
    featureNames = data['feature_names']
    # print(featureNames)
    
    
    # consider only two features and identify their definition domain
    feature1 = extractFeature(features, featureNames, 'alcohol')
    feature2 = extractFeature(features, featureNames, 'malic_acid')
    
    minFeature1 = min(feature1)
    maxFeature1 = max(feature1)
    
    minFeature2 = min(feature2)
    maxFeature2 = max(feature2)
    
    print('def domain for feature1: [', minFeature1, ', ', maxFeature1, ']')
    print('def domain for feature2: [', minFeature2, ', ', maxFeature2, ']')
    feature1normalised = statisticalNormalisation(feature1)
    feature2normalised = statisticalNormalisation(feature2)
    
    plt.plot(feature1, feature2, 'ro', label = 'raw data')
    #plt.plot(feature1scaled01, feature2scaled01, 'b^', label = '0-1 normalised data')
    #plt.plot(feature1centered, feature2centered, 'g*', label = 'centered (around 0) data')
    plt.plot(feature1normalised, feature2normalised, 'y+', label = 'standardised data (mean = 0, stdDev = 1)')
    plt.legend()
    plt.xlabel('alcohol')
    plt.ylabel('malic acid')
    plt.show()


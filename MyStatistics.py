'''
Created on 24 apr. 2020

@author: George
'''
#from math import sqrt


def mean(data):
    n = len(data)
    mean = sum(data)/n
    return mean
    

def stdev(data):
    n = len(data)
    meann = mean(data)
    variance = sum([((datum - meann) ** 2) for datum in data]) / n
    stddev = variance ** 0.5
    return stddev
    '''mean = float(sum(lst)) / len(lst)
    return sqrt(float(lambda x, y: x + y, map(lambda x: (x - mean) ** 2, lst)) / len(lst))'''


'''def variance(data):
    average = sum(data) / len(data)
    varience = sum((average - value) ** 2 for value in data) / len(data)
    return varience'''



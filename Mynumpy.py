'''
Created on 27 apr. 2020

@author: George
'''
from math import floor


def T(m):
    return map(list,zip(*m))

def minor(m,i,j):
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]

def det(m):
    #base case for 2x2 matrix
    if len(m) == 2:
        return m[0][0]*m[1][1]-m[0][1]*m[1][0]

    determinant = 0
    for c in range(len(m)):
        determinant += ((-1)**c)*m[0][c]*det(minor(m,0,c))
    return determinant

def inv(m):
    determinant = det(m)
    #special case for 2x2 matrix:
    if len(m) == 2:
        return [[m[1][1]/determinant, -1*m[0][1]/determinant],
                [-1*m[1][0]/determinant, m[0][0]/determinant]]

    #find matrix of cofactors
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = minor(m,r,c)
            cofactorRow.append(((-1)**(r+c)) * det(minor))
        cofactors.append(cofactorRow)
    cofactors = T(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c]/determinant
    return cofactors

def empty(size):
    return valueNotWrapped([], size, 0)

def ones(size):
    if not isinstance(size, list):
        return valueNotWrapped(1, [size], 0) 
    return valueNotWrapped(1, size, 0)

def zeros(size):
    return valueNotWrapped(0, size, 0)

def valueNotWrapped(value, size, pos):
    l = len(size)
    if pos == l-1:
        return [value for _ in range(size[pos])]
    
    finalReturn = [[] for _ in range(size[pos])]
    
    for x in range(size[pos]):
        finalReturn[x] = valueNotWrapped(value, size, pos+1)
    return finalReturn


def shape(array): # renamed deduceShape to simply shape
    sh = deduceShapeNotWrapped(array, [])
    return sh

def deduceShapeNotWrapped(array, shape):
    try:
        shape.append(len(array))
        deduceShapeNotWrapped(array[0],shape)
    except TypeError:
        pass
    return shape

def liniarize(array, shape):
    result = []
    liniarizeNotWrapped(array, shape, result)
    return result

def liniarizeNotWrapped(array, shape, result):
    if not isinstance(array, list):
        result.append(array)
        return array
    if len(array) == 0:
        return
    liniarizeNotWrapped(array[0],shape[1:],result)
    liniarizeNotWrapped(array[1:],shape,result)
    #print(array)


def reshape(array, shape):
    #1. liniarize
    liniarized = liniarize(array,shape)
    #print("liniarized = ",liniarized)
    #2. reshape
    return reshapeNotWrapped(liniarized,shape)


def getProductOfShape(shape):
    product = 1
    for integer in shape:
        product *= integer
    return product

def reshapeNotWrapped(array, shape):
    l = len(array)
    #TODO: implement shape [-1,x,y,z] or [x,-1] or whatever
    minus = False
    index = 0
    for value in shape:
        if value == -1 and minus == False:
            product = -getProductOfShape(shape)
            dimension = l // product
            shape[index] = dimension
        elif minus == True:
            raise TypeError("cannot have two lengths")
        index += 1
    
    product = getProductOfShape(shape)
    if product >= 0 and product != l:
        raise TypeError("length not match")
    
    return variableValueNotWrapped(array, shape) # TODO: define a value not wrapped with variable replacement value


def variableValueNotWrapped(array, size):
    if len(size) == 0:
        return array[0]
    
    #finalReturn = [[] for _ in range(size[0])]
    finalReturn = []
    l = len(array)
    batch = l / size[0]
    if int(floor(batch)) != batch:
        raise TypeError("cannot split exactly using this shape")
    
    batch = l // size[0]
    for x in range(size[0]):
        start = batch * x
        stop = batch * (x+1)
        finalReturn.append(variableValueNotWrapped(array[start:stop], size[1:]))
    
    return finalReturn



def sum(array):
    sh = shape(array)
    array = liniarize(array, sh)
    suma = 0
    for x in array:
        suma += x 
    return suma

def square(array):
    return power(array,2)


def power(array, exponent):
    sh = shape(array)
    array = liniarize(array, sh)
    final = []
    for elem in array:
        final.append(elem**exponent)
    final = reshape(final,sh)
    return final


def linspace(start, end, points):
    step = (end - start) / (points - 1)
    print(step)
    lista = []
    current = start
    while current < end:
        lista.append(current)
        current += step
    return lista


def c_(array1, array2): # AxisConcatenator
    '''
    np: Translates slice objects to concatenation along the second axis.
                                            -that's what they said
    '''
    sh1 = shape(array1)
    sh2 = shape(array2)
    if sh1 != sh2:
        raise TypeError("shape mismatch")
    if len(sh1) != 2:
        raise TypeError("shape incompatibility")
    array1 = liniarize(array1, sh1)
    array2 = liniarize(array2, sh2)
    joined = []
    for a1,a2 in zip(array1,array2):
        joined.append(a1)
        joined.append(a2)
    l = sh1[0]
    w = sh1[1] + sh2[1]
    joined = reshape(joined,[l,w])
    return joined
    




import numpy as np
if __name__ == '__main__':
    '''l = ones([2,3,1])
    shape = shape(l)
    print(shape)'''
    #l = [[[1], [2], [3]], [[4], [5], [6]]]
    #print(l)
    #shape2 = [3,2,1]
    '''shape2 = [-1,2]
    l = reshape(l, shape2)
    print(l)'''
    '''s = sum(l)
    print(s)'''
    '''l = square(l)
    print(l)'''
    
    '''xLine = linspace(0, 20, 50)
    print(xLine)
    print(len(xLine))
    
    xLine = np.linspace(0, 20, 50)
    print(xLine)
    print(len(xLine))'''
    
    o1 = ones(3)
    print(o1)
    
    o1 = ones([10,1])
    print(o1)
    z2 = zeros([10,1])
    print(z2)
    res = c_(o1,z2)
    print(res)
    
    
    '''l = [0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9]
    l = np.asarray(l)
    
    l = l.reshape([10,-1])
    print(l)
    
    print()
    
    l = l.reshape([5,-1])
    print(l)
    
    print()
    print()
    print()
    
    l = l.reshape([-1,10])
    print(l)
    
    print()
    
    l = l.reshape([-1,5])
    print(l)'''
    
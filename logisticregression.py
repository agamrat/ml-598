import math
import numpy
import random

"""
Create the initial weight vector w0.
"""
def getInitVector(features):
    result = []
    for i in xrange(features):
        result.append(random.randint(-10,10))
    return result

"""
Calculate the next-best weight vector for logistic regression.
"""
def __nextStep(vector, features, outputs):
    result = 0
    for i in xrange(len(outputs)):
        #sigma(-wTx)
        try:
            sigmoid = 1/(1+math.exp(-(numpy.dot(vector,features[i]))))
        except OverflowError:
            sigmoid=1
        #xi*(yi-sigma)
        result = result +numpy.multiply(features[i],(outputs[i] - sigmoid))
    return result 

"""
Train logistic regression on the data with given stepsize and epsilon.
Iteration limit used to avoid infinite loop in case of bad hyper-parameter choices.
"""
def trainLogisticReg(epsilon, stepsize, iterLim, x,y):
    wNext = getInitVector(len(x[0]))

    for i in xrange(iterLim):
        #wk < - prev wk+1
        wCur = wNext
        #wk+1 < - wk + (stepsize * nextstep)
        wNext = wCur + (stepsize * __nextStep(wCur, x,y)) 
 	#have the values changed much between wk+1 and wk?
        isDone= all( abs(q)  < epsilon for q in numpy.subtract(wNext,wCur))
        if isDone: 
            break

    return wNext

"""
Test a set of weights on data.
"""
def testLogisticReg(weightVector, x, y):

    #check valid weight vector length
    if len(weightVector) != len(x[0]) and len(weightVector) != len(x[0]) + 1:
        print "Must have equal number of features as weights (or +1 for intercept)"
        sys.exit(0)

    intercept=0

    #is there a w0 / intercept in the weight vector?
    if len(weightVector) == len(x[0]) + 1:
        intercept = weightVector[0]
        weightVector = weightVector[1:]

    error = 0
    for i in xrange(len(x)):
        # wTx = w0 + w1x1 + w2x2 etc.
        wTx = intercept + numpy.dot(weightVector, x[i])
        # sigmoid = 1 / (1 + e^-t)
        innersigmoid = math.exp(numpy.multiply(-1, wTx))
        sigmoid = 1/(1+innersigmoid)
        #switch sigmoid where yi = 0
        if y[i] == 0:
             sigmoid = 1- sigmoid
        temperror = math.log(sigmoid)
        error = error + temperror
    return -error

def getConfusionMatrix(weightVector,x,y):

    #check valid weight vector length
    if len(weightVector) != len(x[0]) and len(weightVector) != len(x[0]) + 1:
        print "Must have equal number of features as weights (or +1 for intercept)"
        sys.exit(0)

    intercept=0

    #is there a w0 / intercept in the weight vector?
    if len(weightVector) == len(x[0]) + 1:
        intercept = weightVector[0]
        weightVector = weightVector[1:]

    confusion = []
    confusion.append([0,0])
    confusion.append([0,0])

    for i in xrange(len(x)):
        # wTx = w0 + w1x1 + w2x2 etc.
        wTx = intercept + numpy.dot(weightVector, x[i])
        # sigmoid = 1 / (1 + e^-t)
        innersigmoid = math.exp(numpy.multiply(-1, wTx))
        sigmoid = 1/(1+innersigmoid)
        #if over half, assign to class 1
        prediction=0
        if sigmoid > 0.5:
            prediction=1
        #if prediction positive, 1-class is second index
        if prediction == 1:
            confusion[0][int(1-y[i])] = confusion[0][int(1-y[i])] +1
        else:
            confusion[1][int(y[i])] = confusion[1][int(y[i])]+1
    print ""
    return confusion

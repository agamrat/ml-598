import numpy
import math

"""
Train a Naive Bayes Classifier
"""

def trainNaiveBayesMN(x, y):
    points = len(y)
    py1 = sum(y)/float(points)

    theta = []
    for i in xrange(len(x[0])):
	features = numpy.array(x)[:, i]
	num_bins = int(max(features))
	theta.append([])
	for j in xrange(0, num_bins+1):
            theta[i].append([0,0])
    for i in xrange(0, len(x)):
	for j in xrange(0, len(x[i])):
	    class_index = int(y[i])
	    theta[j][int(x[i][j])][class_index] += 1
    for i in xrange(0, len(theta)):
	for j in xrange(0, len(theta[i])):
	    theta[i][j][0] /= float((1-py1) * points)
	    theta[i][j][1] /= float(py1 * points)
    return (py1, theta)
		
def trainNaiveBayes(x,y):
    if len(x) != len(y):
        print "Inputs and outputs must be the same length"
        return

    points = len(y)

    #theta 1 = Pr (y=1)
    #theta j,1 = Pr (x j =1 | y=1)
    #theta j,0 = Pr (x j =1 | y=0).

    #P(y=1)  = (1/n) sum( i=1:n y i)
    py1 = sum(y) /float(points)
 
    #P(x=1|y=0) and P(x=1 | y=1) 
    theta = []
    for i in x[0]:
        theta.append([0,0])

    #sum examples where xj = 1 compared with y value
    for i in xrange(len(x)):
        for j in xrange(len(x[i])):
            if x[i][j] == 1:
                index = int(y[i])
                theta[j][index] = theta[j][index] +1

    for j in xrange(len(theta)):
        theta[j][1] = theta[j][1]/float(py1*points)
        theta[j][0] = theta[j][0]/(float(points*(1.0 - py1)))
    return (py1, theta)

"""
Get confusion matrix of Naive Bayes classifier
"""

def getConfusionMatrixMN(py, theta, x, y):
    confusion = []
    confusion.append([0,0])
    confusion.append([0,0])
    for i in xrange(len(x)):
       	prob = math.log(float(py)/(1-py))
        for j in xrange(0, len(x[i])):
	    if x[i][j] > len(theta[j]) - 1:
		continue
	    t_est = theta[j][int(x[i][j])][1]
	    b_est = theta[j][int(x[i][j])][0]
	    
	    if b_est == 0:
		b_est = 0.5
	    if t_est == 0:
		t_est = 0.5
	    #print theta[j][int(x[i][j])][1], theta[j][int(x[i][j])][0]
            prob += math.log(float(t_est)/b_est)
	if prob >= 0.0:
            confusion[0][int(1-y[i])] = confusion[0][int(1-y[i])] +1
        else:
            confusion[1][int(1-y[i])] = confusion[1][int(1-y[i])]+1


    return confusion

def getConfusionMatrix(py, theta, x,y):
    #Pr(y|x) - > Pr(y)Pr(x|y)

    confusion = []
    confusion.append([0,0])
    confusion.append([0,0])

    for i in xrange(len(x)):
        prob = 1
        for j in x[i]:
            if j ==1:
                prob = prob*theta[int(j)][1]
        if prob >= 0.5:
            confusion[0][int(1-y[i])] = confusion[0][int(1-y[i])] +1
        else:
            confusion[1][int(1-y[i])] = confusion[1][int(1-y[i])]+1
           

    return confusion

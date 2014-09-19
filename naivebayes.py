import numpy

"""
Train a Naive Bayes Classifier
"""
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
        theta[j][0] = theta[j][0]/(float(points*(1 - py1)))

    return (py1, theta)

"""
Get confusion matrix of Naive Bayes classifier
"""
def getConfusionMatrix(py, theta, x,y):
    #Pr(y|x) - > Pr(y)Pr(x|y)

    confusion = []
    confusion.append([0,0])
    confusion.append([0,0])

    for i in xrange(len(x)):
        prob = 1
        for j in x[i]:
            prob = prob*theta[int(j)][1]
        if prob > 0.5:
            confusion[0][int(1-y[i])] = confusion[0][int(1-y[i])] +1
        else:
            confusion[1][int(1-y[i])] = confusion[1][int(1-y[i])]+1
           

    return confusion

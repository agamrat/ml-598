import collections
import numpy
import math

"""
Train an LDA and return (p(y), means, covariance).
"""
def trainLDA(x,y):
    count = collections.Counter(y)
    if len(list(count)) > 2:
        sys.exit("Please use an algorithm that works for more than 2 classes.")

    #calculating p(y=1) and p(y=0)
    pclasses = {}
    for i in list(count):
         pclasses[i] = count[i]

    for j in pclasses:
         pclasses[j]= float(pclasses[j])/sum(count.values())

    #calculating the means of each class
    data = zip(y,x)
    means = [[],[]]
    means[0] = [0]*len(x[0])
    means[1] = [0]*len(x[0])

    for i in list(count):
         i = int(i)
         #initialize result to the number of features
         result = [0]*len(x[0])
         for row in data:
              if row[0] == i:
                  result = numpy.add( result , row[1:][0])
         means[i] = [r/float(count[i]) for r in  result]

    #calculating the cov matrix for y = 0
    cov =[0]*len(x[0])

    #would loop for each class
    #except assume cov matrices the same (so just set k = 0 since k=1 is same as k =0)
    k = 0
    result= []
    #for each data row
    for m in xrange(len(x[0])):
         result.append([0]*len(x[0]))
         #result should be a 4 x 4 matrix

    for i in data:
        #l(yi = 0)
        indicator = 1 if i[0] == 0 else 0 
        #l(yi=0) * (x(i) - mean(k))
        diffFromMean= indicator * (numpy.subtract(i[1] , means[k]))
        #multiply (xi -mean[k]) by its transpose
        tempresult = numpy.multiply(diffFromMean,zip(*[diffFromMean]))
        result = numpy.add(result, tempresult)
        #divide each sum by the number of entries in the opposite class
    cov = [r/float(count[1-k]) for r in  result]

    return (pclasses, means, cov)

"""
Test LDA predictions - returns the misclassification rate
"""
def testLDA(pclasses, means, cov, x, y):
    #classify by using log-odds ratio

    #first log term
    firstTerm = math.log(pclasses[1]/pclasses[0],2)

    #second term
    firstMatrix=numpy.multiply(-0.5,numpy.add(means[0], means[1]))
    try:
        covInvert = numpy.linalg.inv(cov)
    except: #singular matrix
        covInvert = numpy.linalg.pinv(cov)

    subtractedMeans = zip(*[numpy.subtract(means[0],means[1])])
    secondTerm = numpy.dot(numpy.dot(firstMatrix, covInvert), subtractedMeans)

    #third term
    thirdTerm = numpy.dot(covInvert, subtractedMeans)

    misclassifications = 0
    for i in xrange(len(x)):
        endprediction = numpy.dot(x[i],thirdTerm) + firstTerm + secondTerm
        #check if prediction correct
        if (endprediction > 0 and y[i] == 1) or (endprediction <= 0 and y[i] == 0):
            continue
        else:
            misclassifications = misclassifications +1
            
    return misclassifications / float(len(y))


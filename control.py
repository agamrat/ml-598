import csv_parser
import logisticregression
import lda
import sys
import math

def runLDA(x,y, testX, testY, testfilename):
    (prob, mean, cov) = lda.trainLDA(x,y)
    resultfile=open(testfilename + "_result",'w')
    resultfile.write("'Misclassification rate'\n")
    resultfile.write(str(lda.testLDA(prob, mean, cov,testX, testY)))
    resultfile.close()
    return

if len(sys.argv) < 2 or sys.argv[1] == "--help":
    print "Usage: control.py controlfile"
    print "Usage: control.py -lda trainingfile testfile"
    print " Control files look like: "
    print "    First line: trainingfilename, testfilename"
    print "    Any number of subsequent lines: epsilon,stepsize,iteration limit, restarts"
    print "    Training and test files are CSV's with the output variable last"
    print "    Results are stored in testfilename_resultX where X is the line number from the control CSV"
    sys.exit(0)

#lda control here
if sys.argv[1] == "-lda":
   if len(sys.argv) != 4:
        print "Usage: control.py -lda trainingfile testfile"
        sys.exit(0)
   (x,y)= csv_parser.parse_data(sys.argv[2])
   (testX, testY) = csv_parser.parse_data(sys.argv[3])
   runLDA(x,y, testX, testY, sys.argv[3])
   sys.exit(0)   

#default is logistic regression
(controls, training, test) = csv_parser.parse_control(sys.argv[1])

(x,y)= csv_parser.parse_data(training)
(testX, testY) = csv_parser.parse_data(test)

for i in xrange(len(controls)):
    params = controls[i]
    resultfile = open(test + "_result" + str(i), 'w') 
    resultfile.write('"WeightVector","Error","ConfusionMatrix"\n')
    for j in xrange(int(math.floor(params[3]))):
        result= logisticregression.trainLogisticReg(params[0],params[1], int(params[2]), x, y)
        error = logisticregression.testLogisticReg(result, testX, testY)
        confusion = logisticregression.getConfusionMatrix(result, testX,testY)
        resultfile.write(str(result) + "," + str(error) + ","+ str(confusion)+"\n")    
    resultfile.close()
        
 


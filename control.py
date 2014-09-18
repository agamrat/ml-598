import csv_parser
import logisticregression

if len(sys.argv) != 2 or sys.argv[1] == "--help":
    print "Usage: logisticregression.py filename"
    print "Files are CSV's with the output variable last"
    sys.exit(0)

(x,y)= csv_parser.parse_data(sys.argv[1])

epsilon = 0.05
stepsize = 0.25
limIter=10000

result= trainLogisticReg(epsilon, stepsize, limIter, getInitVector(len(x[0])), x, y)

print result

print testLogisticReg(result,x,y)

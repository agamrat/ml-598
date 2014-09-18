import sys

def parse_control(filename):
   try:
      data = open(filename)
   except:
      sys.exit("Can't open file: " + filename)

   controls= []
   trainingfile = ""
   testfile = ""
   first = True

   for line in data:
        if first:
            first = False
            columns = line.strip().split(',')
	    if len(columns) != 2:
                sys.exit("Must include both training and test files on first line of control file") 
            trainingfile = columns[0]
            testfile = columns[1]
            continue

        columns = [float(q) for q in line.strip().split(',')]
        if len(columns) != 4:
            sys.exit("Control file line format: epsilon, stepsize, limIteratios, restarts") 
        controls.append(columns)

   return (controls, trainingfile, testfile)
         

def parse_data(filename):
    try:
        data = open(filename)
    except:
        sys.exit("Can't open file: " + filename)
    x = []
    y = []
 
    featureLength = 0
 
    for line in data:
        columns = [float(q) for q in line.strip().split(',')]
        if featureLength == 0:
           featureLength = len(columns)
        if len(columns) != featureLength:
           sys.exit("Error: column length should be " + 
            str(featureLength) + " but was " + str(len(columns)))  
        y.append(columns[len(columns)-1])
        x.append(columns[0:len(columns)-1])

    return (x,y)

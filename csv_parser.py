import sys
import random

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
         
def parse_data_with_subjects(data_file, subject_file):
    try:
        data = open(data_file)
        word_file = open(subject_file)
    except:
        sys.exit("Can't open file: " + filename)
    x, y, words = [], [], []
 
    for line in data:
        numerical = [float(q) for q in line.strip().split(',')]

        y.append(numerical[len(numerical)-1])
        x.append(numerical[0:len(numerical)-1])
    for line in word_file:
        subjects = [w.lower() for w in line.strip().split(",")]
        words.append(subjects)
	
	limit = 100000
    while (len(x) > 100000):
    	i = random.randint(0, len(x)-1)
    	if (y[i] == 0):
    		del x[i]
    		del y[i]
    		del words[i]
    		
    data.close()
    word_file.close()
    return (x,y, words)

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

import sys

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

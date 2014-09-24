ml-598
======

This repository contains the code to implement the three basic classification algorithms seen in class: Linear Discriminant Analysis, Logistic Regression and Naive Bayes. In addition, we have included the file to scrape the required information from the bulk data download at [GovTrack.us](http://www.GovTrack.us). 

##Scraping the data
scrape_files.py is run with no arguments. You will need to change line 90 to your home directory. Also you should have run the bulk data download found [here](https://www.govtrack.us/developers/data) which will create the proper directory structure.

##Running K-Fold Validation
To run K-Fold one must run the kfolds.py without any arguments. To change the number of groups (k) change the parameter found at line 276. To change the file name used to read in the dataset change the parameter at line 274. Finally the (hyper) parameters used for logistic regression can be changed at line 76 (epsilon and step-size).


##Basic Algorithms

Each algorithm has a trainAlgorithm function which takes a list of features and a matching list of classes. There is also a getConfusionMatrix function for each which takes the test data in the same fashion as well as the parameters returned by the train function.

##Contol File for basic algorithms

control.py runs one way for logistic regression and another for LDA and naive Bayes. However, for all of these the training and test files are CSV's with the classes last. Results will be the confusion matrix outputted to a file called testfilenamehere_resultsX where X is the number of the run.

For LDA and naive Bayes, run like this:
control.py (-lda|-bayes) trainingfile testfile

For logistic regression, run like this:
control.py controlfile

A control file contains
  First line: trainingfilename, testfilename"
  Any number of subsequent lines: epsilon,stepsize,iteration limit, restarts"
   

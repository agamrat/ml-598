import sklearn.linear_model as linmod
import lda as lda_custom
import sklearn.lda as lda
import numpy
import sklearn.naive_bayes as nb
import logisticregression as linmod_custom
import naivebayes as nb_custom

def read_data_from_csv(file_name):
	X = numpy.genfromtxt(file_name, delimiter=',')
        Y = X[:,-1]
        X = numpy.delete(X, -1, 1)
	return X, Y

def split_data(X, Y):
	X_train, Y_train = [], []
	X_test, Y_test = [], []

	num_train_false = 0
	num_train_true = 0
	num_test_false = 0
	num_test_true = 0
	
	for i in xrange(0, X.shape[0]):
		if num_train_false < 9000 and Y[i] == 0:
			num_train_false += 1
			X_train.append(X[i,:].tolist())
			Y_train.append([Y[i]])
		elif num_test_false < 2000 and Y[i] == 0:
			num_test_false += 1
			X_test.append(X[i,:].tolist())
			Y_test.append([Y[i]])
		elif num_train_true < 9000 and Y[i] == 1:
			num_train_true += 1
			X_train.append(X[i,:].tolist())
                        Y_train.append([Y[i]])
		elif num_test_true < 2000 and Y[i] == 1:
			num_test_true += 1
			X_test.append(X[i,:].tolist())
                        Y_test.append([Y[i]])
		if num_train_false == 9000 and num_test_false == 2000 and num_train_true == 9000 and num_train_true == 2000:
			break
	return numpy.array(X_train), numpy.array(Y_train), numpy.array(X_test), numpy.array(Y_test)
	
#lr_model = linmod.LogisticRegression()
#lda_model = lda.LDA()
nb_model = nb.MultinomialNB()
X, Y = read_data_from_csv("final_data.csv")
X_train, Y_train, X_test, Y_test = split_data(X,Y)

#lr_model.fit(X_train, Y_train)
#lda_model.fit(X_train, Y_train)
nb_model.fit(X_train, Y_train.ravel())

#(scales, classes, mean, covar) = lda_custom.trainLDA(X_train.tolist(), Y_train.ravel().tolist())
#print lda_custom.getConfusionMatrix(scales, classes, mean, covar, X_test, Y_test)
#X_train_with_intercepts = numpy.ones((X_train.shape[0], X_train.shape[1]+1))
#X_train_with_intercepts[:,1:] = X_train
#weights, scales = linmod_custom.trainLogisticReg(0.01, 0.00001, 1000, X_train_with_intercepts.tolist(), Y_train.ravel().tolist())
#print linmod_custom.getConfusionMatrix(weights, scales, X_test, Y_test)

(py, theta) = nb_custom.trainNaiveBayesMN(X_train.tolist(), Y_train.ravel().tolist())
print nb_custom.getConfusionMatrixMN(py, theta, X_test.tolist(), Y_test.ravel().tolist())

tp, tn, fp, fn = 0, 0, 0, 0
for i in xrange(0, X_test.shape[0]):
	val = nb_model.predict(X_test[i,:])
	if (val == 1 and Y_test[i] == 1):
		tp += 1
	elif (val == 1 and Y_test[i] == 0):
		fp += 1
	elif (val == 0 and Y_test[i] == 0):
		tn += 1
	elif (val == 0 and Y_test[i] == 1):
		fn += 1
print "TP: %d, FP: %d, TN: %d, FN: %d" % (tp, fp, tn, fn)

import random
import math
import copy
import lda
import logisticregression
import naivebayes
import csv_parser
import sklearn.linear_model as linmod

#k is number of groups needed, data is a list of data rows
def splitdata(k, x, y, words):
	if(len(x) != len(y)):
		print "input lengths not the same... "
		return
	#init k groups
	# x and y list
	groups = []
	for i in xrange(k):
		groups.append([])
		groups[i].append([])
		groups[i].append([])
		groups[i].append([])
		
	#copy data and distribute between groups!
	data_copy_x = copy.deepcopy(x)
	data_copy_y = copy.deepcopy(y)
	data_copy_words = copy.deepcopy(words)
	for i in xrange(len(x)):
		chosen_index = random.randint(0, len(data_copy_x) - 1)
		groups[i%k][0].append(data_copy_x[chosen_index])
		groups[i%k][1].append(data_copy_y[chosen_index])
		groups[i%k][2].append(data_copy_words[chosen_index])
		del data_copy_x[chosen_index]
		del data_copy_y[chosen_index]
		del data_copy_words[chosen_index]
	return groups
		
#k is number of groups needed, data is a list of data rows
import operator
def kfolds_all_algos(k, x, y, train_subjects, isotest_x, isotest_y, isotest_words):
	def word_count(examples, keep):
		words = {}
		for ex in examples:
			for subject in ex:
				if subject in words:
					words[subject] += 1
				else:
					words[subject] = 1
		sorted_x = sorted(words.iteritems(), key=operator.itemgetter(1))
		sorted_x.reverse()
		if keep > len(sorted_x):
		
			return words.keys()
		limit = sorted_x[keep][1]
		print limit
		ret_words = []
		for key in words:
			if words[key] >= limit:
				ret_words.append(key)
		return ret_words
		
	k_groups = splitdata(k, x, y, train_subjects)
	#now we have the k groups, assign each one as test once and run tests!
	print "groups split"
	lda_train_results = []
	lda_test_results = []
	#lda_iso_results = []
	nb_train_results = []
	nb_test_results = []
	#nb_iso_results = []
	lr_train_results = []
	lr_test_results = []
	#r_iso_results = []
	
	for i in xrange(k):
		print "K Fold number " + str(i)
		test = k_groups[i]
		train = []
		train.append([]) #x
		train.append([]) #y
		train.append([]) #words
		for j in xrange(k):
			if(j != i):
				train[0].extend(k_groups[j][0])
				train[1].extend(k_groups[j][1])
				train[2].extend(k_groups[j][2])
		# Perform a word count of the test data and get 15 top subjects.
		top_subjects = word_count(train[2], 50)
		
		# Extend train[0] and test to contain features for the top subjects.
		
		for i in xrange(0, len(train[0])):
			if type(train[0][i]) != list:
				train[0][i] = train[0][i].tolist()
				
		for i in xrange(0, len(train[0])):
			subjects = train[2][i]
			bits = [0] * len(top_subjects)
			for s in subjects:
				if s in top_subjects:
					bits[top_subjects.index(s)] = 1
			train[0][i].extend(bits)
		
		for i in xrange(0, len(test[0])):
			subjects = test[2][i]
			bits = [0] * len(top_subjects)
			for s in subjects:
				if s in top_subjects:
					bits[top_subjects.index(s)] = 1
			test[0][i].extend(bits)
		
		#Now we have test and training data... what shall we do?
		#train on LDA
		#print "Training LDA..."
		#(prob, mean, cov) = lda.trainLDA(train[0], train[1])
		#print str(prob) + "\t" + str(mean) + "\t" + str(cov)
		#print "DONE training LDA."
		print "Training NB..."
		(py, theta) = naivebayes.trainNaiveBayesMN(train[0], train[1])
		#print str(py) + "\t" + str(theta)
		print "DONE training NB"
		print "Training Logistic Regression..."
		t_x = copy.deepcopy(train[0])		
		for i in xrange(len(t_x)):
			temp_row = [1]
			temp_row.extend(t_x[i])
			t_x[i] = temp_row		
		(wvector, scales) = logisticregression.trainLogisticReg(0.01, 0.00001, 100, copy.deepcopy(t_x), copy.deepcopy(train[1]))
		#print str(wvector)
		print "DONE training Logistic Regression.\n"
		
		lr_model = linmod.LogisticRegression()
		lr_model.fit(t_x, train[1])
		for model, name in ((lr_model, "LR"),):
			tp, tn, fp, fn = 0, 0, 0, 0
			for i in xrange(0, len(t_x)):
				val = model.predict(t_x[i])
				if (val == 1 and train[1][i] == 1):
					tp += 1
				elif (val == 1 and train[1][i] == 0):
					fp += 1
				elif (val == 0 and train[1][i] == 0):
					tn += 1
				elif (val == 0 and train[1][i] == 1):
					fn += 1
			print "%s - TP: %d, FP: %d, TN: %d, FN: %d" % (name, tp, fp, tn, fn)
			
		#get Prediction Errors on left out set
		lr_test_error = logisticregression.getConfusionMatrix(wvector,scales, copy.deepcopy(test[0]), copy.deepcopy(test[1]))
		lr_train_error = logisticregression.getConfusionMatrix(wvector,scales, copy.deepcopy(train[0]), copy.deepcopy(train[1]))
		#lr_iso_error = logisticregression.getConfusionMatrix(wvector,scales, isotest_x, isotest_y)
		#lda_test_error = lda.getConfusionMatrix(prob, mean, cov, test[0], test[1])
		#lda_train_error = lda.getConfusionMatrix(prob, mean, cov, train[0], train[1])
		#lda_iso_error = lda.getConfusionMatrix(prob, mean, cov, isotest_x, isotest_y)
		nb_test_error = naivebayes.getConfusionMatrixMN(py, theta, test[0], test[1])
		nb_train_error = naivebayes.getConfusionMatrixMN(py, theta, train[0], train[1])
		#nb_iso_error = naivebayes.getConfusionMatrixMN(py, theta, isotest_x, isotest_y)
		
		#add to sets the false positives (for now)
		lr_train_results.append(lr_train_error)
		lr_test_results.append(lr_test_error)
		#lr_iso_results.append(lr_iso_error)
		#lda_train_results.append(lda_train_error)
		#lda_test_results.append(lda_test_error)
		#lda_iso_results.append(lda_iso_error)
		nb_train_results.append(nb_train_error)
		nb_test_results.append(nb_test_error)
		#nb_iso_results.append(nb_iso_error)
		
	#calc average training and test error for each algorithm
	#avr_lda_train = averageconfusionmatrix(lda_train_results)
	#avr_lda_test = averageconfusionmatrix(lda_test_results)
	#avr_lda_iso = averageconfusionmatrix(lda_iso_results)
	avr_lr_train = averageconfusionmatrix(lr_train_results)
	avr_lr_test = averageconfusionmatrix(lr_test_results)
	#avr_lr_iso = averageconfusionmatrix(lr_iso_results)
	avr_nb_train = averageconfusionmatrix(nb_train_results)
	avr_nb_test = averageconfusionmatrix(nb_test_results)
	#avr_nb_iso = averageconfusionmatrix(nb_iso_results)
	#return [avr_lr_train, avr_lr_test, avr_lr_iso, avr_lda_train, avr_lda_test, avr_lda_iso, avr_nb_train, avr_nb_test, avr_nb_iso]
	#return [avr_lr_train, avr_lr_test, avr_lda_train, avr_lda_test, avr_nb_train, avr_nb_test]
	return [avr_lr_train, avr_lr_test, avr_nb_train, avr_nb_test]
	#return [avr_lr_train, avr_lr_test]
def averageconfusionmatrix(listofmatrices):
	tp = 0
	tn = 0
	fp = 0
	fn = 0
	for matrix in xrange(len(listofmatrices)):
		tp += listofmatrices[matrix][0][0]
		fp += listofmatrices[matrix][0][1]
		tn += listofmatrices[matrix][1][1]
		fn += listofmatrices[matrix][1][0]
		
	tp = tp / float(len(listofmatrices))
	fp = fp / float(len(listofmatrices))
	tn = tn / float(len(listofmatrices))
	fn = fn / float(len(listofmatrices))
	
	return [tp, fp, tn, fn]

def getfeaturesubset(indices, x):
	new_set = []
	for row in x:
		new_row = []
		for col in xrange(len(row)):
			if col in indices:
				new_row.append(row[col])
		new_set.append(new_row)
	return new_set
	
#subsets chosen are hard coded
def testfeaturesubsets(k, x, y, subjects, iso_x, iso_y, iso_subjects):
	if(len(x) != len(y)):
		print "data not same length."
		return
	
	results = {}
	#full dataset
	results["dataset_full"] = kfolds_all_algos(k, x, y, subjects, iso_x, iso_y, iso_subjects)

	return (results)#,  temp_x_onlyspon, temp_x_congress, temp_x_nopers)

def accuracycalc(matrix):
	return (matrix[0] + matrix[2])/float(matrix[0] + matrix[3] + matrix[1] + matrix[2])

def errorcalc(matrix):
	return (matrix[1] + matrix[3])/float(matrix[0] + matrix[3] + matrix[1] + matrix[2])
	
def precisioncalc(matrix):
	if(float(matrix[0] + matrix[1]) == 0):
		return 0
	return (matrix[0])/float(matrix[0] + matrix[1])
	
def recallcalc(matrix):
	if(float(matrix[0] + matrix[3]) == 0):
		return 0
	return (matrix[0])/float(matrix[0] + matrix[3])
	
def fcalc(matrix):
	precision = precisioncalc(matrix)
	recall = recallcalc(matrix)
	if precision == 0 and recall == 0:
		return -1
	return 2*(precision*recall)/(precision + recall)

def kfolds_control(k, x, y, subjects):
	# set aside 20% to do final testing on
	if(len(x) != len(y)):
		print "data not right size"
		return
	# Shuffle the training data.
	combined = zip(x, y, subjects)
	random.shuffle(combined)
	x, y, subjects = zip(*combined)
	
	x = list(x)
	y = list(y)
	subjects = list(subjects)
	
	testset = [[], [], []] #x and y and subjects
	sizeoftest = int(0.1 * len(x))
	for i in xrange(sizeoftest):
		chosen_index = random.randint(0, len(x) - 1)
		testset[0].append(x[chosen_index])
		testset[1].append(y[chosen_index])
		testset[2].append(subjects[chosen_index])
		del x[chosen_index]
		del y[chosen_index]
		del subjects[chosen_index]
	
	#we have test set, now create training set with 50/50 balance
	num_positives = 0
	train_x = []
	train_y = []
	train_words = []
	oversample_rate = 1
	for i in xrange(len(y)):
		if(y[i] == 1):
			for _ in xrange(oversample_rate):
				train_x.append(x[i])
				train_y.append(y[i])
				train_words.append(subjects[i])
				num_positives = num_positives + 1
	
	print "number of positives: " + str(num_positives)
	
	#now compliment with fails
	# Randomize choosing the negatives so they are spread out over time.
	while (num_positives > 0):
		i = random.randint(0, len(x)-1)
		if(y[i] == 0):
			train_x.append(x[i])
			train_y.append(y[i])
			train_words.append(subjects[i])
			del subjects[i]
			del x[i]
			del y[i]
			num_positives = num_positives - 1
	
	print "size of x: " + str(len(train_x)) + " size of y: " + str(len(train_y)) + "size of testset: " + str(len(testset[0]))

	#start off by doing feature selection
	#(results,  temp_x_onlyspon, temp_x_congress, temp_x_nopers) = testfeaturesubsets(k, train_x, train_y, testset[0], testset[1])
	(results) = testfeaturesubsets(k, train_x, train_y, train_words, testset[0], testset[1], testset[2])
	for set_name, algo_data in results.iteritems():
		print (set_name + "\n LR train" + "\t" + str(algo_data[0]) + "\t accuracy: " + str(accuracycalc(algo_data[0]))  + "\t accuracy: " + str(fcalc(algo_data[0]))
			+ "\n LR valid" + "\t" + str(algo_data[1]) + "\t A: " + str(accuracycalc(algo_data[1]))  + "\t F: " + str(fcalc(algo_data[1]))
			#+ "\n LR isolated" + "\t" + str(algo_data[2]) + "\t Acuracy: " + str(accuracycalc(algo_data[2]))  + "\t F: " + str(fcalc(algo_data[2]))
			#+ "\n LDA train" + "\t" + str(algo_data[2]) + "\t A: " + str(accuracycalc(algo_data[2]))  + "\t F: " + str(fcalc(algo_data[2]))
			#+ "\n LDA valid" + "\t" + str(algo_data[3]) + "\t A: " + str(accuracycalc(algo_data[3]))  + "\t F: " + str(fcalc(algo_data[3]))
			#+ "\n LDA isolated" + "\t" + str(algo_data[5]) + "\t A: " + str(accuracycalc(algo_data[5]))  + "\t F: " + str(fcalc(algo_data[5]))
			+ "\n NB train" + "\t" + str(algo_data[2]) + "\t A: " + str(accuracycalc(algo_data[2]))  + "\t F: " + str(fcalc(algo_data[2]))
			+ "\n NB valid" + "\t" + str(algo_data[3]) + "\t A: " + str(accuracycalc(algo_data[3]))  + "\t F: " + str(fcalc(algo_data[3]))
			#+ "\n NB isolated" + "\t" + str(algo_data[8]) + "\t A: " + str(accuracycalc(algo_data[8]))  + "\t F: " + str(fcalc(algo_data[8]))
			+ "\n")


		
print "parsing..."
(x,y, subjects) = csv_parser.parse_data_with_subjects("final_data_with_subjects.csv", "subjects.csv")
print "parsed"
kfolds_control(4, x, y, subjects)

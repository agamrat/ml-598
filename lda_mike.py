import numpy
import math

# Reads the data from a csv file and returns X and Y.
def read_data_from_csv(file_name):
	X = numpy.genfromtxt(file_name, delimiter=',')
	X = numpy.delete(X, 0, 0)
	X = numpy.delete(X, 3, 1)
	Y = X[:,-1]
	X = numpy.delete(X, -1, 1)
	num_false = 0
	return X, Y

class LDA_classifier(object):
	class NotTrainedException(Exception):
		pass
	
	def __init__(self):
		self.trained = True

	# Calculates the mu_1, mu_0, sigma, P_0, P_1 needed for the classifier.	
	def train(self, X, Y):
		N_0 = 0
		N_1 = 0
		for y in Y:
			if int(y) == 1:
				N_1 += 1
			else:
				N_0 += 1
		# Calculate the probabilities we are in each class.
		self.P_0 = float(N_0)/float(N_0 + N_1)
		self.P_1 = float(N_1)/float(N_0 + N_1)
		# Calculate the class means.
		self.mu_0 = numpy.zeros(X.shape[1])
		self.mu_1 = numpy.zeros(X.shape[1])
		for i in xrange(0, X.shape[1]):
			if (Y[i] == 0):
				self.mu_0 += X[i,:]
			elif (Y[i] == 1):		
				self.mu_1 += X[i,:]
		self.mu_0 = self.mu_0 / float(N_0)
		self.mu_1 = self.mu_1 / float(N_1)
		# Caclulate the covariance matrix.
		self.sigma = numpy.zeros((X.shape[1], X.shape[1]))
		for k in [0, 1]:
			for i in xrange(0, X.shape[0]):
				
				if (Y[i] == k):
					self.sigma += numpy.transpose(X[i,:])*X[i,:]/N_1
		self.trained = True

	def predict(self, x):
		if (not self.trained):
			raise NotTrainedException()
		term1 = math.log(float(self.P_1)/self.P_0)

		mu_sum = self.mu_0 + self.mu_1
		mu_diff = self.mu_0 - self.mu_1
		mu_sum = mu_sum.reshape((1,mu_sum.size))
		mu_diff = mu_diff.reshape((1,mu_diff.size))
		term2 = (1.0/2.0) * mu_sum.dot(numpy.linalg.pinv(self.sigma).dot( numpy.transpose(mu_diff)))[0]
		term3 = numpy.transpose(x).dot(numpy.linalg.pinv(self.sigma).dot((numpy.transpose(mu_diff))))[0]
		print term1, term2, term3
		result = term1 - term2 + term3
		if (result >= 0):
			return 1
		return 0
if __name__=='__main__':
	lda = LDA_classifier()
	X, Y = read_data_from_csv('data_clean.csv')
	for i in xrange(0, X.shape[0]):
		for j in xrange(0, X.shape[1]):
			if type(X[i][j]) < 0:
				print type(X[i][j]), i, j
	lda.train(X, Y)
	results = [lda.predict(X[i,:]) == Y[i] for i in xrange(0, 100)]
	print results

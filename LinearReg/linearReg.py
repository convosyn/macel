import numpy as np;
import matplotlib.pyplot as plt

def costFunction(X, y, theta):
	#calculate the number of training examples
	m = X.shape[0];
	#J stores the value of the cost obtained
	J = (1/(2 * m)) * ((X.dot(theta) - y) ** 2).sum()
	return J
	
def gradient(X, y, theta, alpha, num_iters):
	# the for loop iterates over the 
	
	m = X.shape[0]
	J = np.zeros((num_iters, 1))
	
	for i in range(num_iters):
		theta -= ((alpha / m) * X.transpose().dot(X.dot(theta) - y))
		J[i] = costFunction(X, y, theta)
		#print(J[i]);
	return (J, theta)


def plot(X, y, options):
	n = X.shape[1]
	plt.figure(1)
	rows = np.floor(np.sqrt(n))
	columns = np.ceil(n/rows)
	for i in range(1,n):
		plt.subplot(rows, columns, i)
		plt.plot(X[:, i], y, options)
	plt.show()
	
def normalize(X):
	mean_col = X.mean(0)
	std_col = X.std(0)
	X = (X - mean_col)/ std_col;
	return X
	

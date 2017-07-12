import numpy as np

def fcost(theta, X, y):
	theta = theta.reshape((-1, 1))
	m = X.shape[0]
	hx = fsigmoid(X.dot(theta))
	J = (-1/m) * np.sum(np.sum(y * np.log(hx) + (1-y) * np.log(1 - hx)))
	grad = (1/m) * (X.T.dot(hx - y))
	return J, grad

def fsigmoid(z):
	return (1 / (1 + np.exp(-z)))

def foptim(init_theta, alpha, max_iters, *args):
	X, y = args[0], args[1]
	init_theta = init_theta.reshape((-1, 1))
	J = np.zeros((max_iters, 1))
	for i in range(1,max_iters):
		J[i] = fcost(init_theta, X, y)[0]
		init_theta = init_theta - alpha * fcost(init_theta, X, y)[1]

	return init_theta, J
	
def fnormalize(X):
	return (X - X.mean(axis = 0))/X.std(axis = 0);

def fpredict(theta, X, threshold = 0.5):
	theta = theta.reshape((-1, 1))
	y = np.int8(fsigmoid(X.dot(theta)) >= threshold)
	return y


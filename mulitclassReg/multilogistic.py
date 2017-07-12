import numpy as np
from scipy import optimize
from scipy import special
from matplotlib import pyplot as plt

def fcost(theta, X, y, lambd):
	m = X.shape[0]
	theta = theta.reshape((-1, 1))
	hx = fsigmoid(X @ theta)
	log_loss_one = y * np.log(hx)
	log_loss_zero = (1 - y) * np.log(1 - hx)
	log_loss = (-1 / m) * np.sum((log_loss_one + log_loss_zero))
	J = log_loss + (lambd * np.linalg.norm(theta[1:]) ** 2) / (2 * m)
	return J

def fnormalize(X):
	res = X - X.mean(axis=0)
	return res

def fsigmoid(z):
	res = special.expit(z)
	return res

def fgrad(theta, X, y, lambd):
	m = X.shape[0]
	in_theta = theta.reshape((-1, 1))
	hx = fsigmoid(X @ in_theta)
	grad = (1 / m) * (X.T @ (hx - y))
	grad[1:] += (lambd / m) * grad[1:]
	return grad.flatten()

def fone_vs_rest(X, y, lambd):
	uniq = np.unique(y)
	print(uniq)
	k = uniq.size
	n = X.shape[1]
	Theta = np.zeros((n, k))
	for i in range(k):
		print("\nloop: ", i)
		y_one = (y == uniq[i])
		theta = np.zeros((n, 1))
		myargs=(X, y_one, lambd)
		theta = optimize.minimize(fcost, theta, method="Newton-CG", jac=fgrad, args=myargs).x
		#plt.plot(np.arange(J.size), J, "r-")
		#plt.xlabel("iters")
		#plt.ylabel("J")
		#plt.show()
		#print(theta)
		Theta[:, [i]] = theta.reshape((-1, 1))
	return Theta

def fpredict(Theta, X):
	pred = fsigmoid(X @ Theta)
	#print("pred: ", pred)
	pred = np.argmax(pred, axis=1).reshape((-1, 1))
	return pred


def fminimize(cost, theta, grad, args=(), alpha=1e-5, epsilon=1e-4, max_iters = 100):
	ftheta = theta.reshape((-1, 1))
	J = np.zeros((max_iters, 1))
	prev_cost = cost(ftheta, *args)
	J[0] = prev_cost
	cur_step = 1
	while cur_step < max_iters:
		#print(prev_cost, end="|")
		g = grad(ftheta, *args)
		new_theta = ftheta - alpha * g
		new_cost = cost(new_theta, *args)
		if np.abs(new_cost - prev_cost) < epsilon:
			break
		if(prev_cost < new_cost and np.isnan(new_cost)):
			print("alpha decreasing", end="|")
			print("step increment", end="|")
			alpha = alpha *  0.1
		else:
			print(".",end="")
			ftheta = new_theta
			cur_step += 1
			prev_cost = new_cost
			J[cur_step-1] = prev_cost
	return ftheta, J

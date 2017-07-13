
# # # # # # # # # # # # # # # # # # # # # # # # # # #
#   simClev copyright (c) 2017.All rights resrved   #
# # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
from matplotlib import pyplot as plt

class Neural:

	#TODO:
	#1 - Add regularization to the cost and grad functions
	#2 - add support for more than one hidden layers
	#3 - Inbuilt support for train and validate
	#4 - optimize the mimization function
	#5 - Add support for more minimization algorithm types
	#6 - test predictor

	def __init__(self, number_of_hidden_layers = 1, alpha = 1e-3, epsilon = 1e-7, max_iters = 100, lambd = 1, hidden_layer_size = 75, debug_epsilon = 1e-4, init_epsilon = 1e-1*2):
		self.number_of_hidden_layers = number_of_hidden_layers
		self.alpha = alpha
		self.epsilon = epsilon
		self.max_iters = max_iters
		self.lambd = lambd
		self.hidden_layer_size = hidden_layer_size
		self.debug_epsilon = debug_epsilon
		self.init_epsilon = init_epsilon

	def _init_theta(self, input_layer_size, hidden_layer_size, num_labels, single_layer = True):
		number_of_hidden_layers = self.number_of_hidden_layers
		if single_layer == True:
			_theta = np.random.rand(1, (hidden_layer_size * (input_layer_size + 1)) + (num_labels * (hidden_layer_size + 1)))
			
		else:
			pass
			#TODO: for multiple hidden layer system 
		return _theta

	def _y_to_classifier(self, y):
		_num_labels = np.unique(y).size
		m = y.size
		_temp = np.int0(np.tile(np.arange(_num_labels), m).reshape((m, _num_labels)) == np.tile(y.reshape((-1, 1)), _num_labels))
		return _temp, _num_labels

	def fit(self, X, y, perform_gradient_checking = True, plot_cost = True, show_cost = True):
		y, self.num_labels = self._y_to_classifier(y)
		self.input_layer_size = X.shape[1]
		self.Theta = self._init_theta(self.input_layer_size, self.hidden_layer_size, self.num_labels)
		self.Initial_Theta = self.Theta
		
		X = self._norm(X)
		J = self._minimize(X, y)

		if perform_gradient_checking is True:
			self._grad_debug(X, y, self.Theta)
		if show_cost == True:
			self._plot(J, show_cost)

	def _plot(self, func, show_cost = True):
		func = np.array(func)

		if show_cost  == True:
			print("cost function: {!s}".format(func.flatten()))

		plt.plot(np.arange(func.size), func, "b-")
		plt.show()

	def _feed_forward(self, X, Theta):
		input_layer_size = self.input_layer_size
		hidden_layer_size = self.hidden_layer_size
		num_labels = self.num_labels

		Theta = Theta.reshape((1, -1))
		Theta1 = Theta[:, :hidden_layer_size * (input_layer_size + 1)].reshape((hidden_layer_size, input_layer_size + 1))
		Theta2 = Theta[:, hidden_layer_size * (input_layer_size + 1):].reshape((num_labels, hidden_layer_size + 1))

		m = X.shape[0]
		a1 = np.column_stack([np.ones((m, 1)), X])

		z2 = a1 @ Theta1.T
		a2 = self._sigmoid(z2)
		a2_biased = np.column_stack([np.ones((m, 1)), a2])

		z3 = a2_biased @ Theta2.T
		a3 = self._sigmoid(z3)

		#print("feed_forward_shapes: {!s} {!s} {!s}".format(a1.shape, a2.shape, a3.shape))
		#print("feed_for_values: a1:\n{!s}\nz2:\n{!s}\na2_biased:\n{!s}\nz3:\n{!s}\na3:\n{!s}\n".format(a1, z2, a2_biased, z3, a3))

		return a3, z3, a2_biased, z2, m
		
	def _back_prop(self, a3, z3, a2_biased, z2, m, Theta, X, y):
		input_layer_size = self.input_layer_size
		hidden_layer_size = self.hidden_layer_size
		num_labels = self.num_labels

		Theta = Theta.reshape((1, -1))
		Theta1 = Theta[:, :hidden_layer_size * (input_layer_size + 1)].reshape((hidden_layer_size, input_layer_size + 1))
		Theta2 = Theta[:, hidden_layer_size * (input_layer_size + 1):].reshape((num_labels, hidden_layer_size + 1))
		d3 = a3 - y

		gzd = np.column_stack([np.ones((m, 1)), self._sigmoid_grad(z2)])
		d2 = (d3 @ Theta2) * gzd
		d2 = d2[:, 1:]
		D2 = d3.T @ a2_biased
		D1 = d2.T @ np.column_stack([np.ones((m, 1)), X])

		D2 /= m
		D1 /= m

		#print("back_prop_shapes: D1: {!s} {!s} {!s} {!s}".format(d2.shape, d3.shape, D1.shape, D2.shape))
		#print("bac_prop_values: d2:\n{!s}\nd3:\n{!s}\nD1:\n{!s}\nD2:\n{!s}".format(d2, d3, D1, D2))

		grad = np.column_stack([D1.reshape((1, -1)), D2.reshape((1, -1))]).reshape((1, -1))
		return grad
	
	def _grad_function(self, X, y, Theta):
		values = self._feed_forward(X, Theta)		
		grad = self._back_prop(*values, Theta, X, y)
		return grad

	def _cost_function(self, X, y, Theta):
		a3, _, _, _, m = self._feed_forward(X, Theta)
		J = (-1/m) * np.sum((y * np.log(a3) + (1-y) * np.log(1 - a3)))
		return J

	def _sigmoid(self, z):
		return (1 / (1 + np.exp(-z)))

	def _sigmoid_grad(self, z):
		gz = self._sigmoid(z)
		return gz * (1 - gz)
		
	def _minimize(self, X, y):
		theta = self.Theta
		J = []
		prev_cost = self._cost_function(X, y, theta)
		#print("x:\n{!s}\ny:\n{!s}".format(X, y))
		for i in range(self.max_iters):
			print("current iteration: {!s}\r".format(i), end="")
			theta -= self.alpha * self._grad_function(X, y, theta)
			cur_cost = self._cost_function(X, y, theta)
			J.append(cur_cost)
			if(cur_cost < prev_cost and np.abs(prev_cost - cur_cost) < self.epsilon):
				break

		self.Theta = theta

		return J

	def _norm(self, X):
		return X - np.mean(X, axis=0)

	def predict(self, X):
		X = self._norm(X)
		a3, _, _, _, _ = self._feed_forward(X, self.Theta)
		y = np.argmax(a3, axis=1)
		return a3, y

	def _grad_debug(self, X, y, fin_theta):
		n = fin_theta.size
		fin_theta = fin_theta.reshape((-1, 1))
		numgrad = np.zeros(fin_theta.shape)
		perturb = np.zeros(fin_theta.shape)
		print("final_theta_shape: {!s}".format(fin_theta.shape))
		e = self.epsilon
		for i in range(n):
			perturb[i] = e
			theta_up = fin_theta + perturb
			theta_down = fin_theta - perturb
			numgrad[i] = (self._cost_function(X, y, theta_up) - self._cost_function(X, y, theta_down)) / (2 * e)
			perturb[i] = 0

		grad = self._grad_function(X, y, fin_theta)
		print("Similarity found: ")
		print(np.column_stack([grad.reshape((-1, 1)), numgrad]))

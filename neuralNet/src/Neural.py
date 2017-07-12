import numpy as np
import scipy.special


def fgrad(Theta, input_layer_size, hidden_layer_size, num_labels, X, y, lambd = 1):

	Theta = Theta.reshape((-1, 1))
	m = X.shape[0]

	#print("Theta_dim: {!s}".format(Theta.shape))

	Theta1 = Theta[:((hidden_layer_size) * (input_layer_size + 1)), :].reshape((hidden_layer_size, (input_layer_size + 1)))
	Theta2 = Theta[((hidden_layer_size) * (input_layer_size + 1)):, :].reshape((num_labels, hidden_layer_size + 1))

	Theta1_grad = np.zeros((Theta1.shape))
	Theta2_grad = np.zeros((Theta2.shape))

	#feed forward propagation for three layer neural network
	a1 = np.column_stack([np.ones((m, 1)), X])
	z2 = a1 @ Theta1.T
	a2 = np.column_stack([np.ones((m, 1)), fsigmoid(z2)])
	z3 = a2 @ Theta2.T
	a3 = fsigmoid(z3)

	#print("values feedforward: a1 \n {!s} \n z2 \n {!s} \n a2 \n {!s} \n z3 \n {!s} \n a3 \n {!s}".format( a1, z2,a2,z3,a3))
	print("values feedforward: a1: {!s} | z2: {!s} | a2: {!s} | z3: {!s} | a3: {!s}".format( a1.shape, z2.shape, a2.shape, z3.shape, a3.shape))
	#backward propagation for three layer system
	#since the system is assumed to be a three layer system only two deltas will be there one for each layer theta values
	del3 = a3 - y
	del2 = (del3 @ Theta2) * np.column_stack([np.ones((m, 1)), fsigmoid_grad( fsigmoid(z2))])

	Del2 = del3.T @ a2
	Del1 = del2[:, 1:].T @ a1

	Del2 /= m
	Del1 /= m

	#print("back prop: del3 \n {!s} \n del2 \n {!s} \n DEl2 \n {!s} \n Del1 \n {!s} \n".format(del3, del2, Del2, Del1))
	print("back prop: del3: {!s} | del2: {!s} | DEl2: {!s} | Del1: {!s}".format(del3.shape, del2.shape, Del2.shape, Del1.shape))

	Del2[:, 1:] += (lambd / m) * Theta2[:, 1:]
	Del1[:, 1:] += (lambd / m) * Theta1[:, 1:]

	Theta2_grad[:, 1:] = Del2[:, 1:]
	Theta1_grad[:, 1:] = Del1[:, 1:]

	#finally unroll the vectors into a single value
	grad = np.row_stack([Theta1_grad.reshape((-1, 1)), Theta2_grad.reshape((-1, 1))])
	grad = grad.reshape((-1, 1))

	#print("grad: {!s}".format(grad.shape))

	return grad.flatten()

def fcost(Theta, input_layer_size, hidden_layer_size, num_labels, X, y, lambd = 1):

	Theta = Theta.reshape((-1, 1))
	#print("Theta_dim: {!s}".format(Theta.shape))
	m = X.shape[0]

	Theta1 = Theta[:((hidden_layer_size) * (input_layer_size + 1)), :].reshape((hidden_layer_size, (input_layer_size + 1)))
	Theta2 = Theta[((hidden_layer_size) * (input_layer_size + 1)):, :].reshape((num_labels, hidden_layer_size + 1))

	#feed forward propagation for three layer neural network
	#finally calculate the cost function
	a1 = np.column_stack([np.ones((m, 1)), X])
	z2 = a1 @ Theta1.T
	a2 = np.column_stack([np.ones((m, 1)), fsigmoid(z2)])
	z3 = a2 @ Theta2.T
	a3 = fsigmoid(z3)

	J = (-1 / m) * np.sum(y * np.log(a3) + (1-y) * np.log(1 - a3))
	#add the regularization term to the cost 
	#only after the first rows should be added to the cost
	reg_exp = (lambd / (2 * m)) * (np.sum(Theta1[:, 1:] ** 2) + np.sum(Theta2[:, 1:] ** 2))
	J += reg_exp

	return J

fsigmoid_grad = lambda z: z * (1 - z)

fsigmoid = lambda z: scipy.special.expit(z)

def checkGradient():
	pass

def predict(X, Theta, input_layer_size, hidden_layer_size, num_labels):

	Theta = Theta.reshape((-1, 1))
	#print("Theta_dim: {!s}".format(Theta.shape))
	m = X.shape[0]

	Theta1 = Theta[:((hidden_layer_size) * (input_layer_size + 1)), :].reshape((hidden_layer_size, (input_layer_size + 1)))
	Theta2 = Theta[((hidden_layer_size) * (input_layer_size + 1)):, :].reshape((num_labels, hidden_layer_size + 1))

	#feed forward propagation for three layer neural network
	#finally calculate the cost function
	a1 = np.column_stack([np.ones((m, 1)), X])
	z2 = a1 @ Theta1.T
	a2 = np.column_stack([np.ones((m, 1)), fsigmoid(z2)])
	z3 = a2 @ Theta2.T
	a3 = fsigmoid(z3)

	np.savetxt("pred_temp.txt", a3)
	pred = np.argmax(a3, axis=1).reshape((-1, 1))

	return pred
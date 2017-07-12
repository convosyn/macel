import numpy as np
import pandas as pd
import DataManipulate
import Optimize
import Neural
import sys
import scipy.optimize
import matplotlib.pyplot

def main(filename):

	X_train, y_train, X_val, y_val, num_labels = DataManipulate.load_train_val_classifier(filename)

	input_layer_size = X_train.shape[1]
	hidden_layer_size = 55

	Theta_dim = (input_layer_size + 1) * hidden_layer_size + num_labels * (hidden_layer_size + 1)
	Theta = np.random.rand(Theta_dim, 1)

	lambd = 1
	myargs = (input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambd)
	#Theta = scipy.optimize.minimize(Neural.fcost, Theta, method="Newton-CG",jac=Neural.fgrad, args=myargs).x
	Theta, J = Optimize.fminimize(Neural.fcost, Theta, Neural.fgrad, args=myargs, alpha=1e-1 + 1e-2 * 5, max_iters=10)

	matplotlib.pyplot.plot(np.arange(J.size), J, "g-")
	matplotlib.pyplot.show()

	print("Theta: \n{!s}\n".format(Theta))

	pred = Neural.predict(X_val, Theta.reshape((-1, 1)), input_layer_size, hidden_layer_size, num_labels)

	y_val = np.argmax(y_val, axis=1)
	print("accuracy: {!s}".format(float(np.mean(y_val.reshape((-1, 1)) == pred.reshape((-1, 1))))))

if __name__ == "__main__":

	if(len(sys.argv) < 2):
		print("Usage: program-name <file-name>")	
		sys.exit(0)

	main(sys.argv[1])
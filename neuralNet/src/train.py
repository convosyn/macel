import numpy as np
import Neural
import Optimize
from matplotlib import pyplot as plt

X = np.random.rand(10, 4)
y = np.random.randint(0, 3, (10, 1))
z = np.zeros((10, 3))
for i in range(y.size):
	z[i, y[i]] = 1
y = z

input_layer_size = 4
hidden_layer_size = 3
num_labels = 3
lambd = 1
myargs = (input_layer_size, hidden_layer_size, num_labels, X, y, lambd)
Theta_dim = hidden_layer_size * (input_layer_size + 1) + num_labels * (hidden_layer_size + 1)
#print("print theta dimension: {!s}".format(Theta_dim))
Theta = np.random.rand(Theta_dim, 1)

Theta, J = Optimize.fminimize(Neural.fcost, Theta, Neural.fgrad, myargs, epsilon = 1e-7, max_iters= 50)

plt.plot(np.arange(J.size), J, "y-")
plt.show()

print("Theta: \n{!s}\nJ: \n{!s} ".format(Theta, J))




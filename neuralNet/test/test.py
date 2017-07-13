import numpy as np
from src import Neural 

x = np.loadtxt("x_val")
y = np.loadtxt("y_val")

clf = Neural.Neural(hidden_layer_size = 10, max_iters = 100, alpha = 1e-1 * 10, epsilon = 1e-4)
clf.fit(x, y.reshape((-1, 1)))


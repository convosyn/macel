import numpy as np
from src import Neural 

x = np.loadtxt("x_val")
y = np.loadtxt("y_val")

clf = Neural.Neural(hidden_layer_size = x.shape[1], max_iters = 1000, alpha = 1e-1 * 10, epsilon = 1e-5)
clf.fit(x, y.reshape((-1, 1)))

_, yp = clf.predict(x)
print("accuracy: {!s}%".format(float(np.mean(yp.reshape((-1, 1)) == y.reshape((-1, 1))) * 100)))



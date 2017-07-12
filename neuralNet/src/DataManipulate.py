import pandas as pd
import numpy as np
import Optimize

def load_train_val_classifier(filename:str, ratio:float=0.8, y_pos:int = 0, norm:bool = True):

	data = pd.read_csv(filename)
	data = np.array(data)
	X = np.column_stack([data[:, 0:y_pos], data[:, y_pos+1:]])
	y = np.column_stack([data[:, y_pos]]).reshape((-1, 1))

	if(norm == True):
		Optimize.normalize(X)

	print("X:\n{!s}\n{!s}\ny:\n{!s}\n{!s}\n".format(X[:10, :], X.shape, y[:10, :], y.shape))
	num_labels = np.unique(y).size
	m = X.shape[0]
	z = np.zeros((m, num_labels))
	for i in range(y.size):
		z[i, y[i]] = 1
	y = z

	m_train = int(0.8 * m)

	X_train = X[:m_train, :]
	y_train = y[:m_train, :]

	X_val = X[m_train:, :]
	y_val = y[m_train:, :]

	return X_train, y_train, X_val, y_val, num_labels 

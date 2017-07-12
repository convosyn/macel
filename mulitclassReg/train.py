import sys
import pandas as pd
import numpy as np
import multilogistic as mlg


def main(fin:str):
	print("loading data...", end="")
	data = pd.read_csv(fin)
	#data = np.random.shuffle(np.array(data), ax)
	#print(data)
	X = np.array(data)[:, 1:]
	X = mlg.fnormalize(X)
	y = np.array(data)[:, 0].reshape((-1, 1))
	print("Done!")

	print("header data X: \n{!s}\ny: \n{!s}".format(X[:10, :], y[:10, :]))
	print("splitting data -- model selection problem ... ", end="")
	m = X.shape[0]
	print("m: ", m)
	m_train = int(np.around(m * 0.8)) #using 60% of data for training classifier

	X_train = X[:m_train, :]	
	y_train = y[:m_train, :]

	X_train = np.column_stack([np.ones((X_train.shape[0], 1)), X_train])
	print("Done!")
	print("optimization started ...", end="")
	lambd = 1
	Theta = mlg.fone_vs_rest(X_train, y_train, lambd)
	print("Done!")

	print("Theta: \n", Theta)
	print("Cross- validation test ...", end="\n")
	m_val = int((m - m_train) * 1)
	X_val = X[m_train: m_train+m_val, :]
	X_val = X_val
	y_val = y[m_train: m_train+m_val, :]
	X_val = np.column_stack([np.ones((X_val.shape[0], 1)), X_val])
	pred = mlg.fpredict(Theta, X_val)
	print("saving ...", end="")
	#print(pred.shape, y_val.shape)
	np.savetxt("outParam.txt", Theta)
	print("optim params saved to outParam.txt.")
	fc = np.column_stack([pred, y_val]);
	np.savetxt("temp_cross_res.txt", fc)
	accuracy = float(np.mean(pred.reshape((-1, 1)) == y_val.reshape((-1, 1)))) * 100
	print("accuracy: ", accuracy, "%")


if __name__ == "__main__":
	if(len(sys.argv) < 2):
		print("Usage:python3.x <prog-name> <input-file>")
		sys.exit(0)
	else:
		main(sys.argv[1])

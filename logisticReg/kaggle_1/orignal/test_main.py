import numpy as np
import logistic
import sys
import pandas as pd

def main(fin, optim, othresh, fout="final_pred.csv"):
	data = np.loadtxt(fin, delimiter=",")
	theta = np.loadtxt(optim)
	threshold = np.loadtxt(othresh)
	X = data[:, 1:]
	X = logistic.fnormalize(X)
	m = data.shape[0]
	X = np.column_stack([np.ones((m, 1)), X])
	pred = logistic.fpredict(theta, X)
	#print("accuracy: {!s}%".format(float(np.mean(pred[:, 0] == y)) * 100))
	pid = np.array(data[:, 0], dtype=np.uint64)
	to_save = list(zip(pid.flatten(), pred.flatten()))
	to_save = pd.DataFrame(data=to_save, columns=["PassengerId", "Survived"])
	to_save.to_csv(fout, index=None)

if __name__ == "__main__":
	if(len(sys.argv) < 3):
		print("Usage: <prog_name> <test-file> <param-file>")
		sys.exit(0)
	#print(sys.argv)
	main(sys.argv[1], sys.argv[2], sys.argv[3])
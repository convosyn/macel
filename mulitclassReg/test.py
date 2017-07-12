import pandas as pd
import numpy as np
import sys
import multilogistic as mlg

def main(fin:str, foptim:str, fout:str="out.csv"):
	data = pd.read_csv(fin)
	#print(np.array(data))
	X = np.array(data)
	X = mlg.fnormalize(X)
	X = np.column_stack([np.ones((X.shape[0], 1)), X])
	Theta = np.loadtxt(foptim)
	pred = mlg.fpredict(Theta, X)

	out = pd.DataFrame(np.column_stack([np.arange(1, pred.size+1), pred]))
	out.to_csv(fout, header=["ImageId", "Label"], index=None)
	print("output saved.")

if __name__ == "__main__":
	if(len(sys.argv) < 3):
		print("Usage: <prog_name> <inputfile> <param-file> [out file]")
	main(sys.argv[1], sys.argv[2])



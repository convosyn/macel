import os
import numpy as np
import logistic
import sys
from matplotlib import pyplot as plt

def main(filename, fout:str = "optim_val.txt", falpha:str="optim_alpha.txt"):
	data = np.loadtxt(filename, delimiter=",")
	m, n = data.shape
	np.random.shuffle(data)
	m_train = int(np.round(m * 0.9))

	#obtaining the training data
	X_train = data[:m_train, 1:]
	X_train = logistic.fnormalize(X_train)
	X_train = np.column_stack([np.ones((m_train, 1)), X_train])
	y_train = data[:m_train, 0].reshape((-1, 1))

	init_theta = np.zeros((n, 1))

	#print("Checking fcost ...", logistic.fcost(init_theta, X_train[0:4], y_train[0:4]), sep="\n")
	#print("...Successfully executed!")

	print("optimizing ...")
	alpha = np.linspace(0.05, 0.05, 1)
	max_iters = np.linspace(1000, 1000, 1)
	width = alpha.size * max_iters.size
	Theta = np.zeros((n, width))

	print("calculating for all combos ...")
	#plt.figure();
	i = 0
	for al in alpha:
		for n_iter in max_iters:
			print("checking: " + str(i) + "\r", end="")
			Theta[0:n, [i]], J = logistic.foptim(init_theta, al, int(n_iter), X_train, y_train)
			#plt.subplot(alpha.size, max_iters.size, i+1)
			#plt.plot(np.arange(1, J.size+1), J)
			i += 1
	#plt.show()

	print("Done checking!")
	print("cross-validating ... \n")

	#obtaining the testing data
	X_val = data[m_train:, 1:] 
	X_val = logistic.fnormalize(X_val)
	X_val = np.column_stack([np.ones((m - m_train, 1)), X_val])
	y_val = data[m_train:, 0].reshape((-1, 1))

	threshold = np.linspace(0.3, 0.7, 10)
	accuracy = np.zeros((width, threshold.size))
	for i in range(width):
		for j in range(threshold.size):
			#J[i] = logistic.fcost(Theta[:, i], X_val, y_val)[0]
			pred_val = logistic.fpredict(Theta[:, i], X_val, threshold[j])
			pred_train = logistic.fpredict(Theta[:, i], X_train, threshold[j])
			acc_train = float(np.mean(pred_train == y_train)) * 100
			accuracy[i, j] = (float(np.mean(pred_val == y_val)) * 100)
			print("{!s} | {!s} | {!s}".format((i, j), accuracy[i, j], acc_train), end = " - ")
			tp =  int(np.sum((pred_val == 1) & (y_val == 1)))
			fp = int(np.sum((pred_val == 1) & (y_val == 0)))
			fn = int(np.sum((pred_val == 0) & (y_val == 1)))
			prec = tp/(tp + fp)
			recall = tp/(tp + fn)
			f1score = (2 * prec * recall) / (prec + recall)
			print("{!s} | {!s} | {!s}".format(prec, recall, f1score))

	val_max = np.argmax(accuracy)
	i = val_max / threshold.size
	j = val_max % threshold.size

	print("i,j: {!s}".format((i,j)))
	optim_theta = Theta[:, int(i)]
	optim_treshold = threshold[j]
	print("optim theta: ", optim_theta, sep="\n")
	print("optim threshold: ", optim_treshold)

	#saving the theta to a file
	print("Saving to: ", fout,"..." , end=" ")
	np.savetxt(fout, optim_theta)
	try:
		with open(falpha, "w") as ft:
			ft.write(str(optim_treshold))
	except IOError as e:
		print(str(e))
	
	print("Done!")

if __name__ == "__main__":
	if len(sys.argv) <= 1:
		print("usage: python <prog-name> <in-file> [<out-file>]")
		os.exit()
	print("Argument passed: ", sys.argv[1])
	if(len(sys.argv) <= 2):
		main(sys.argv[1])
	elif(len(sys.argv) <= 3):
		main(sys.argv[1], sys.argv[2])

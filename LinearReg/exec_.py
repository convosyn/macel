#!/usr/bin/env python3.6

import numpy as np
import linearReg as lr
import matplotlib.pyplot as plt
import sys


if(len(sys.argv) < 2):
	print("Usage: python3.6 <program-name> <dataSetLocation> [(opt)alpha]")
	sys.exit()

filename = sys.argv[1]
outfilename = "trainedParam.DAT";

print("filename:" + str(filename))

Data = np.loadtxt(filename)
m, n = Data.shape
trainX = Data[:, :n-1].reshape(-1, n-1)
trainX = lr.normalize(trainX)
trainX = np.column_stack([np.ones((m, 1)), trainX])

trainY = Data[:, n-1].reshape(-1, 1)
theta = np.zeros((n, 1)).reshape(-1, 1)

if(len(sys.argv) >= 3):
	alpha = float(sys.argv[2])
else:
	alpha = 0.1
	
num_iters = 100
#lr.plot(trainX, trainY, "rd")

#print("X -> \n" + str(trainX))
#print("y -> \n" + str(trainY))

J, theta = lr.gradient(trainX, trainY, theta, alpha, num_iters)
print("theta -> \n" + str(theta))
print("J -> \n" + str(J))

plt.plot(range(1, num_iters+1), J)
#lr.plot(trainX, trainX.dot(theta), "b.")
plt.xlabel("number of iterations")
plt.ylabel("J")
plt.show()

np.savetxt(outfilename, theta)

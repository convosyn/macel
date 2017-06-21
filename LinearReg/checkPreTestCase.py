import numpy as np
import linearReg as lr
import sys

if len(sys.argv) < 3:
	printf("usage: python3.6 <program_name> <trainedParameterFile> <testCaseFile> ")
	sys.exit();
	
dataSetFile = sys.argv[2]
trainedParamFile = sys.argv[1]

Data = np.loadtxt(dataSetFile)
theta = np.loadtxt(trainedParamFile)

print("Data.shape -> " + str(Data.shape))
m, n = Data.shape

targetX = Data[:, :n-1].reshape(-1, n-1)
targetX = lr.normalize(targetX)
targetX = np.column_stack([np.ones((m, 1)), targetX])
targetY = Data[:, n-1].reshape(-1, 1)
theta = theta.reshape(-1, 1)

calcY = targetX.dot(theta);

for i in range(0, m):
	print("{!s} - {!s}".format(calcY[i], targetY[i]))


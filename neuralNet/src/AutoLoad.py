import numpy as 
class AutoLoad:

	""" **this Module is intended at automatically loading the data for training through the neural network

	    - the parameter that assumed to passed are defined as follows: 
	     	> data format either both i/p & o/p are to passed or not
	     	> if data for i/p and o/p is in same or differet file through a ** boolean **  value
	     	> names of files that the data is located at
	     	> should prediction model be used or not 
	     	  --- prediction model divides the data into three parts, namely : 

	     	  		training data
	     	  		cross-validation data
	     	  		testing data for the hypothesis
	"""


	def __init__(self, bothInOut:bool = True, inOneFile:bool = True, dataFiles, usePredictionModel = False, defaultOPAt = 0):

		self.bothInOut = bothInOut
		self.inOneFile = inOneFile
		self.fileNames = dataFiles
		self.usePredictionModel = usePredictionModel
		self.defaultOPAt = defaultOPAt

		_load_data()


	def _load_data(self):

		if self.inOneFile  == True:
			data = pd.read_csv(self.fileNames)
			data = np.array(self.data)
			self.X = self.data[:, defaultOPAt+1:]
			self.Y = self.data[:, 0]

		else:

			# the function assumes that:
			# the first file is specified to contain the input values and
			# the second datafile is supposed to contain the corresponding output values

			self.X = pd.read_csv(self.fileNames[0])
			self.Y = pd.read_csv(self.fileNames[1]).resize((-1, 1))

			if(self.X.shape[0] != self.Y.shape[0]):
				self.X = self.Y = ""
				raise ValueError("mismatch between the shapes of the datavalues")		





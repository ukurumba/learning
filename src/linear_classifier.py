import random as rd
from math import exp 

class LC():
	def __init__.py(self,num_inputs,threshold):
		if threshold != 'perceptron' and threshold != 'logistic':
			raise ValueError('Threshold argument must be either perceptron or logistic.')
		if type(num_inputs)!=int:
			raise TypeError('Number of inputs must be an integer')
		if num_inputs <= 0:
			raise ValueError('Number of inputs must be greater than 0')
		weights = {-1:rd.uniform(0,0.5)} #random bias weight
		for i in range(num_inputs):	
			self.weights[i] = rd.uniform(0,0.5)
		self.threshold = threshold

	def propagate(self,xvals):
		#xvals is a 1-D list of the xvalues for a record
		wx = self.weights[-1]
		for i,val in enumerate(xvals):
			wx += self.weights[i] * val
		return self.threshold(wx)

	def threshold(self,wx):
		if self.threshold == 'perceptron':
			if wx <0:
				return 0
			elif wx >= 0:
				return 1
			else:
				raise ValueError('Something weird. wx should be a numeric')
		else:
			return 1 / (1 + exp(-wx))

	def train(self,xvals,yvals,alpha=.01):

		#xvals is a list of records, where each record is fed into the classifier
		#yvals is a 1-D list of the yvals. len(yvals) == len(xvals)
		self.alpha = alpha 
		for X,y in zip(xvals,yvals):
			a = self.propagate(X)
			if self.threshold == 'perceptron':
				delta = self.alpha * (y-a)
			else:
				delta = self.alpha * (y-a) * a * (1-a)
			for i,x_i in enumerate(X):
				self.weights[i] += delta * x_i
			self.weights[-1] += delta

	def predict(self,X):
		outputs = []
		for x in X:
			outputs.append(self.propagate(x))
		return outputs





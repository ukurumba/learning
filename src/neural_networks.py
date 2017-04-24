import collections
import random as rd
import copy
from math import exp

class Node():
	node_type = None
	def __init__(self, node_index, input_indices):
		# node_index is an integer, input_indices is an array
		if type(node_index) != int or type(input_indices) != list:
			raise TypeError('Please supply integer for node index and list for input indices')
		else:
			if input_indices == [-100]:
				self.node_type = 'input'
				self.index = node_index
			else:
				self.node_type = 'non-input'
				self.weights = collections.OrderedDict()
				self.weights[-1] = 1
				for i in input_indices:
					if i < 0:
						raise ValueError('Indices must be greater than or equal to 0')
					self.weights[i] = rd.random()
				self.inputs = input_indices
				self.index = node_index

	def get_weights(self):
		return self.weights

	def get_index(self):
		return self.index

	def get_inputs(self):
		return self.inputs

	def propagate(self,xvals):
		if self.node_type == 'input' : 
			if len(xvals) != 1:
				raise ValueError('Input nodes must only receive one input')
			return xvals[0]

		else:
			wx = self.weights[-1] #bias weight
			for key in xvals:
				print(self.index,wx)
				wx += xvals[key] * self.weights[key]
			print(self.index,wx)
		return 1 / (1 + exp(-wx))

	def set_weights(self,new_weights):
		for key,value in new_weights.items():
			self.weights[key] = value

	def get_weight(self,input_index):
		return self.weights[input_index]

	def set_weight(self,input_index,weight):
		self.weights[input_index] = weight

class NN():
	def __init__(self,num_inputs,num_outputs,num_hidden_layers,num_nodes_per_hidden_layer):
		self.layers = []
		self.layers.append({i:Node(i,[-100]) for i in range(num_inputs)})
		next_index = num_inputs
		inputs = list(self.layers[0].keys())
		self.alpha = .1

		for i in range(num_hidden_layers):
			temp_layer = {}
			for j in range(num_nodes_per_hidden_layer):
				temp_layer[next_index] = Node(next_index,inputs)
				next_index += 1

			self.layers.append(temp_layer)
			inputs = list(temp_layer.keys())
		self.layers.append({next_index+i:Node(next_index+i,inputs) for i in range(num_outputs)})
		self.outputkeys = list(self.layers[-1].keys())

	def get_layers(self):
		return self.layers

	def set_weights(self,layer_num,node_index,new_weights):
		self.layers[layer_num][node_index].set_weights(new_weights)


	def train_input_nodes(self,inputs):
		avals = {}
		for i in range(len(inputs)):
			avals[i] = inputs[i]
		return avals

	def train_hidden_nodes(self,avals):
		if len(self.layers) == 2:
			return avals
		else:
			prev_avals = copy.copy(avals)
			new_avals = {}
			for i in range(1,len(self.layers)-1,1):
				for node in self.layers[i].values():
					new_avals[node.get_index()] = node.propagate(prev_avals)
				for key,value in zip(new_avals.keys(),new_avals.values()):
					avals[key] = value
				prev_avals = copy.copy(new_avals)
		return avals

	def train_output_nodes(self,avals,yvals):
		#this fx returns the deltas from the output nodes
		deltas = {}
		random_output = rd.choice(list(self.layers[-1].values())) #just getting the input nodes to the output layer
		prev_avals = {i:avals[i] for i in random_output.get_inputs()}
		print('outputkeys',self.outputkeys)
		for i,key in zip(range(len(yvals)),self.outputkeys):
			avals[key] = self.layers[-1][key].propagate(prev_avals)
			deltas[key] = (yvals[i] - avals[key])*avals[key]*(1-avals[key])
			print('outputkey',key)
		return deltas,avals

	def propagate_deltas(self,deltas,avals):
		#this function computes the deltas for every node
		if len(self.layers) == 2:
			return deltas
		else:
			for l in range(len(self.layers)-2,0,-1): # for every hidden layer iterating backwards
				if l == len(self.layers)-2:  #if it's not the last hidden layer (for indexing purposes)
					outputs = self.outputkeys
					print('outputs',outputs)
				else:
					outputs = self.layers[l+2].values()[0].get_inputs() #gets inputs to a node 2 layers down
				print('outputs',outputs)
				for node in self.layers[l].values():
					deltas[node.get_index()] = 0
					print('deltas',deltas)
					for key in outputs: #adds the delta_k * weight_jk value for each output k
						print(key)
						deltas[node.get_index()] += deltas[key] * self.layers[l+1][key].get_weight(node.get_index())
					deltas[node.get_index()] *= avals[node.get_index()] * (1-avals[node.get_index()])
			print('deltas',deltas)

			return deltas

	def update_weights(self,deltas,avals):
		for l in range(1,len(self.layers)):
			for key,node in self.layers[l].items():
				for inkey in node.get_inputs():
					node.set_weight(inkey,node.get_weight(inkey) + self.alpha * avals[inkey] * deltas[key])
				node.set_weight(-1,node.get_weight(-1) + self.alpha * deltas[key])

	def train(self,xvals,yvals):
		''' This function trains the neural network on the given input data, utilizing the backtracking algorithm.

		Inputs
		------

		xvals : list of lists (each list is a set of xvalues of length equal to the number of input nodes)

		yvals : list of lists (each list is a set of yvals of length equal to the number of output nodes)

		Example
		-------

		x1 = [1,2]
		x2 = [2,2]
		y1 = [3]
		y2 = [4]
		x = [x1,x2]
		y = [y1,y2]
		nn = neural_network.NN(2,1,1,2)
		nn.train(x,y)

		Returns
		-------

		None '''

		for X,y in zip(xvals,yvals):
			avals = self.train_input_nodes(X)
			avals = self.train_hidden_nodes(avals)
			deltas,avals = self.train_output_nodes(avals,y)
			deltas = self.propagate_deltas(deltas,avals)
			self.update_weights(deltas,avals)







			






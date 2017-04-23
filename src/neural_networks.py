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

class NN():
	def __init__(self,num_inputs,num_outputs,num_hidden_layers,num_nodes_per_hidden_layer):
		self.layers = []
		self.layers.append({i:Node(i,[-100]) for i in range(num_inputs)})
		next_index = num_inputs
		inputs = list(self.layers[0].keys())

		for i in range(num_hidden_layers):
			temp_layer = {}
			for j in range(num_nodes_per_hidden_layer):
				temp_layer[next_index] = Node(next_index,inputs)
				next_index += 1

			self.layers.append(temp_layer)
			inputs = list(temp_layer.keys())
		self.layers.append({next_index+i:Node(next_index+i,inputs) for i in range(num_outputs)})

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
		random_output = rd.choice(list(self.layers[:-1].values()))
		prev_avals = {i:avals[i] for i in random_output.get_inputs()}
		for i,key in zip(range(len(yvals)),self.layers[:-1].keys()):
			avals[key] = self.layers[:-1][key].propagate(prev_avals)
			deltas[key] = (yvals[i] - avals[key])*avals[key]*(1-avals[key])
		return deltas,avals






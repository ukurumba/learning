import nose 
import sys
import src.neural_networks as nn
from math import exp

def test_node_constructor():
	inputs = [i for i in range(1,6,1)]
	n1 = nn.Node(1,inputs)
	assert n1.get_index() == 1
	assert len(n1.get_weights()) == len(inputs) + 1
	assert n1.get_weights()[5] <= 1.0
	assert n1.get_weights()[5] >= 0.0


def test_set_weights():
	inputs = [i for i in range(1,6,1)]
	n1 = nn.Node(1,inputs)
	new_weights = {1:.5,5:.5}
	n1.set_weights(new_weights);
	assert n1.get_weights()[1] == .5
	assert n1.get_weights()[5] == .5 

def test_node_propagtor():
	inputs = [i for i in range(5)]
	n1 = nn.Node(5,inputs)
	n2 = nn.Node(2,[-100])
	new_weights = {i:.3 for i in range(5)}
	n1.set_weights(new_weights);
	x_vals = {i:1 for i in range(5)}
	a = n1.propagate(x_vals);
	ainput = n2.propagate([3])
	assert abs(a - 1/(1+exp(-2.5))) <.000001
	assert ainput == 3

	
def test_nn_constructor():
	nn1 = nn.NN(4,4, 4,5);
	nn2 = nn.NN(1,1,0,0); 
	input1 = nn1.get_layers()[0]
	hidden1 = nn1.get_layers()[1]
	out = nn1.get_layers()[-1]
	input2 = nn2.get_layers()[0]
	total = nn1.get_layers()
	for layer in total:
		for i in layer.keys():
			print(i)

	assert len(input1) == 4
	assert len(hidden1) == 5
	assert len(out) == 4
	assert len(input2) == 1
	assert hidden1[4].get_index() == 4
	assert input1[3].get_index() == 3
	assert out[27].get_index() ==27

def test_train_input():
	nn1 = nn.NN(5,1,1,2)
	x1 = [i for i in range(1,6,1)]
	avals = nn1.train_input_nodes(x1)
	assert avals[0] == 1
	assert avals[4] == 5

def test_train_hidden():
	nn1 = nn.NN(5,1,2,2)
	x1 = [1 for i in range(1,6,1)]
	weights = {i:0.3 for i in range(0,5,1)}
	weights7 = {5:0.4,6:0.4}
	avals = nn1.train_input_nodes(x1)
	nn1.set_weights(1,5,weights)
	nn1.set_weights(1,6,weights)
	nn1.set_weights(2,7,weights7)
	avals = nn1.train_hidden_nodes(avals)
	assert avals[5] == 1 / (1 + exp(-(1+.3*5)))
	assert avals[6] == 1 / (1 + exp(-(1+.3*5)))
	assert avals[7] == 1 / (1 + exp(-(1+.8 * avals[6])))



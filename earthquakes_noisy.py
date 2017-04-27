import src.linear_classifier as lc 
import src.neural_networks as nn 
import random as rd
import sys
with open('./examples/earthquake-noisy.data.txt','r') as file:
	data = []
	for line in file:
		data.append(line.rstrip('\n'))
data = [data[i].split(',') for i in range(len(data))]
data = data[:-2] #blanks
rd.shuffle(data)
data = nn.kfold(data,k=6)
totalacc = 0
for i in range(len(data)):
	tune = data[i]
	train = [j for j in data if j != i]
	train = nn.combine(train)
	trainnames = [float(j[2]) for j in train]
	train = [[float(j) for j in i[0:2]] for i in train]
	tunenames = [float(j[2]) for j in tune ]
	tune = [[float(j) for j in i[0:2]] for i in tune]
	lc1 = lc.LC(2,sys.argv[1])
	for i in range(100):
		lc1.train(train,trainnames)
	outputs = lc1.predict(tune)
	if sys.argv[1] == 'logistic':
		outputs = lc.log_threshold(outputs)
	acc = nn.accuracy(outputs,tunenames)
	print('Acc:',acc)
	totalacc+=acc
totalacc/=len(data)
print('Total Acc:',totalacc)

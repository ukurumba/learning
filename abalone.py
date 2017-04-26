import src.neural_networks as nn 
import numpy as np 
import random as rd 

with open('./examples/banknote_authentication.txt','r') as file:
	data = []
	for line in file:
		data.append(line.rstrip('\n'))


data = [data[i].split(',') for i in range(len(data))]
rd.shuffle(data)


data = nn.kfold(data,k=10)
totalacc = 0
for i in range(len(data)):
	tune = data[i]
	train = [j for j in data if j != i]
	train = nn.combine(train)
	trainages = [float(i[4]) for i in train]
	train = [[float(j) for j in i[0:4]] for i in train]

	tuneages = [float(i[4]) for i in tune]
	tune = [[float(j) for j in i[0:4]] for i in tune]

	for i, datum in enumerate(trainages):
		if datum == 1.0:
			trainages[i] = [1,0]
		elif datum == 0.0:
			trainages[i] = [0,1]
		else:
			raise ValueError('Improper Label')
	for i, datum in enumerate(tuneages):
		if datum == 1.0:
			tuneages[i] = [1,0]
		elif datum == 0.0:
			tuneages[i] = [0,1]

	nn1 = nn.NN(4,2,0,0)
	for i in range(100):
		nn1.train(train,trainages)
	outputs = nn1.predict(tune)
	outputs = nn.threshold(outputs)
	acc = nn.accuracy(outputs,tuneages)
	print('Accuracy',acc)
	totalacc += acc
totalacc /= len(data)
print('Average Accuracy:', totalacc)
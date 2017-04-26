import src.neural_networks as nn 
import numpy as np 
import random as rd 
import time

with open('./examples/banknote_authentication.txt','r') as file:
	data = []
	for line in file:
		data.append(line.rstrip('\n'))


data = [data[i].split(',') for i in range(len(data))]
rd.shuffle(data)
data = nn.kfold(data,k=10)
totaltime=0
totalacc=0
for l in range(len(data)):
	tune = data[l]
	train = [j for j in data if j != l]
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
	start = time.time()
	for i in range(100):
		nn1.train(train,trainages)
	end = time.time()
	totaltime += end-start
	outputs = nn1.predict(tune)
	outputs = nn.threshold(outputs)
	acc = nn.accuracy(outputs,tuneages)
	print('Fold no.',l+1)
	print('Accuracy: ',acc)
	print('Time:',end-start)
	totalacc += acc
totalacc /= len(data)
totaltime /= len(data)
print('Average Accuracy:', totalacc)
print('Average Time:' , totaltime)

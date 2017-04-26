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
accs = []
times = []
data = nn.kfold(data,k=10)
for l in [1,10,20,50,70,100,200,400,1000,2000,3000,5000]:

	totalacc = 0
	totaltime = 0
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
		start = time.time()
		for i in range(l):
			nn1.train(train,trainages)
		end = time.time()
		totaltime += end-start
		outputs = nn1.predict(tune)
		outputs = nn.threshold(outputs)
		acc = nn.accuracy(outputs,tuneages)
		print('Accuracy',acc)
		totalacc += acc
	totalacc /= len(data)
	totaltime /= len(data)
	print('Average Accuracy:', totalacc)
	print('Average Time:' , totaltime)
	accs.append(totalacc)
	times.append(totaltime)

print(accs,times)
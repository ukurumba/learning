import src.neural_networks as nn 
import src.linear_classifier as lc
import sys
import numpy as np 
import random as rd 
import time

with open('./examples/banknote_authentication.txt','r') as file:
	data = []
	for line in file:
		data.append(line.rstrip('\n'))


data = [data[i].split(',') for i in range(len(data))]
rd.shuffle(data)
totaltime = 0
totalacc = 0
data = nn.kfold(data,k=10)
for i in range(len(data)):
	tune = data[i]
	train = [j for j in data if j != i]
	train = nn.combine(train)
	trainages = [float(i[4]) for i in train]
	train = [[float(j) for j in i[0:4]] for i in train]

	tuneages = [float(i[4]) for i in tune]
	tune = [[float(j) for j in i[0:4]] for i in tune]

	lc1 = lc.LC(4,sys.argv[1])
	start = time.time()
	for i in range(100):
		lc1.train(train,trainages)
	end = time.time()
	totaltime += end-start
	outputs = lc1.predict(tune)
	if sys.argv[1] == 'logistic':
		outputs = nn.threshold([[i] for i in outputs])
		outputs = [i[0] for i in outputs]
	acc = nn.accuracy(outputs,tuneages)
	print('Accuracy',acc)
	totalacc += acc
totalacc /= len(data)
totaltime /= len(data)
print('Average Accuracy:', totalacc)
print('Average Time:' , totaltime)

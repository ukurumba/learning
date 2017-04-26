import src.neural_networks as nn 
import numpy as np 
import random as rd
import time
with open('./examples/iris.txt' ,'r') as file:
	data = []
	for line in file:
		data.append(line[:-1])
	data = data[:-1] #get rid of end newline

data = [data[i].split(',') for i in range(len(data))]
rd.shuffle(data)
ksplit = nn.kfold(data,k=6)

totalacc = 0
totaltime = 0
data = ksplit
for l,k in enumerate(ksplit):
	tune = data[l]
	train = [j for j in data if j != l]
	train = nn.combine(train)
	trainnames = [i[4] for i in train]
	train = [[float(j) for j in i[0:4]] for i in train]

	tunenames = [i[4] for i in tune]
	tune = [[float(j) for j in i[0:4]] for i in tune]
	traintargets = []
	tunetargets = []
	for name in trainnames:
		if name == 'Iris-versicolor':
			traintargets.append([1,0,0])
		elif name == 'Iris-virginica':
			traintargets.append([0,1,0])
		elif name == 'Iris-setosa':
			traintargets.append([0,0,1])
	for name in tunenames:
		if name == 'Iris-versicolor':
			tunetargets.append([1,0,0])
		elif name == 'Iris-virginica':
			tunetargets.append([0,1,0])
		elif name == 'Iris-setosa':
			tunetargets.append([0,0,1])
	nn1 = nn.NN(4,3,1,7)
	start = time.time()
	for i in range(100):
		nn1.train(train,traintargets)
	end = time.time()
	totaltime += end-start
	outputs = nn1.predict(tune)
	outputs = nn.threshold(outputs)
	acc = nn.accuracy(outputs,tunetargets)
	print('Fold No.:',l+1)
	print('Accuracy:',acc)
	print('Time:',end-start)
	totalacc += acc
totalacc /= len(ksplit)
totaltime /= len(data)
print('Average Accuracy:', totalacc)
print('Average Time: ',totaltime)






# #select first 50 for tuning, last 100 for training

# indices = [i for i in range(150)]
# random_indices = np.random.choice(indices,50,replace=False)
# tune_data = []
# train_data = []
# tune_targets = []
# train_targets = []
# for index in range(150):
# 	if index in random_indices:
# 		tune_data.append(inputs[index])
# 		tune_targets.append(targets[index])
# 	else:
# 		train_data.append(inputs[index])
# 		train_targets.append(targets[index])

# print(train_targets)
# print('Shape Tune',len(tune_data),'First element',len(tune_data[0]))
# print('Shape train',len(train_data),'First element',len(train_data[0]))

# nn1 = nn.NN(4,3,1,7)
# for i in range(100):
# 	nn1.train(train_data,train_targets)

# outputs = nn1.predict(tune_data)
# print(outputs)
# outputs = nn.threshold(outputs)
# acc = nn.accuracy(outputs,tune_targets)
# print('Accuracy',acc)







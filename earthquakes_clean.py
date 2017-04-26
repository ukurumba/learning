import src.linear_classifier as lc 
import src.neural_networks as nn 

with open('./examples/earthquake-clean.data','r') as file:
	data = []
	for line in file:
		data.append(line.rstrip('\n'))
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
	lc1 = LC(2,'perceptron')
	for i in range(10):
		lc1.train(train,trainnames)
	outputs = lc1.predict(tune)
	acc = nn.accuracy(outputs,tunenames)
	print('Acc:',acc)
	totalacc+=acc
totalacc/=len(data)
print('Total Acc:',totalacc)







print(data) 
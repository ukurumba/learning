import src.neural_networks as nn 

with open('./examples/iris.txt' ,'r') as file:
	data = []
	for line in file:
		data.append(line[:-1])
	data = data[:-1] #get rid of end newline

data = [data[i].split(',') for i in range(len(data))]
inputs = [[float(j) for j in i[0:4]] for i in data]
names = [i[4] for i in data]
targets = []

for name in names:
	if name == 'Iris-versicolor':
		targets.append([1,0,0])
	elif name == 'Iris-virginica':
		targets.append([0,1,0])
	elif name == 'Iris-setosa':
		targets.append([0,0,1])

#select first 50 for tuning, last 100 for training

tune_data = inputs[0:50]
train_data = inputs[50:]
tune_targets = targets[0:50]
train_targets = targets[50:]

nn1 = nn.NN(4,2,4,3)
nn1.train(train_data,train_targets)







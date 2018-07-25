import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

def load_file(filename):
	data_source=open(filename,'r')
	dataset=[]
	for line in data_source:
		row=[]
		features=line.strip('\n').split(',')#list
		for f in features:
			row.append(float(f))
		dataset.append(row)
	data_source.close()

    #initialization
	arr=np.array(dataset[0])
	#row
	for j in range(1,len(dataset)):
		arr=np.vstack((arr,np.array(dataset[j])))
	#normaliztion
	#column
	max_items=arr.max(axis=0)
	for col in range(1,len(dataset[0])):
		arr[:,col]=arr[:,col]/max_items[col]

	return arr
def create_trainset(data):
#select 50% data to be training data
	trainset=[]
	trainclass=[]
	for p in range(0,np.size(data,0)):#np.size(data,0)=#row
		if p %2==0:
			trainset.append(data[p,1:])
			trainclass.append(data[p,0])
	trainset=np.asarray(trainset)
	temp_class=[]
	for i in range(len(trainclass)):
		temp=[]
		if trainclass[i]==1:
			temp=[1.0,0.0,0.0]
		elif trainclass[i]==2:
			temp=[0.0,1.0,0.0]
		else:
			temp=[0.0,0.0,1.0]
		temp_class.append(temp)
	trainclass=np.asarray(temp_class)
	return trainset,trainclass
    
def create_novelset(data):
#select 50% data to be novel data
	novelset=[]
	novelclass=[]
	for p in range(0,np.size(data,0)):#np.size(data,0)=#row
		if p %2==1:
			novelset.append(data[p,1:])
			novelclass.append(data[p,0])
	novelset=np.asarray(novelset)
	temp_class=[]
	for i in range(len(novelclass)):
		temp=[]
		if novelclass[i]==1:
			temp=[1.0,0.0,0.0]
		elif novelclass[i]==2:
			temp=[0.0,1.0,0.0]
		else:
			temp=[0.0,0.0,1.0]
		temp_class.append(temp)
	novelclass=np.asarray(temp_class)

	return novelset,novelclass

start_time=time.time()

# Make results reproducible
seed=1234
np.random.seed(seed)
tf.set_random_seed(seed)

filename='wine.data.txt'
dataset=load_file(filename)
X_train,y_train=create_trainset(dataset)
X_test,y_test=create_novelset(dataset)

# Session
sess=tf.Session()

# Interval / Epochs
interval=50
epoch=500

# Initialize placeholders
X_data=tf.placeholder(shape=[None, 13], dtype=tf.float32)
y_target=tf.placeholder(shape=[None, 3], dtype=tf.float32)

# Input neurons : 13
# Hidden neurons : 8
# Output neurons : 3
hidden_layer_nodes=8

# Create variables for Neural Network layers
w1=tf.Variable(tf.random_normal(shape=[13,hidden_layer_nodes])) # Inputs -> Hidden Layer
b1=tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))   # First Bias
w2=tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,3])) # Hidden layer -> Outputs
b2=tf.Variable(tf.random_normal(shape=[3]))   # Second Bias

# Operations
hidden_output=tf.nn.relu(tf.add(tf.matmul(X_data, w1), b1))
final_output=tf.nn.softmax(tf.add(tf.matmul(hidden_output, w2), b2))

# Cost Function
loss=tf.reduce_mean(-tf.reduce_sum(y_target * tf.log(final_output), axis=0))

# Optimizer
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# Initialize variables
init=tf.global_variables_initializer()
sess.run(init)

xlist=[]
los=[]
# Training
print('Training the model...')
for i in range(1, (epoch + 1)):
    sess.run(optimizer, feed_dict={X_data: X_train, y_target: y_train})
    if i % interval == 0:
        print('Epoch', i, '|', 'Loss:', sess.run(loss, feed_dict={X_data: X_train, y_target: y_train}))
        xlist.append(i)
        los.append(sess.run(loss, feed_dict={X_data: X_train, y_target: y_train}))
        

# Prediction
ac=len(X_test)
for i in range(len(X_test)):
    if not (np.array_equal(y_test[i],((np.rint(sess.run(final_output, feed_dict={X_data: [X_test[i]]})))[0]))) :
        #print('wrong')
        ac-=1
cost=time.time()-start_time
print('-'*30)
print('novel set accuracy: '+str(ac/len(X_test)*100)+'%')
print('total cost: '+str(cost))


xray=np.asarray(xlist)
yloss=np.asarray(los)

plt.plot(xray,yloss,'r--')
plt.show()

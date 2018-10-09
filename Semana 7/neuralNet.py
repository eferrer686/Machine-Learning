import numpy as np
import tensorflow as tf
#Pandas
import pandas as pd
#Sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

#Read File
filename = 'trainClean.csv'
titanic = np.genfromtxt(filename,delimiter=',',dtype=float)
#Eliminar Headers
titanic = np.delete(titanic, (0), axis=0)


# seed random
seed = 42
tf.set_random_seed(seed)
np.random.seed(seed)


# Datos a utilizar como input
x = np.array([x[2:7] for x in titanic])
#Sobrevivio?
y = np.array([x[1] for x in titanic])

#Crear Test y Train Arrays
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)


# normalization
mms = MinMaxScaler()
x_train = mms.fit_transform(x_train)
x_test = mms.fit_transform(x_test)


# placeholder
x_data = tf.placeholder(shape=[None, 5], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)


# parameters
W1 = tf.Variable(tf.random_normal(shape=[5, 6]))
b1 = tf.Variable(tf.random_normal(shape=[6]))
hidden_1 = tf.nn.relu(tf.add(tf.matmul(x_data, W1), b1))


W2 = tf.Variable(tf.random_normal(shape=[6, 4]))
b2 = tf.Variable(tf.random_normal(shape=[4]))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W2), b2))


W3 = tf.Variable(tf.random_normal(shape=[4, 1]))
b3 = tf.Variable(tf.random_normal(shape=[1]))
output = tf.nn.relu(tf.add(tf.matmul(hidden_2, W3), b3))

# loss
loss = tf.reduce_mean(tf.square(y_target - output))


# optimize
optimizer = tf.train.GradientDescentOptimizer(0.005)
train_step = optimizer.minimize(loss)

#placeholder for tensorBoard
placeholder1 = tf.placeholder(tf.float32, name='uno')
sum1 = tf.summary.scalar('uno',placeholder1)




batchSize = 50
with tf.Session() as sess:
   
    # initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)

    train_loss = []
    test_loss = []
    
    for i in range(30):

        random_index = np.random.choice(len(x_train), size= batchSize)
        
        random_x = x_train[random_index]
        random_y = np.transpose([y_train[random_index]])
        
        sess.run(train_step, feed_dict={x_data:random_x,y_target: random_y})
        
        tempTrainLoss = sess.run(loss, feed_dict={x_data: random_x,y_target: random_y})
        
        tempTestLoss = sess.run(loss, feed_dict={x_data: x_test,y_target: np.transpose([y_test])})
        
        print(i, str([tempTrainLoss, tempTestLoss]))
        
        train_loss.append(sess.run(tf.sqrt(tempTrainLoss)))
        test_loss.append(sess.run(tf.sqrt(tempTestLoss)))  
    
    #SessGraph
    writer = tf.summary.FileWriter('./graphs', sess.graph)

    #summaryTrain = tf.summary.scalar('TrainScalar', train_loss)
    #summaryTest = tf.summary.scalar('TestScalar', test_loss)
    exsummary = sess.run(sum1, placeholder1, feed_dict={tempTrainLoss})

    # merge summaries
    merged = tf.summary.merge_all()
    summary = sess.run(merged)
    #writer
    writer.add_summary(summary)
    writer.flush()       


    
 
        


    



plt.plot(train_loss, 'k-', label='train loss')
plt.plot(test_loss, 'r--', label='test loss')
plt.legend(loc='upper right')
plt.show()


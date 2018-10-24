import tensorflow as tf

#placeholder for input 4
x_data = tf.placeholder(shape=[None,4],dtype=tf.float32)
y_data = tf.placeholder(shape=[None,3],dtype=tf.float32)
#Placeholder for input layer 4
W1 = tf.Variable(tf.random_normal(shape=[4,4]))
b1 = tf.Variable(tf.random_normal(shape=[4]))
a1 = tf.nn.relu(tf.add(tf.matmul(x_data,W1),b1))
hidden_1 = tf.sigmoid(a1)
#Placeholder for hidden layer 5
W2 = tf.Variable(tf.random_normal(shape=[4,5]))
b2 = tf.Variable(tf.random_normal(shape=[5]))
a2 = tf.nn.relu(tf.add(tf.matmul(hidden_1,W2),b2))
hidden_2 = tf.sigmoid(a2)
#Placeholder for hidden layer 6
W3 = tf.Variable(tf.random_normal(shape=[5,6]))
b3 = tf.Variable(tf.random_normal(shape=[6]))
a3 = tf.nn.relu(tf.add(tf.matmul(hidden_2,W3),b3))
hidden_3 = tf.sigmoid(a3)
#Placeholder for hidden layer 4
W4 = tf.Variable(tf.random_normal(shape=[6,4]))
b4 = tf.Variable(tf.random_normal(shape=[4]))
a4 = tf.nn.relu(tf.add(tf.matmul(hidden_3,W4),b4))
hidden_4 = tf.sigmoid(a4)
#Placeholder for hidden layer 3
W5 = tf.Variable(tf.random_normal(shape=[4,3]))
b5 = tf.Variable(tf.random_normal(shape=[3]))
a5 = tf.nn.relu(tf.add(tf.matmul(hidden_4,W5),b5))
hidden_5 = tf.sigmoid(a5)
#Placeholder for output layer 3
W6 = tf.Variable(tf.random_normal(shape=[3,3]))
b6 = tf.Variable(tf.random_normal(shape=[3]))
a6 = tf.nn.relu(tf.add(tf.matmul(hidden_5,W6),b6))
output = tf.sigmoid(a6)

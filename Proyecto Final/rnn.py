import tensorflow as tf

#Setup of RNN
class Brain(object):
    def __init__(self,nodesInput,nodesHidden,nodesOutput):
        self.inputs = tf.placeholder(shape=[None,1],dtype=tf.float32)

        self.W1 = tf.Variable(tf.random_normal(shape=[nodesInput,nodesHidden]))
        self.b1 = tf.Variable(tf.random_normal(shape=[nodesHidden]))
        self.hidden_1 = tf.nn.sigmoid(tf.add(tf.matmul(self.inputs,self.W1),self.b1))

        self.W2 = tf.Variable(tf.random_normal(shape=[nodesHidden,nodesOutput]))
        self.b2 = tf.Variable(tf.random_normal(shape=[nodesOutput]))
        self.output = tf.nn.sigmoid(tf.add(tf.matmul(self.hidden_1,self.W2),self.b2))

        self.sess = tf.Session()

        #Inicializar variables y pesos
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        self.mutateTensor = tf.placeholder(dtype=tf.float32)
        self.rW = tf.Variable(tf.random_uniform(tf.TensorShape(self.mutateTensor),0,0.5),validate_shape=False)
        self.newW = tf.multiply(self.mutateTensor,self.rW)
        self.mutate = tf.assign(self.mutateTensor,self.newW)

    def predict(self,input):
        predict = self.sess.run(self.output,feed_dict={self.inputs:input})
        return predict

    def mutate(self,randomRate = 0.5):
        rW1 = tf.Variable(tf.random_uniform(self.W1.shape,0,0.5))
        self.W1 = tf.multiply(self.W1,rW1)

        rW2 = tf.Variable(tf.random_uniform(self.W2.shape,0,0.5))
        self.W2 = tf.multiply(self.W2,rW2)
        
        rB1 = tf.Variable(tf.random_uniform(self.b1.shape,0,0.5))
        self.b1 = tf.multiply(self.b1,rB1)
        
        rB2 = tf.Variable(tf.random_uniform(self.b2.shape,0,0.5))
        self.b2 = tf.multiply(self.b2,rB2)

        self.sess.run(mutate,feed_dict={mutateTensor: self.W1})


#Establecer # de nodos en las capas
brain = Brain(1,5,1)

#Prediccion sin ajuste
array = brain.predict([[0.5]])
print(array)

#Prediccion sin ajuste, debe ser el mismo valor puesto que no se inicializa otra vez los pesos
array = brain.predict([[0.5]])
print(array)

#Mutar pesos
brain.mutate()

#Prediccion con nuevos pesos, debe ser diferente puesto que los pesos son modificados
array = brain.predict([[0.5]])
print(array)


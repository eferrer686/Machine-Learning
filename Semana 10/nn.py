
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
get_ipython().run_line_magic('matplotlib', 'inline')
print("Using TensorFlow Version %s" %tf.__version__)


# In[2]:


#Creación de los datos usando make_moons
np.random.seed(0)
X, Y = make_moons(500, noise=0.2)

# División de los datos usando train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=73)


# # Revisa el valor de las siguientes variables:

# In[3]:


# Definición de la estructura de la red:
nInputRows = X_train.shape[0] # cuantos renglones
nInputsColumns = X_train.shape[1] #cuantas columnas

# Capa
nHiddenNeurons = 4 # número de neuronas
nOutputs = 1 # número de salidas, binario


# In[4]:


class NeuronalNet(object):
    
    def __init__(self, sess, X, Y, n_hidden=4, learning_rate=1e-2):
        self.sess = sess #Sesión para correr TensorFlow
        self.X = X # Entradas
        self.Y = Y.reshape(-1,1) #Salidas
        self.n_inputs = X.shape[0] # Renglones
        self.n_input_dim = X.shape[1] #Columnas
        self.n_output = 1 # Como es binario, sólo UNA salida
        self.learning_rate = learning_rate 
        '''
        La tasa de aprendizaje es un hiper-parámetro que controla 
        cuánto estamos ajustando los pesos de nuestra red con respecto 
        al gradiente de pérdida. Cuanto más bajo sea el valor, 
        más lento viajamos a lo largo de la pendiente descendente. 
        Si bien esto podría ser una buena idea (usar una tasa de aprendizaje baja) 
        para asegurarnos de no perder ningún mínimo local, 
        también podría significar que nos llevará mucho tiempo converger, 
        especialmente si nos quedamos atascados. 
        '''
        self.n_hidden = n_hidden # número de neuronas 
        
        # Create NeuronalNet
        self.X_input, self.y, self.logits, self.cost = self.createNeuralNet()
        
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate) #optimizador

        self.train_op = self.optimizer.minimize(self.cost)#operación de entreanmiento


    #Metodo de construcción de la red:
    def createNeuralNet(self):
        # Investiga qué es el xavier_initializer???? Reporta en el código:
        initializer = tf.contrib.layers.xavier_initializer()

        # Agrega el placeholder para X_input, entradas
        X_input = tf.placeholder(tf.float32, [None, self.n_input_dim], name = 'input')
        # Agrega el placeholder para y, salidas
        y = tf.placeholder(tf.float32, [None, self.n_output], name = 'output')

        # Agrega una capa fully_connected de X_input * n_hidden, con una activación (activation_fn) tf.nn.elu, inicializa los pesos weights_initializer
        #Reporta qué es una función de activación tf.nn.elu
        hidden1 = fully_connected(X_input, self.n_hidden, activation_fn=tf.nn.elu, weights_initializer = initializer)
        # Agrega una capa fully_connected de hidden1 * n_output , con una activación (activation_fn) tf.nn.sigmoid, inicializa los pesos weights_initializer
        logits = fully_connected(hidden1, self.n_output, activation_fn=tf.nn.elu, weights_initializer = initializer)
        #calcula el error con tf.nn.sigmoid_cross_entropy_with_logits
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
        #calcula el costo con tf.reduce_mean
        cost = tf.reduce_mean(loss)
        
        #Regresa X_input, y, logits, costo
        return X_input, y, logits, cost
    
    #funcion de entrenamiento 
    def train(self):
    	#ejecuta el entrenamiento debes correr train_op, cost y en el diccionario manda X_input, X , y, Y
        _, cost = self.sess.run([self.train_op, self.cost], feed_dict={self.X_input: self.X, self.y: self.Y})        
        return cost
    
	#funcion de predicción
    #NO modificar, dudas con el código?
    def predict(self, X_test):
        pred = self.sess.run([self.logits], feed_dict={ self.X_input: X_test})[0]        
        return pred
    
    #función de graficación, NO modificar
    #Fuente: http://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html#sphx-glr-auto-examples-neural-networks-plot-mlp-alpha-py
    def plot_decision_boundary(self):
        x_min, x_max = self.X[:, 0].min()-0.1, self.X[:, 0].max()+0.1
        y_min, y_max = self.X[:, 1].min()-0.1, self.X[:, 1].max()+0.1
        spacing = min(x_max - x_min, y_max - y_min) / 100
        XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing),
                       np.arange(y_min, y_max, spacing))
        data = np.hstack((XX.ravel().reshape(-1,1), 
                          YY.ravel().reshape(-1,1)))
        db_prob = self.predict(data)
        clf = np.where(db_prob<0.5,0,1)
        Z = clf.reshape(XX.shape)
        plt.figure(figsize=(10,8))
        plt.contourf(XX, YY, Z, cmap=plt.cm.Spectral, alpha=0.8)
        plt.scatter(X[:,0], X[:,1], c=Y, 
                    cmap=plt.cm.Spectral)
        plt.show()


# In[5]:


#NO MODIFICAR
# función para obtener el costo del modelo al ejecutarlo n número de iteraciones
def getTrainCost(model, n_iters=1000):
    model.sess.run(tf.global_variables_initializer())
    cost = []
    for i in range(n_iters):
        cost.append(model.train())
    return cost


# In[6]:


#NO MODIFICAR
n_iters = 10000
print("%d Neuronas" % 1)
tf.reset_default_graph() #BORRA EL GRAFO ANTERIOR EN LA SESIÓN
sess = tf.Session() #SESIÓN DE TENSORFLOW
net = NeuronalNet(sess, X_train, Y_train, n_hidden=50)

cost = getTrainCost(net, n_iters) #COSTO DEL MODELO
pred_prob = net.predict(X_test) #PROBABILIDADES DE LA PREDICCIÓN
y_hat = np.where(pred_prob<0.5,0,1) #YHAT

precision = np.sum(Y_test.reshape(-1,1)==y_hat) / len(Y_test) #CALCULAR LA PRECISIÓN
print("Precisión%.2f" % precision)
net.plot_decision_boundary() #imprimir los datos


# # Completa la práctica con 5, 10, 25, 50, 100 neuronas en una sola capa

# # Completa la práctica con 3, 5, 10 capas de 10 neuronas cada capa

import keras
from keras.layers import Dense
from keras.initializers import glorot_normal
from keras import Sequential

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
def plot_decision_boundary(modelo,X,Y):
        x_min, x_max = X[:, 0].min()-0.1, X[:, 0].max()+0.1
        y_min, y_max = X[:, 1].min()-0.1, X[:, 1].max()+0.1
        spacing = min(x_max - x_min, y_max - y_min) / 100
        XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing),
                       np.arange(y_min, y_max, spacing))
        data = np.hstack((XX.ravel().reshape(-1,1), 
                          YY.ravel().reshape(-1,1)))
        db_prob = modelo.predict(data)
        clf = np.where(db_prob<0.5,0,1)
        Z = clf.reshape(XX.shape)
        plt.figure(figsize=(10,8))
        plt.contourf(XX, YY, Z, cmap=plt.cm.Spectral, alpha=0.8)
        plt.scatter(X[:,0], X[:,1], c=Y, 
                    cmap=plt.cm.Spectral)
        plt.show()




#Creación de los datos usando make_moons
np.random.seed(0)
X, Y = make_moons(500, noise=0.2)

# División de los datos usando train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=73)


out_dimension = 1
nHidden = 2
nInputDimensions = X_train.shape[1]
print(X_train.shape)

modelo = Sequential()
#Input Layers
modelo.add(Dense(nHidden,
        input_dim=nInputDimensions,
        activation='elu',
        kernel_initializer = glorot_normal(seed=0)))
#Hidden Layers
modelo.add(Dense(50,
        activation='elu',
        kernel_initializer = glorot_normal(seed=0)))
#Output Layer
modelo.add(Dense(out_dimension,
        activation='sigmoid',
        kernel_initializer = glorot_normal(seed=0)))

modelo.compile(loss = 'binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

modelo.fit(X_train,
        Y_train,
        epochs=100)

modelo.evaluate(X_test,Y_test)

plot_decision_boundary(modelo,X,Y)

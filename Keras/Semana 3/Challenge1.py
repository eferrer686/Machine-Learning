
# coding: utf-8

# In[1]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import RandomNormal

from keras.initializers import glorot_normal
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)


# In[2]:


bikes = numpy.loadtxt("hour.csv", delimiter=",")
# split into input (X) and output (Y) variables

bikesTrain = bikes[:,:]

bikesTest = bikes[10000:,:]

X = bikesTrain[0:,:6]
Y = bikesTrain[0:,6]

XTest = bikesTest[0:,0:5]
YTest = bikesTest[0:,6]


# In[3]:


for cont in range(5):
    X[:,cont] = (X[:,cont]/(X[:,cont].max()))
    
    
Y = Y/Y.max()


# In[16]:


# create model

model = Sequential()
model.add(Dense(6, input_dim=6, activation='elu',kernel_initializer = 'normal'))
model.add(Dense(50, activation='elu',kernel_initializer = 'normal'))
model.add(Dense(1, activation='sigmoid',kernel_initializer = 'normal'))


# In[17]:


# Compile model
model.compile(loss='poisson', optimizer='adam', metrics=['accuracy'])


# In[18]:


# Fit the model
model.fit(X,
          Y,
          epochs=100)


# In[7]:


print(model.predict(X))


# In[8]:


print(Y)


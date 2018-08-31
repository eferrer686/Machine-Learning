
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


bikes = pd.read_csv("hour.csv")
bikes


# In[3]:


bikes[:].plot(x="dteday",y="cnt")


# In[4]:


from numpy import genfromtxt
bikes = genfromtxt('hour.csv', delimiter=',')

# np.random.shuffle(bikes)

X = bikes[1:,2:16]
Y = bikes[1:,16:17]

print(X)


# In[5]:


#X = np.array(([[4.5,6.7,8.9],[9.1,7.6,6.5],[8.7,6.5,7.0]]),dtype=float)
#Y = np.array(([[89,90],[80,70],[69,89]]),dtype=float)

Xn = X


for cont in range(14):
    Xn[:,cont] = (Xn[:,cont]/(Xn[:,cont].max()))
    
print(Xn)


# In[6]:


Yn = Y/Y.max()


# In[7]:


print("Xn: ")
print(len(Xn))


# In[19]:


class NeuralNetwork():
    def __init__(self,inputs,outputs,hidden):
        self.inputs = inputs
        self.outputs = outputs
        self.hidden = hidden
        self.W1 = np.random.randn(self.inputs,self.hidden)
        self.W2 = np.random.randn(self.hidden,self.outputs)
        self.error = []

    def sigmoid(self,Z):
        return 1 / (1+np.exp(-Z))

    def feedforward(self,X):
        self.Z2 = X @ self.W1
#         print("Z2")
#         print(self.Z2)
        self.a2 = self.sigmoid(self.Z2)
#         print("a2")
#         print(self.a2)
        self.Z3 = self.a2 @ self.W2
#         print("Z3")
#         print(self.Z3)
        self.yhat = self.sigmoid(self.Z3)

#         print("yhat: ")
#         print(self.yhat)

        return self.yhat

    def derSigmoid(self, Z):
        return np.exp(-Z)/((1+np.exp(-Z))**2)
    
    
    def cost (self,X,y):
        self.yhat=self.feedforward(X)
        Costo = 0.5*sum((y-self.yhat)**2)
        return Costo

    def costDer(self,X,y):
        #Predicciones
        self.yhat = self.feedforward(X)

        #delta3 = diferecia de y - yhat *derSigmoid(Z3)
        self.delta3 = np.multiply(-(y-self.yhat), self.derSigmoid(self.Z3))
      
        self.a2 = self.a2[:, np.newaxis]
        self.delta3 = self.delta3[:, np.newaxis]
        
        #djW2 = transpuesta de a2 * delta3
        djW2 = self.a2 @ self.delta3 

        #delta2 = delta3 * (transpuesta de W2) * derSigmoid(Z2)

        self.delta2 = self.delta3  @ self.W2.T * self.derSigmoid(self.Z2)
        
        
        
        
        X = X[:, np.newaxis]
        #djW1 = transpuesta de X * delta2

        djW1 = X @ self.delta2

        return djW1,djW2
    
    def backwardPropagation(self,Xn,Yn,alpha):
    
#         self.feedforward(Xn)

        djW1, djW2 = self.costDer(Xn,Yn)
        #alpha=0.01
        self.W1 = self.W1 + alpha*djW1
        self.W2 = self.W2 + alpha*djW2
        error = nn.cost(Xn,Yn)

        return error
    
    def trainBP(self,Xn,Yn,alpha,expError):
#         while error>expError:
#         error = 10000000
        
        row = 0
        ef = False
        while row<len(Xn):
            
            er = self.backwardPropagation(Xn[row],Yn[row],alpha)
            
            error = self.getError(er)
            if error>expError:
                ef = False
            else:
                ef = True
            
            sys.stdout.write("\rRow #%r - Error : %f - Ef: %d" % (row , error, ef))
            sys.stdout.flush()
            
            row=row+1
            
        return error
        
            
    def getError(self,er):
        self.error.extend([er])
        return sum(self.error) / float(len(self.error))


# In[20]:


nn = NeuralNetwork(14,1,6)


# In[ ]:


expError = .02
error = 10000
alpha = 0.5

while error > expError:
    nn.error = []
    error = nn.trainBP(Xn,Yn,alpha,expError)
    


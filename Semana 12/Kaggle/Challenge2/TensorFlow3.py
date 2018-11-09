# Import keras modules

import numpy as np
import tensorflow as tf

import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
validate = pd.read_csv('gender_submission.csv')


# # Drop features-... is this good?
train = train.drop(['Name'],axis=1)
test = test.drop(['Name'],axis=1)

# # Sex mapping 1 - male, 0 - female
train['Sex_binary']=train['Sex'].map({'male':1,'female':0})
test['Sex_binary']=test['Sex'].map({'male':1,'female':0})


# # Age cleaning
train['Age'] = train['Age'].fillna(train['Age'].mean())
test['Age'] = test['Age'].fillna(train['Age'].mean())


features = ['Pclass','Age','Sex_binary','SibSp','Parch','Fare']
target = 'Survived'


X_train = np.array(train[features])

y_train = np.array(train[target])
y_train = y_train.reshape(-1,1)

model = tf.keras.models.Sequential([
   tf.keras.layers.Dense(12, activation=tf.nn.relu),
  tf.keras.layers.Dense(6, activation=tf.nn.relu),
  tf.keras.layers.Dense(6, activation=tf.nn.relu),
  tf.keras.layers.Dense(6, activation=tf.nn.relu),
  tf.keras.layers.Dense(6, activation=tf.nn.relu),
  tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=1000)


#Predecir 
X_test = np.array(test[features])
predicciones = model.predict(X_test)


predicciones = predicciones.tolist()

pre = pd.Series(predicciones)
validate['prediccion'] = pre
validate['prediccion'] = validate['prediccion'].str.get(0)


coincidencias = []
for dato in validate.prediccion:
    if dato >= 0.5:
        coincidencias.append(1)
    else:
        coincidencias.append(0)
validate['final'] = coincidencias


coincide = 0
coincide = sum(validate['Survived'] == validate['final'])
print(coincide/ len(validate))


match = 0
nomatch = 0
for val in validate.values:
    if val[1] == val[3]:
        match = match +1
    else:
        nomatch = nomatch +1
print(match/len(validate))



toKaggle = pd.DataFrame({'PassengerId':validate['PassengerId'],
                         'Survived':validate['final']})

toKaggle.head()


# # output file with your prediction
from datetime import datetime

archivo = 'Titanic10-3'+'.csv'

toKaggle.to_csv(archivo,index=False)

print('Creado: ' + archivo)


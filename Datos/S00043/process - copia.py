import os

#Get Local Path
raw=""
localPath = str(os.getcwd())

print(localPath)
#Find data ending in txt
for file in os.listdir(localPath):
    if file.endswith(".txt"):
        print(os.path.join(localPath, file))
        raw = os.path.join(localPath, file)

#Read File
lines = open(raw, 'r')

newLines = open("temp.csv","w+")
#Quitar lineas sin datos o "Head"
cont = 0
for line in lines:
    if cont>5:
        newLines.write(line)
    cont += 1

#Resto del codigo

import pandas as pd 
from pandas import DataFrame as df
datos = pd.read_csv("temp.csv", sep='\t', lineterminator='\r')

#Encontrar inicio y fin de datos de Nike
init = 0
count = 0 
finish = 0
stimulusNameFlag = True
for p in datos[datos.columns[5]]:
    if str(p) == "Nike__What_are_girls_made_of_video2m" and stimulusNameFlag:
        print(count)
        init = count
        stimulusNameFlag = False
    elif str(p) == "Nike__What_are_girls_made_of_video2m" or str(p)==" ":
        finish = count
    count+=1
print(finish-init)

print((init,finish))
header=["FrameNo","StimulusName","Frustration Evidence"] #Seleccionamos columnas
datos=datos[init:finish] #Datos de Nike
datos.to_csv('datos2.csv', columns=header, index=False) #Creamos el video que queramos entrenar


datos2 = pd.read_csv("datos2.csv") 
dicc = [] #creamos dicc
for i in range (0,finish-init): #Uno mas que el numero de frames a limpiar //250frames=2:06 //500 frames=10:00 // 1775
    cont=0 #contador para el promedio
    res=0 #suma acumalada para el promedio
    avg=0 #promedio de vector por frame
    row=0 #variable para filas
    found = True #bandera
    frame0 = False
    
    while found:
        print(datos2.shape[0])
        if row < datos2.shape[0]:#si el renglon es menor al numero de filas en datos2
            print(datos2["FrameNo"].iloc[row])
            if 0 == datos2["FrameNo"].iloc[row]:
                frame0 = True
            
            if i == datos2["FrameNo"].iloc[row] and frame0: #si el valor i es igual al de renglon
                cont = cont + 1 #sumamos el contador
                res = res+datos2["Frustration Evidence"].iloc[row] #acumulamos res
            elif i < datos2["FrameNo"].iloc[row] or row == datos2.shape[0] - 1: #checamos si acabo con los frames
                found = False #apagamos bandera
            row = row + 1 
        else:
            found = False #apagamos bandera
    if cont > 0: 
        avg=res/cont #sacamos promedio

        dicc.append({"FrameNo": i, "FrustrationEvidence": avg}) #append al dixionario con llave frame y value sens
        print ("Guardando datos del frame: ",i)
df = pd.DataFrame.from_dict(dicc) #pasamos a df
df.to_csv("datos3.csv", index= False, sep=',') #guardamos como csv


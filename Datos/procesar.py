import pandas as pd 
from pandas import DataFrame as df
datos = pd.read_csv('Dump042_S00013.csv') #lee datos crudos
header=["FrameNo","StimulusName","Frustration Evidence"] #Seleccionamos columnas
datos=datos[22506:39465] #Datos de Nike
datos.to_csv('datos2.csv', columns=header, index=False) #Creamos el video que queramos entrenar
datos2 = pd.read_csv("datos2.csv") 
dicc = [] #creamos dicc
for i in range (0,1776): #Uno mas que el numero de frames a limpiar //250frames=2:06 //500 frames=10:00 // 1775
    cont=0 #contador para el promedio
    res=0 #suma acumalada para el promedio
    avg=0 #promedio de vector por frame
    row=0 #variable para filas
    found = True #bandera
    while found:
        if row < datos2.shape[0]:#si el renglon es menor al numero de filas en datos2
            if i == datos2["FrameNo"].iloc[row]: #si el valor i es igual al de renglon
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

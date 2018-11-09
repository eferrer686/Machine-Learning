import cv2
import numpy as np
import os
import sys
from PIL import Image

# Get Local Path
localPath = os.getcwd()
# Find data ending in wmv
for root, dirs, files in os.walk(localPath):
    for name in files:
        if name.endswith("wmv"):
            raw = os.path.join(localPath, name)
            print(raw)



# set video file path of input video with name and extension
vid = cv2.VideoCapture(raw)

if not os.path.exists('images'):
    os.makedirs('images')

# for frame identity
index = 0
while(True):
    # Extract images
    ret, frame = vid.read()
    # end of frames
    if not ret:
        break
    # Saves images
    name = './images/frame' + str(index) + '.jpg'
    print('Creating...' + name)
    cv2.imwrite(name, frame)
    # next frame
    index += 1

print("Finished taking frames")



def crop(im):
    # load the image and show it
    area=(250,80,385,240)
    cropped=im.crop(area)

i = 0


for f in os.listdir('./images'):
    if f.endswith('.jpg'):
        f = str("./images/"+f)
        im = Image.open(f)
        print('Dimensiones de las imagenes')
        print(im.size)
        break

xBegin = int(input("X Inicio: "))
yBegin = int(input("Y Inicio: "))
xFinish = int(input("X Fin: "))
yFinish = int(input("Y Fin: "))


for f in os.listdir('./images'):
    if f.endswith('.jpg'):
        f = str("./images/"+f)
        im = Image.open(f)
        print("Cropping frame no [",str(i),"]")
        area=(xBegin,yBegin,xFinish,yFinish)
        cropped=im.crop(area)
        if not os.path.exists("./cropped"):
            os.makedirs("./cropped")
        cropped.save(("./cropped/Cropped #" +str(i)+".jpg" ))
        i=i+1
print("All frames cropped")
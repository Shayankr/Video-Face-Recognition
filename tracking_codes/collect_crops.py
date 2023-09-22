
import json
import sys
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%matplotlib inline
import random
import os
import cv2

args = sys.argv
#folder = args[1].split('.')[0]
video=args[1]
folder = video.split('.')[0]
boxes=folder+'/'+args[2]
file_o = open(boxes,"r")
os.makedirs(folder+"/crops")
lines = file_o.readlines()
frame_count = int (lines[0])
det_count = [0 for i in range(600)]
crops_folder = folder+"/crops"
max_ = 0
cap = cv2.VideoCapture(video)
w=int (cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
h=int (cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
resolution = (w, h) #(1920, 1080) 
for i in range(len(lines)-1):
    w=lines[i+1].split()
    n_dets = int (w[1])
    f = int (w[0])
    
    cap.set(1,f)  # Where frame_no is the frame you want
    ret, frame = cap.read()  # Read the frame
    cv2_img = cv2.resize(frame, resolution)
    #print (f"frame {f}")
    for j in range (n_dets):
        det_id = int (w[2+(j*6)])
        #print (det_id)
        left= int (w[3+(j*6)]); bottom= int (w[4+(j*6)]); 
        right= int (w[5+(j*6)]); top= int (w[6+(j*6)]); 
        filename = crops_folder+'/'+w[2+(j*6)]+'/'+str(det_count[det_id])+'.jpg'
        #img = cv2.rectangle(img, (left,bottom),(right, top), color[det_id], 4) 
        if det_count[det_id]==0:
          os.mkdir(crops_folder+'/'+w[2+(j*6)])
          #print ('created 27')
        to_file = cv2_img[bottom:top, left:right]
        cv2.imwrite(filename, to_file)
        det_count[det_id]+=1
        max_ = max (max_, det_id)
    #cv2.imshow("window", img)
    #cv2.waitKey(150)
file_o.close()
print (f"total {max_} identities detected & collected into folders")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:52:55 2023

@author: kiranchaitanya
"""

import sys
from random import randint
import cv2

args = sys.argv
color = []
for i in range(500):
    color.append((randint(0, 128), randint(0, 255), randint(0, 200)))
video=args[1]
folder = video.split('.')[0]
boxes=folder+'/'+args[2]
cap = cv2.VideoCapture(video)
w=int (cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
h=int (cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
resolution = (w, h) #(1920, 1080) 
#resolution = (1280, 676) #(1920, 1080)
file_o = open(boxes,"r")
lines = file_o.readlines()
frame_count = int (lines[0])
out = cv2.VideoWriter(folder+'/output.avi',cv2.VideoWriter_fourcc(*'DIVX'), 5, resolution)
 


for i in range(frame_count):
    w=lines[i+1].split()
    n_dets = int (w[1])
    f = int (w[0])
    cap.set(1,f)  # Where frame_no is the frame you want
    ret, frame = cap.read()  # Read the frame
    img = cv2.resize(frame, resolution)
    for j in range (n_dets):
        det_id = int (w[2+(j*6)])
        left= int (w[3+(j*6)]); bottom= int (w[4+(j*6)]); 
        right= int (w[5+(j*6)]); top= int (w[6+(j*6)]); 
        #filename = folder+'/'+w[2+(j*6)]+'/'+str(det_count[det_id])+'.jpg'
        img = cv2.rectangle(img, (left,bottom),(right, top), color[det_id], 4)
    #cv2.imshow("window", img)
    #cv2.waitKey(150)
    out.write(img)
    
file_o.close()
cap.release()
out.release()
#cv2.waitKey(150)
cv2.destroyWindow("window")
cv2.destroyAllWindows()

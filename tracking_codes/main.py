#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 20:05:32 2023

@author: kiranchaitanya
"""

import sys
import subprocess
import os

from retinaface import RetinaFace
import numpy as np
import pandas as pd
import cv2
from random import randint
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
import yaml
from absl import logging
import math
import cv2
import os
import requests
import errno
import json 

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Input,
)
from tensorflow.keras.applications import (
    MobileNetV2,
    ResNet50
)

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))

def _regularizer(weights_decay=5e-4):
    return tf.keras.regularizers.l2(weights_decay)


def Backbone(backbone_type='ResNet50', use_pretrain=True):
    """Backbone Model"""
    weights = None
    if use_pretrain:
        weights = 'imagenet'

    def backbone(x_in):
        if backbone_type == 'ResNet50':
            return ResNet50(input_shape=x_in.shape[1:], include_top=False,
                            weights=weights)(x_in)
        elif backbone_type == 'MobileNetV2':
            return MobileNetV2(input_shape=x_in.shape[1:], include_top=False,
                               weights=weights)(x_in)
        else:
            raise TypeError('backbone_type error!')
    return backbone


def OutputLayer(embd_shape, w_decay=5e-4, name='OutputLayer'):
    """Output Later"""
    def output_layer(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = BatchNormalization()(x)
        x = Dropout(rate=0.5)(x)
        x = Flatten()(x)
        x = Dense(embd_shape, kernel_regularizer=_regularizer(w_decay))(x)
        x = BatchNormalization()(x)
        return Model(inputs, x, name=name)(x_in)
    return output_layer


def ArcHead(num_classes, margin=0.5, logist_scale=64, name='ArcHead'):
    """Arc Head"""
    def arc_head(x_in, y_in):
        x = inputs1 = Input(x_in.shape[1:])
        y = Input(y_in.shape[1:])
        x = ArcMarginPenaltyLogists(num_classes=num_classes,
                                    margin=margin,
                                    logist_scale=logist_scale)(x, y)
        return Model((inputs1, y), x, name=name)((x_in, y_in))
    return arc_head


def NormHead(num_classes, w_decay=5e-4, name='NormHead'):
    """Norm Head"""
    def norm_head(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = Dense(num_classes, kernel_regularizer=_regularizer(w_decay))(x)
        return Model(inputs, x, name=name)(x_in)
    return norm_head


def ArcFaceModel(size=None, channels=3, num_classes=None, name='arcface_model',
                 margin=0.5, logist_scale=64, embd_shape=512,
                 head_type='ArcHead', backbone_type='ResNet50',
                 w_decay=5e-4, use_pretrain=True, training=False):
    """Arc Face Model"""
    x = inputs = Input([size, size, channels], name='input_image')

    x = Backbone(backbone_type=backbone_type, use_pretrain=use_pretrain)(x)

    embds = OutputLayer(embd_shape, w_decay=w_decay)(x)

    if training:
        assert num_classes is not None
        labels = Input([], name='label')
        if head_type == 'ArcHead':
            logist = ArcHead(num_classes=num_classes, margin=margin,
                             logist_scale=logist_scale)(embds, labels)
        else:
            logist = NormHead(num_classes=num_classes, w_decay=w_decay)(embds)
        return Model((inputs, labels), logist, name=name)
    else:
        return Model(inputs, embds, name=name)

def set_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices(
                    'GPU')
                logging.info(
                    "Detect {} Physical GPUs, {} Logical GPUs.".format(
                        len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logging.info(e)


def load_yaml(load_path):
    """load yaml file"""
    with open(load_path, 'r') as f:
        loaded = yaml.load(f, Loader=yaml.Loader)

    return loaded


def get_ckpt_inf(ckpt_path, steps_per_epoch):
    """get ckpt information"""
    split_list = ckpt_path.split('e_')[-1].split('_b_')
    epochs = int(split_list[0])
    batchs = int(split_list[-1].split('.ckpt')[0])
    steps = (epochs - 1) * steps_per_epoch + batchs

    return epochs, steps + 1


def l2_norm(x, axis=1):
    """l2 norm"""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    output = x / norm

    return output

class BatchNormalization(tf.keras.layers.BatchNormalization):
    """Make trainable=False freeze BN for real (the og version is sad).
       ref: https://github.com/zzh8829/yolov3-tf2
    """
    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


class ArcMarginPenaltyLogists(tf.keras.layers.Layer):
    """ArcMarginPenaltyLogists"""
    def __init__(self, num_classes, margin=0.5, logist_scale=64, **kwargs):
        super(ArcMarginPenaltyLogists, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.logist_scale = logist_scale

    def build(self, input_shape):
        self.w = self.add_variable(
            "weights", shape=[int(input_shape[-1]), self.num_classes])
        self.cos_m = tf.identity(math.cos(self.margin), name='cos_m')
        self.sin_m = tf.identity(math.sin(self.margin), name='sin_m')
        self.th = tf.identity(math.cos(math.pi - self.margin), name='th')
        self.mm = tf.multiply(self.sin_m, self.margin, name='mm')

    def call(self, embds, labels):
        normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')

        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t')
        sin_t = tf.sqrt(1. - cos_t ** 2, name='sin_t')

        cos_mt = tf.subtract(
            cos_t * self.cos_m, sin_t * self.sin_m, name='cos_mt')

        cos_mt = tf.where(cos_t > self.th, cos_mt, cos_t - self.mm)

        mask = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes,
                          name='one_hot_mask')

        logists = tf.where(mask == 1., cos_mt, cos_t)
        logists = tf.multiply(logists, self.logist_scale, 'arcface_logist')

        return logists

class ArcFace():
    def __init__(self, model_path = None):
        if model_path == None:
            try:
                from astropy.utils.data import download_file
            except ImportError:
                raise ImportError("Please install astropy (pip install astropy) if you want to use the pre-trained ArcFace network.")
            tflite_path = download_file("https://www.digidow.eu/f/datasets/arcface-tensorflowlite/model.tflite", cache=True)
        else:
            tflite_path = model_path

        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def calc_emb(self, imgs):
        """Calculates the embedding from an (array of) images. These images can be cv2-image-files or a path to a file.
        Parameters:
            imgs (str|list): either a list of images or the image itself. The image can be a cv2-image or a path to a file. The images should already be aligned!
        Returns:
            ndarray: 512-d embedding of the supplied image(s)
        Example:
            calc_emb("~/Downloads/test.jpg")
        """
        if isinstance(imgs, list):
            return self._calc_emb_list(imgs)
        return self._calc_emb_single(imgs)


    def get_distance_embeddings(self, emb1, emb2):
        """Calculates the distance (L2 norm) between two embeddings. Larger values imply more confidence that the two embeddings are from different people.
        Parameters:
            emb1 (ndarray): embedding of a person (e.g. generated by calc_emb(...))
            emb2 (ndarray): embedding of a person (e.g. generated by calc_emb(...))
        Returns:
            int: distance between emb1 and emb2
        Example:
            get_distance_embeddings(calc_emb("person1.jpg"), calc_emb("person2.jpg"))
        """
        diff = np.subtract(emb1, emb2)
        dist = np.sum(np.square(diff))
        return dist
        
    def _calc_emb_list(self, imgs):
        embs = []
        for img in imgs:
            embs.append(self._calc_emb_single(img))
        return embs

    def _calc_emb_single(self, img):
        if isinstance(img, str):
            img = cv2.imread(img)
            if img is None:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), img)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112))
        img = img.astype(np.float32) / 255.
        if len(img.shape) == 3:
            img = np.expand_dims(img, 0)
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        emb = l2_norm(output_data)
        return emb[0]

def get_avg_emb (lst):
  x=lst[0]
  n = len(lst)
  for i in range(n-1):
    x=x+lst[i+1]
  return x/np.linalg.norm(x)



class ID:
    def update_emb (self, emb):
      #self.imgs.append (cv_img)
      self.emb = self.emb + emb
      self.embs_lst.append (emb)
    def update_state (self, state):
      self.state = state
    def __init__(self, i, cv_img):
      self.id = i
      face_rec = ArcFace()
      self.embs_lst = []
      self.emb = face_rec.calc_emb (cv_img)
      self.embs_lst.append (self.emb)
      self.state = 0
      self.update_state (1)
    def get_id (self):
      return self.id
    #def get_templates (self):
      #return self.imgs
    def get_emb (self):
      return self.emb/np.linalg.norm(self.emb)
    def get_embs_lst (self):
      return self.embs_lst 


if __name__ == "__main__":
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'retina-face'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'PyYAML'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'astropy'])
    args = sys.argv
    video=args[1]
    #'demo.mp4'
    #os.system ('submain.py')
    
    #print(video)
    # process each frame, detect all faces in each frame, write them to boxes
    
    folder = video.split('.')[0]
    
    
    
    cap = cv2.VideoCapture(video)
    print (args[2])
    if args[2]=='': frames_start = 1
    else: frames_start = int (args[2])
    if args[3]=='': frames_end = int (cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else: frames_end = int (args[3]) 
    frames=np.arange(frames_start, frames_end, 1)
    w=int (cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
    h=int (cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
    resolution = (w, h) #(1920, 1080) 
    
    IDs=[]
    idcount = []
    os.makedirs(folder)
    file_b = open(folder+"/boxes.txt","w")
    file_b.write(str(len(frames)))  # writing the number of frames to text file
    file_b.write('\n')
    file_b.close()
    s=''
    
    for f in frames:
        cap.set(1,f)  # Where frame_no is the frame you want
        ret, frame = cap.read()  # Read the frame
        frame = cv2.resize(frame, resolution)
        #img_name = folder+'/'+str(f)+'.jpg'
        #cv2.imwrite(img_name, frame)
        #cv2_imshow(frame)
        
        if f==frames_start or f%5==0: 
            print (f"processing frame {f}/{frames_end}")
        #path = folder+'/'+str(f)+'.jpg'
        dets = RetinaFace.detect_faces(frame)
        #img =  cv2.imread(frame)
        
        if len(dets)==0:  # no detections recorded in current frame
            s = str(f)+' '+str(len(dets))
            file_b = open(folder+"/boxes.txt","a")
            file_b.write(s)
            file_b.write('\n')
            file_b.close()
            continue
        
        if len(IDs)==0: # no IDs recorded so far, but detections available, add all of them as new IDs
            #print (f"frame {f}, detections {len(dets)}")
            #print (str(len(dets)))
            s = str(f)+' '+str(len(dets))
            for i in range (len(dets)): # iterate through detections
                key = 'face_'+str(i+1)
                face = dets[key]
                face_area = face['facial_area']
                face_area_img = frame[face_area[1]:face_area[3], face_area[0]:face_area[2]] #cv2 crop of face
                #face_rec = ArcFace()
                #emb = face_rec.calc_emb (face_area_img)
                emb_id = len(IDs)+1
                id = ID (emb_id, face_area_img)
  
                IDs.append (id)
                #idcount.append(1)
                s = s+f" {emb_id} {face_area[0]} {face_area[1]} {face_area[2]} {face_area[3]} {round(face['score'],6)}"
            file_b = open(folder+"/boxes.txt","a")
            file_b.write(s)
            file_b.write('\n')
            file_b.close()
        else:
            s = str(f)+' '+str(len(dets))
            embs=[] # to collect the corresponding embeddings
            maps = [] # tuples of (detection id, face id)
            scores=[] # corresponding scores
    
            ID_set = set([k+1 for k in range(len(IDs))]) 
            det_set = set([k+1 for k in range(len(dets))])
            for i in range (len(dets)): #process each detection
                key = 'face_'+str(i+1)
                face = dets[key]
                face_area = face['facial_area']
                face_area_img = frame[face_area[1]:face_area[3], face_area[0]:face_area[2]]
                #cv_imgs.append (face_area_img)
                face_rec = ArcFace()
                emb = face_rec.calc_emb (face_area_img)
                embs.append(emb)
                for j in range (len(IDs)): # calculate similarity between the current detection & all existing
                    face_emb = IDs[j].get_emb ()
                    score = cosine_similarity(emb, face_emb)
                    if (score>0.65):
                        #print (f'{i+1}, {j+1}, {score}\n')
                        scores.append (score)
                        maps.append ((i+1, IDs[j].get_id()))
            sorted_index = np.argsort (-1*np.array(scores))
            sorted_dets = np.array(maps)[sorted_index]
            for l in sorted_dets:
                if (l[1] in ID_set) and (l[0] in det_set):
                    ID_set.remove (l[1])
                    det_set.remove (l[0])
                    IDs[l[1]-1].update_emb (embs[l[0]-1]) # update template images of the matchd ID
                    key = 'face_'+str(l[0]) # write that matched detection to string
                    face = dets[key]
                    face_area = face['facial_area']
                    s = s+f" {l[1]} {face_area[0]} {face_area[1]} {face_area[2]} {face_area[3]} {round(face['score'],6)}"
            for d in det_set: # detections without match
                key = 'face_'+str(d)
                face = dets[key]
                face_area = face['facial_area']
                face_area_img = frame[face_area[1]:face_area[3], face_area[0]:face_area[2]]
                emb_id = len(IDs)+1
                id = ID (emb_id, face_area_img)
                IDs.append (id)
                s = s+f" {emb_id} {face_area[0]} {face_area[1]} {face_area[2]} {face_area[3]} {round(face['score'],6)}"
                
            file_b = open(folder+"/boxes.txt","a")
            file_b.write(s)
            file_b.write('\n')
            file_b.close()
    
    cap.release()
    cv2.destroyAllWindows()
    embs={}
    for id in IDs:
        lst=[]
        for emb in id.get_embs_lst():
            lst.append(emb.tolist())
            embs[id.get_id()]=lst
    with open(folder+"/embs.json", "w") as outfile:
        json.dump(embs, outfile)
# coding: utf-8
"""
Created on Sat Apr 22 13:43:53 2019

@author: wqf
"""
 
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import statistics
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from train import BalloonConfig,CLASS_NAMES

#def apply_mask(mask, color, alpha=0.5):
#    """Apply the given mask to the image.
#    """
#    h= mask.shape[0]
#    w= mask.shape[1]
#    image = np.zeros((h,w,3), np.uint8)
#    for c in range(3):
#        image[:, :, c] = np.where(mask == 1,
#                                  image[:, :, c] *
#                                  (1 - alpha) + alpha * color[c] * 255,
#                                  image[:, :, c])
#    return image




def get_mask_center(mask):
        print(mask.shape)
        h= mask.shape[0]
        w= mask.shape[1]
        img = np.zeros((h,w,3), np.uint8)
        mat = [[col, row] for col in range(w) for row in range(h) if mask[row, col] != 0] 
        mat = np.array(mat).astype(np.float32)#have to convert type for PCA 
        #mean (e. g. the geometrical center) 
        #and eigenvectors (e. g. directions of principal components) 
        m, e = cv2.PCACompute(mat, mean = np.array([])) 
        #now to draw: let's scale our primary axis by 100, 
        #and the secondary by 50 
        center = tuple(m[0]) 
        endpoint1 = tuple(m[0] + e[0]*100) 
        endpoint2 = tuple(m[0] + e[1]*50) 
        print('center:',center)
        print(endpoint1)
        print(endpoint2)
        alpha = math.atan2((center[1]-endpoint1[1]), (endpoint1[0]-center[0]))
        alpha = 180*alpha/math.pi


        store = []
        print(np.any(mask, axis=0).shape)
        horizontal_indicies = np.where(np.any(mask, axis=0))[0]
        print(horizontal_indicies.shape)
        vertical_indicies = np.where(np.any(mask, axis=1))[0]
        print(vertical_indicies.shape)
        w = len(horizontal_indicies)
        #print(w)
        h = len(vertical_indicies)

        #print(h)
        #print(len(horizontal_indicies))
        if w > h:
            h_index = len(horizontal_indicies) / 2 
            print(int(h_index))
            print(mask[int(h_index),:])
            for i in range(len(mask[int(h_index),:])):
                if mask[int(h_index),i] == True:
                    store.append(i)
            v_index = store[int(len(store)/2)]
            x, y = v_index, h_index
            return int(x), int(y), int(alpha)
        else:
            v_index = len(vertical_indicies) / 2 
            print(int(v_index))
            print(mask[int(v_index),:])
            for i in range(len(mask[int(v_index),:])):
                if mask[int(v_index),i] == True:
                    store.append(i)
            h_index = store[int(len(store)/2)]
            x, y = v_index, h_index
            return int(x), int(y), int(alpha)
                    
#test from here
EVENT_FOLDER='balloon20190420T1809'
MODEL_FILE='mask_rcnn_balloon_0021.h5'   
TEST_IMGAGE='./images/14.jpg'

MODEL_DIR = os.path.join("logs",EVENT_FOLDER)

COCO_MODEL_PATH =os.path.join(MODEL_DIR,MODEL_FILE)

config = BalloonConfig()
config.display()
 

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
 

model.load_weights(COCO_MODEL_PATH, by_name=True)

 
class_names = CLASS_NAMES

image = skimage.io.imread(TEST_IMGAGE)
cv2.imshow('image', image)
print(image.shape)
results = model.detect([image], verbose=1)

r = results[0]
m = r['masks']
print(r['masks'])
print(r['masks'].shape)
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],  class_names, r['scores'])

x,y,alpha = get_mask_center(r['masks'])
print('x:',x,'y:',y,'angle:', alpha)

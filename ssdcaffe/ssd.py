'''
Created on Oct 12, 2017

@author: inayat
'''

from __future__ import print_function
from imutils import paths
import numpy as np
import argparse
import imutils
import random
import caffe
import cv2
import datetime


class SSD(object):
    '''
    
    This class implements the MobileNetSSD detector using sdd-caffe and
    opencv
    ref: https://github.com/weiliu89/caffe/tree/ssd
    
    
    
    '''
    
    


    def __init__(self, prototxt, caffemodel):
        '''
        
        The initialize the caffe net with preset input parameters and paths
        '''
        
        # initialize the list of class labels MobileNet SSD was trained to
        # detect, then generate a set of bounding box colors for each class
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"]
        
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
        
        #self.PROTOTXT = "MobileNetSSD_deploy.prototxt.txt"
        self.PROTOTXT = prototxt
        
        #self.CAFFEMODEL = "MobileNetSSD_deploy.caffemodel"
        self.CAFFEMODEL = caffemodel

        self.USEGPU = True
        
        self.NOGPUs = 1
        
        self.VERBOSE = True
        
        if self.USEGPU :
            self._debug("using GPU")
            caffe.set_device(0)
            caffe.set_mode_gpu()
            
        else:
            self._debug("Using CPU Only")
        
        self._debug("Loading caffe model:  " + self.CAFFEMODEL )
        
        self.net = caffe.Net(self.PROTOTXT, self.CAFFEMODEL,caffe.TEST)

        
    def preProcess(self, imgSrc):
        
        img = imgSrc.astype(np.float32)
        img = cv2.resize(img, (300,300))
        
        
        img = img - 127.5
        img = img * 0.007843
        
        img = img.transpose((2, 0, 1))
        
        return img
    
    def postProcess(self, img, out):
        
        (h, w) = img.shape[:2]
        bbox = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])
        
        clslbl= out['detection_out'][0,0,:,1]
        
        conf = out['detection_out'][0,0,:,2]
        
        return (bbox.astype(np.int32), conf, clslbl)
    
    
    def detect(self, imgOrig):
        
        img = self.preProcess(imgOrig)
        
        outPut = []
        
        
        #img = img.transpose((2, 0, 1))
        
        
        self.net.blobs['data'].data[...] = img
        
        
        outPut = self.net.forward()
        
        return outPut
        #(bbox, conf, clslbl) = self.postProcess(imgOrig, outPut)
        
        #return (bbox, conf, clslbl)
    
    def drawDetections(self, imgOrig, bbox, conf, clslbl):
        
        
        bbox = bbox.astype(np.int32)
        for i in range(len(bbox)):
            
            if(conf[i] > 0.20):
                
                startPt = (bbox[i][0], bbox[i][1])
                endPt = (bbox[i][2], bbox[i][3])
                
                cv2.rectangle(imgOrig, startPt, endPt, self.COLORS[int(clslbl[i])])
                
                p3 = (max(startPt[0], 15), max(startPt[1], 15))
                
                title = "%s:%.2f" % (self.CLASSES[int(clslbl[i])], conf[i]*100)
                cv2.putText(imgOrig, title, p3, cv2.FONT_ITALIC, 0.6, self.COLORS[int(clslbl[i])], 2)
        
        return imgOrig
        
          
    def _debug(self, msg, msgType="[INFO]"):
        
        '''
           the _debug  method, which can be used to (optionally) write debugging messages
        '''
        # check to see if the message should be printed
        if self.VERBOSE:
            print("**** {} {} - {}".format('\033[92m' + msgType + '\033[0m', msg, datetime.datetime.now()))
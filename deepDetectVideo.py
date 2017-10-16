'''
Created on Oct 12, 2017

@author: inayat
@email: inayatkh@gmail.com
'''


# import the required  packages
from imutils.video import FileVideoStream
#from imutils.video import FPS
import numpy as np
import argparse
import time
import cv2
import imutils

from  ssdcaffe import SSD 

from utils.fps2 import FPS2

#from imutils.video import FPS


if __name__ == '__main__':
    
    
    # Initialize the argument parse which is used to parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True,
                    help="path to input video file")
    args = vars(ap.parse_args())
    
       
       
    ssd = SSD("./deepModel/MobileNetSSD_deploy.prototxt.txt",
               "./deepModel/MobileNetSSD_deploy.caffemodel")
    
   
   
    dispWin="Real-Time Deep Object Detection using MobileNetSSD"
    
    ssd._debug(" starting to read a video file ...")
    fvs = FileVideoStream(args["video"]).start()
    time.sleep(1.0)
    
    # start the frame per second  (FPS) counter
    fps = FPS2().start() 
    #fps = FPS().start() 
    
    cv2.namedWindow(dispWin, cv2.WINDOW_NORMAL)
    # loop over the frames obtained from the video file stream
    skindex=1;
    while fvs.more():
        
        
        # grab each frame from the threaded video file stream,
        frame = fvs.read()
        
        if(skindex % 2) == 0:
            
            frame = imutils.resize(frame, width=640)
            #frame=cv2.resize(frame, (300,300))
            
            outPut = ssd.detect(frame)
            
            (bbox, conf, clslbl) = ssd.postProcess(frame, outPut)
            
            frame = ssd.drawDetections(frame, bbox, conf, clslbl)
            
            cv2.putText(frame, "FPS: {:.2f}  frame sz={}x{}".format(fps.fps(), frame.shape[1], frame.shape[0]),
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # show the frame and update the FPS counter
            cv2.imshow(dispWin, frame)
        
        
        
        
        
        #fps.update()
        #cv2.putText(frame, "FPS: {:.2f}".format(fps.fps()),
        #            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        
        
        key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        fps.update()
        
        '''
        if(skindex % 5) == 0:
            cv2.putText(frame, "FPS: {:.2f}".format(fps.fps()),
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # show the frame and update the FPS counter
            cv2.imshow(dispWin, frame)
        '''
        skindex+=1
        
    # stop the timer and display FPS information
    fps.stop()
    ssd._debug(" elasped time: {:.2f}".format(fps.elapsed()))
    ssd._debug(" approx. FPS: {:.2f}".format(fps.fps()))
    
    # do a bit of cleanup
    cv2.destroyAllWindows()
    fvs.stop()

# Real-time Object Detection with deep MobileNetSSD detector

## [MobileNetSSD](https://arxiv.org/abs/1704.04861)

This is an efficient use  of a deep detector usinf python 3, [Caffe](https://github.com/weiliu89/caffe/tree/ssd) and openCV.

In order to achieve a very high frame rate (FPS) , we have used VideoStream class of [imutils](https://github.com/jrosebr1/imutils) for python 3.

A class name SSD has been implemented for loading the MobileNetSSD model using Caffe, and manipulating other required pre and post processing.

To run this code, follow these steps

- Step 1: 
	Using pip install these packages
	- imtuils
	- caffe
	- opencv ver 3.2
- Step 2:
   Run the detector
   $ pyton3 deepDetectVideo -v ./videos//Kids-Truck-Bus-video.mkv


- you can find a  sample  in the vdieos folder obtained from [youtube](https://www.youtube.com/watch?v=CFlBtMD2Qis)
   
  
##  An example of the output is shown bellow
![ ](./deepObjDetect.gif  "MobileNetSSD detector")

you can watch the full video [here](https://youtu.be/s1v9iZ4BAaM)

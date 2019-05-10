# Maskrcnn
For testing performance on mask rcnn. just run test.py (if not missing logs folder)
training.py is used for training. The whole training time is about 2 hours.
The mrcnn folder contains the configuration, model and utils files. model.py is for building CNN architecture. configuration.py is for setting up parameters. utils.py is for several image processing calculations, such as calculate iou, overlapping region, non max suppression and so on. (It also needs to notice here the files in this folder, I partially rewrote and edited based on the open source frame work from the implementation by matterport and the mask rcnn paper by Kaiming He. And I keep their name in this folder's  python file.)
The .h5 file in logs/ballon folder is the trained CNN weight used for testing. If you are using github to open this, probably there is no logs folder since the github would not let users to upload the file larger than 200MB. You can email me to get this. 
The dataset folder contains the training and testing tools data.
Images folder contains some new images for seeing the performance. 


# Mask rcnn Grasping 
The codes for grasping on RaspberryPi are in folder grasping. In this folder, camera calibration is for setting the calibration between coordinates from carmera and coordinates form the end of the arm. In the file prediction_from_mkrcnn will calculate the mask and the coordinates and angle from the image sent by camera.
In the file grasping_for_maskrcnn2.py is a series of actions by robot for grasping after receiving the coordinates and angle. But sometimes the action have some delay because the average performance  on Maskrcnn is 5FPS which is not really suitable for real time. So this is why I write another grasping_for_maskrcnn.py, which used for input the predict coordinates and angle by hand. The kinematics.py is used for planing action. The remaining configuration files such as setup servo or calculate the servo's position for RaspberryPi I did not upload . If you want you can email me by qifan.wu@tufts.edu

# Two videos 
two video files : single and multiple  are the fragments from the experiment to see the performance on grasping.

# requirments
numpy
scipy
Pillow
cython
matplotlib
scikit-image
tensorflow
keras
opencv-python
h5py
imgaug
IPython

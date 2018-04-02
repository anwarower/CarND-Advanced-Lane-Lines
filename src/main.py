import cameraModel as cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import preprocessor as prep
import homography as homo

#cam = cm.Camera();
#cam.init(9, 6, 'camera_cal/calibration*.jpg')
#cam.calibrate();
#cam.validate();
preprocessor = prep.Preprocessor();
homographyOp = homo.Homography();

testImg = cv2.imread('test_images/test3.jpg')
#testImg = cv2.cvtColor(testImg, cv2.COLOR_RGB2GRAY)
croppingData = preprocessor.crop(testImg)
testImg = croppingData['imageR']
sobelImg = preprocessor.extractEdges(testImg, 'sat')
testImg = homographyOp.rectify(croppingData);
sobelImg = preprocessor.extractEdges(testImg, 'sat')
cv2.imshow('cropped', croppingData['imageR'])
cv2.imshow('rectified', testImg)
cv2.imshow('edges', sobelImg)
cv2.waitKey()

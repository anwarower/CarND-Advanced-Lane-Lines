import cameraModel as cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import preprocessor as prep
import homography as homo
import laneDetector as lf
import pipeline as fp
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from IPython.core.debugger import set_trace
#cam = cm.Camera();
#cam.init(9, 6, 'camera_cal/calibration*.jpg')
#cam.calibrate();
#cam.validate();
#cam = cm.Camera();
#preprocessor = prep.Preprocessor();
#homographyOp = homo.Homography();
#laneLinesFinder = lf.LaneLinesFinder();
algoPipeline = fp.FramePipeline();
#cam.init(9, 6, 'camera_cal/calibration*.jpg')
#cam.calibrate();

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#InputImg = cv2.imread('test_images/test1.jpg')
#output   = algoPipeline.processFrame(InputImg)
#cv2.imshow('out', output)
#cv2.waitKey()
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
inputPath = 'test.mp4'
outputPath = 'test_out.mp4'
clip1 = VideoFileClip(inputPath);
white_clip = clip1.fl_image(algoPipeline.processFrame) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'white_clip.write_videofile(outputPath, audio=False)')
#croppingData = preprocessor.crop(InputImg)
#croppedImg = croppingData['imageR']
#sobelImg = preprocessor.extractEdges(croppedImg, 'sat')
#rectImg = homographyOp.rectify({'imageR': sobelImg, 'controlPts':croppingData['controlPts']});
#sobelImg = preprocessor.extractEdges(testImg, 'sat')

#laneLinesFinder.showHist(rectImg);
#cv2.imshow('rectified', rectImg)
#cv2.imshow('cropped', croppingData['imageR'])
#cv2.imshow('original', InputImg)
#cv2.waitKey()
#laneLinesFinder.showHist(rectImg);
#warped_Result = laneLinesFinder.slidingWindowSearch(rectImg)
#output = laneLinesFinder.composeOutputFrame(rectImg, warped_Result, homographyOp.Minv, InputImg)

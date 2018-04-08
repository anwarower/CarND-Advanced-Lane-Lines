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

inputPath = 'project_video.mp4'
outputPath = 'project_submission.mp4'
clip1 = VideoFileClip(inputPath);
algoPipeline = fp.FramePipeline(clip1.size[0], clip1.size[1]);
videoClip = clip1.fl_image(algoPipeline.processFrame) #NOTE: this function expects color images!!
videoClip.write_videofile(outputPath, audio=False)

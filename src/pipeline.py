import cameraModel as cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import preprocessor as prep
import homography as homo
import laneDetector as lf
import visualizer as visu

class FramePipeline:
    cam = cm.Camera();
    preprocessor = prep.Preprocessor();
    homographyOp = homo.Homography();
    laneLinesFinder = None
    currOriginalFrame = None
    visualizer = None

    def __init__(self, frameWidth, frameHeight):
        self.frameWidth = frameWidth
        self.frameHeight = frameHeight
        self.cam.init(9, 6, 'camera_cal/calibration*.jpg')
        self.cam.calibrate();
        self.laneLinesFinder = lf.LaneLinesFinder(frameWidth, frameHeight);
        self.visualizer = visu.Visualizer(self.laneLinesFinder, self);


    def processFrame(self, InputImg):

        self.currOriginalFrame = InputImg

        undistortedImg= self.cam.undistortImg(InputImg)

        sobelImg     = self.preprocessor.extractEdges(undistortedImg, 'all')

        croppingData = self.preprocessor.crop(sobelImg)

        croppedImg   = croppingData['imageR']

        rectImg      = self.homographyOp.rectify({'imageR': sobelImg,
                                        'controlPts':croppingData['controlPts']});

        warped_out   = self.laneLinesFinder.findLane(rectImg)

        output = self.visualizer.visualizeFrame(rectImg)

        #only for the report at the end
        """cv2.imshow('after Sobel', sobelImg)
        cv2.imshow('after Cropping', croppedImg)
        cv2.imshow('after Rectifying', rectImg)
        cv2.imshow('after fitting', warped_out)
        cv2.imshow('after warping back', output)
        cv2.waitKey()"""
        return output;

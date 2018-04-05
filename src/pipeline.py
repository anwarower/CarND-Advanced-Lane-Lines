import cameraModel as cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import preprocessor as prep
import homography as homo
import laneDetector as lf

class FramePipeline:
    cam = cm.Camera();
    preprocessor = prep.Preprocessor();
    homographyOp = homo.Homography();
    laneLinesFinder = lf.LaneLinesFinder();

    def processFrame(self, InputImg):

        sobelImg     = self.preprocessor.extractEdges(InputImg, 'all')
        #cv2.imshow('after Sobel', sobelImg)
        #cv2.waitKey()
        croppingData = self.preprocessor.crop(sobelImg)
        croppedImg   = croppingData['imageR']
        #cv2.imshow('after Cropping', croppedImg)
        #cv2.waitKey()
        rectImg      = self.homographyOp.rectify({'imageR': sobelImg, 'controlPts':croppingData['controlPts']});
        #cv2.imshow('after Rectifying', rectImg)
        #cv2.waitKey()
        warped_Result= self.laneLinesFinder.findLane(rectImg)
        #cv2.imshow('after fitting', warped_Result)
        #cv2.waitKey()
        output       = self.laneLinesFinder.composeOutputFrame(rectImg,
                                                        warped_Result,
                                                        self.homographyOp.Minv,
                                                        InputImg)
        #cv2.imshow('after warping back', output)
        #cv2.waitKey()
        return output;

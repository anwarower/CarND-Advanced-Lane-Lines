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
        croppingData = self.preprocessor.crop(InputImg)
        croppedImg   = croppingData['imageR']
        sobelImg     = self.preprocessor.extractEdges(croppedImg, 'sat')
        rectImg      = self.homographyOp.rectify({'imageR': sobelImg, 'controlPts':croppingData['controlPts']});
        warped_Result= self.laneLinesFinder.slidingWindowSearch(rectImg)
        output       = self.laneLinesFinder.composeOutputFrame(rectImg,
                                                        warped_Result,
                                                        self.homographyOp.Minv,
                                                        InputImg)
        return output;

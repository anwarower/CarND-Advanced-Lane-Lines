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
        self.homographyOp.setFrameSize(frameWidth, frameHeight)
        self.homographyOp.estimateRoadHomography()
        self.laneLinesFinder = lf.LaneLinesFinder(frameWidth, frameHeight)
        self.visualizer = visu.Visualizer(self.laneLinesFinder, self)


    def processFrame(self, InputImg):

        self.currOriginalFrame = InputImg

        undistortedImg= self.cam.undistortImg(InputImg)

        sobelImg     = self.preprocessor.extractEdges(undistortedImg, 'all')

        croppedImg = self.preprocessor.crop(sobelImg)

        rectImg      = self.homographyOp.warp(croppedImg);

        warped_out   = self.laneLinesFinder.findLane(rectImg)

        output = self.visualizer.visualizeFrame(rectImg)

        #only for the report at the end
        #cv2.imwrite('afterUndist.jpg', undistortedImg)
        #cv2.imwrite('afterSobel.jpg', sobelImg)
        #cv2.imwrite('afterCropping.jpg', croppedImg)
        #cv2.imwrite('afterRectifying.jpg', rectImg)
        #cv2.imwrite('afterFitting.jpg', warped_out)
        #cv2.imwrite('afterWarpingBack.jpg', output)
        #cv2.waitKey()
        return output;

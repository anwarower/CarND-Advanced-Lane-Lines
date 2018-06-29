import cv2
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Homography:
    M  = [];
    Minv = [];
    goalPts2D = [];
    cropPts = [];
    inputImage = None;
    warpedImage = None;
    isScaleAdjusted = False
    frameWidth = 1280
    frameHeight = 720
    sourcePts = np.float32([[190, 720], [580, 460], [700, 460], [1100, 720]])
    dstPts    = np.float32([[190, 720], [190, 0  ], [1100, 0 ], [1100, 720]])

    def setFrameSize(self, newWidth, newHeight):
        self.frameWidth = newWidth
        self.frameHeight = newHeight

    def warp(self, inputImage):
        img_size = (inputImage.shape[1], inputImage.shape[0]);
        warpedImage = cv2.warpPerspective(inputImage, self.M, img_size, flags=cv2.INTER_LINEAR)
        return warpedImage

    def unwarp(self, inputImage):
        img_size = (self.frameWidth, self.frameHeight);
        unwarpedImage = cv2.warpPerspective(inputImage, self.Minv, img_size, flags=cv2.INTER_LINEAR)
        return unwarpedImage

    def calcM(self):
        self.M = cv2.getPerspectiveTransform(self.sourcePts, self.dstPts)
        self.Minv = cv2.getPerspectiveTransform(self.dstPts, self.sourcePts)

    def estimateRoadHomography(self):
        self.adjustBoundingRegionScale()
        self.calcM()
        #for report
        #refPath = 'test_images/straight_lines1.jpg'
        #straightLinesImg = cv2.imread(refPath)
        #self.setFrameSize(straightLinesImg.shape[1], straightLinesImg.shape[0])
        #self.adjustBoundingRegionScale()
        #self.calcM()
        #validImg = plt.imread(refPath);
        #self.validate(validImg)

    def adjustBoundingRegionScale(self):
        newWidth = self.frameWidth
        newHeight= self.frameHeight
        wR = newWidth/1280
        wH = newHeight/720
        self.sourcePts[:, 0] = self.sourcePts[:, 0] * wR
        self.sourcePts[:, 1] = self.sourcePts[:, 1] * wH
        self.dstPts[:, 0]    = self.dstPts[:, 0] * wR
        self.dstPts[:, 1]    = self.dstPts[:, 1] * wH
    #for the final report 
    def validate(self, inputImage):
        warpedImage = self.warp(inputImage)

        polyptsArray  = np.array(self.sourcePts, np.int32)
        polyptsArray  = polyptsArray.reshape((-1, 1, 2))
        polyptsArray2 = np.array(self.dstPts, np.int32)
        polyptsArray2 = polyptsArray2.reshape((-1, 1, 2))

        cv2.polylines(inputImage, [polyptsArray], True, (255, 0, 0), 10)
        cv2.polylines(warpedImage,[polyptsArray2], True, (255, 0, 0), 10)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.imshow(inputImage)
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(warpedImage)
        ax2.set_title('Warped Result', fontsize=30)
        plt.show()

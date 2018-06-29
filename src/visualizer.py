import numpy as np
import cv2
import laneDetector as ld
import pipeline as pl
class Visualizer:
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2
    laneFinder             = None
    algoPipeline           = None

    def __init__(self, laneFinder, algoPipeline):
        self.laneFinder = laneFinder
        self.algoPipeline = algoPipeline

    def addTextToImg(self, img, addedText, pos):
        cv2.putText(img,addedText,
                    pos,
                    self.font,
                    self.fontScale,
                    self.fontColor,
                    self.lineType)

    def addCurvInfoToImg(self, img, laneLeft, laneRight):
        curvatureL = laneLeft.evaluateIn3D(img.shape[0], 'curvature');
        curvatureR = laneRight.evaluateIn3D(img.shape[0], 'curvature');
        addedText = 'Left curvature: {:.2f}, Right curvature: {:.2f}'.format(curvatureL, curvatureR)
        pos = (10,30)
        self.addTextToImg(img, addedText, pos)

    def addLaneWidth(self, img, laneLeft, laneRight):
        xL  = laneLeft.evaluateX(img.shape[0]);
        xR  = laneRight.evaluateX(img.shape[0]);

        laneWidth = (xR - xL) * 3.2/920;
        addedText = 'Lane Width = {:.2f} m'.format(laneWidth)
        pos = (10, 60)
        self.addTextToImg(img, addedText, pos)

    def addDistanceToCenter(self, img, distanceValue):
        addedText = 'Vehicle distance from center: {:.2f} m'.format(distanceValue)
        pos = (10, 90)
        self.addTextToImg(img, addedText, pos)

    def visualizeFrame(self, img):
        lbLeft = self.laneFinder.laneBoundaryLeft;
        lbRight = self.laneFinder.laneBoundaryRight;
        output = self.addLanesToImg(img,
                                    self.algoPipeline.homographyOp.Minv,
                                    self.algoPipeline.currOriginalFrame,
                                    lbLeft,
                                    lbRight)
        self.addCurvInfoToImg(output, lbLeft, lbRight)
        self.addLaneWidth(output, lbLeft, lbRight)
        self.addDistanceToCenter(output, self.laneFinder.calculateDistanceFromCenter())
        return output;

    def addLanesToImg(self, binary_warped, Minv, undist, laneLeft, laneRight):
        result = undist;
        if((laneLeft.isDetectedLastFrame)&(laneRight.isDetectedLastFrame)):
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
            left_fitx  = laneLeft.evaluateX(ploty)
            right_fitx = laneRight.evaluateX(ploty)

            #try to vis
            warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
            color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

            # Recast the x and y points into usable format for cv2.fillPoly()
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

            # Warp the blank back to original image space using inverse perspective matrix (Minv)
            warpedBack = cv2.warpPerspective(color_warp, Minv, (binary_warped.shape[1], binary_warped.shape[0]))
            # Combine the result with the original image
            result = cv2.addWeighted(undist, 1, warpedBack, 0.3, 0)
        return result;

import numpy as np
import matplotlib.pyplot as plt
import cv2
import plausibility as plaus
class LaneBoundary:
    lane_inds = [];
    x = [];
    y = [];
    fit = []; #2nd degree polynomial coefficients
    isDetectedLastFrame = False;

    def reset(self):
        self.lane_inds = []
        self.x = []
        self.y = []
        self.fit = []
        self.isDetectedLastFrame = False;

    def evaluateX(self, y):
        #evaluate the polynomial
        if(len(self.fit) == 3):
            return (self.fit[0]*(y**2) + self.fit[1]*y +self.fit[2]);
        else:
            return -1

    def evaluateCurvature(self, y):
        #evaluate the polynomial
        if(len(self.fit) == 3):
            return ((1 + (2*self.fit[0]*y + self.fit[1])**2)**1.5) / np.absolute(2*self.fit[0])
        else:
            return -1

    def evaluateIn3D(self, y, quantity):
        if(len(self.fit) < 3):
            return 0
        my = 35/720 # meters per pixel in y dimension
        mx = 3.2/900 # meters per pixel in x dimension
        tempLane = LaneBoundary();
        #create a scaled polynomial in 3D
        tempLane.fit = [self.fit[0] * mx / (my ** 2),
                        self.fit[1] * (mx/my),
                        self.fit[2]
                        ]
        y3D = y * my;
        if(quantity == 'curvature'):
            return tempLane.evaluateCurvature(y3D);
        else:
            return tempLane.evaluateX(y3D);


class LaneLinesFinder:
    laneBoundaryLeft = LaneBoundary();
    laneBoundaryRight = LaneBoundary();
    plausiModule      = None
    nonzerox = [];
    nonzeroy = [];
    frameWidth = 0;
    frameHeight = 0;
    distanceFromCenter = 0;

    def __init__(self, frameWidth, frameHeight):
        self.frameWidth = frameWidth;
        self.frameHeight = frameHeight
        self.plausiModule = plaus.PFilter(frameWidth, frameHeight)

    def showHist(self, img):
        histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
        plt.plot(histogram)
        plt.show()


    def reset(self, img):
        self.laneBoundaryLeft.reset()
        self.laneBoundaryRight.reset()

    def calcLaneWidthInPx(self):
        yAtEvaluation = self.frameHeight;
        return self.laneBoundaryRight.evaluateX(yAtEvaluation) - self.laneBoundaryLeft.evaluateX(yAtEvaluation)

    def calculateDistanceFromCenter(self):
        imgCenter = self.frameWidth/2
        laneCenter = (self.laneBoundaryRight.evaluateX(self.frameHeight) + self.laneBoundaryLeft.evaluateX(self.frameHeight))/2
        distanceFromCenter = (laneCenter - imgCenter) * 3.2/920
        return distanceFromCenter

    def findLane(self, binary_warped):
        output = binary_warped
        if(self.laneBoundaryLeft.isDetectedLastFrame & self.laneBoundaryRight.isDetectedLastFrame):
            output = self.BoundingRegionSearch(binary_warped)
            self.plausiModule.filter(self.laneBoundaryLeft,
                                             self.laneBoundaryRight)
        else:
            output = self.slidingWindowSearch(binary_warped)
            self.plausiModule.filter(self.laneBoundaryLeft,
                                             self.laneBoundaryRight)
        return output


    def slidingWindowSearch(self, binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        self.laneBoundaryLeft.lane_inds = []
        self.laneBoundaryRight.lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (0,255,0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) &
            (self.nonzerox >= win_xleft_low) &  (self.nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) &
            (self.nonzerox >= win_xright_low) &  (self.nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            self.laneBoundaryLeft.lane_inds.append(good_left_inds)
            self.laneBoundaryRight.lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(self.nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(self.nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        self.laneBoundaryLeft.lane_inds  = np.concatenate(self.laneBoundaryLeft.lane_inds)
        self.laneBoundaryRight.lane_inds = np.concatenate(self.laneBoundaryRight.lane_inds)

        # Extract left and right line pixel positions
        self.laneBoundaryLeft.x = self.nonzerox[self.laneBoundaryLeft.lane_inds]
        self.laneBoundaryLeft.y = self.nonzeroy[self.laneBoundaryLeft.lane_inds]
        self.laneBoundaryRight.x = self.nonzerox[self.laneBoundaryRight.lane_inds]
        self.laneBoundaryRight.y = self.nonzeroy[self.laneBoundaryRight.lane_inds]

        # Fit a second order polynomial to each
        if(len(self.laneBoundaryLeft.x) > 0):
            self.laneBoundaryLeft.fit  = np.polyfit(self.laneBoundaryLeft.y, self.laneBoundaryLeft.x, 2)
            self.laneBoundaryLeft.isDetectedLastFrame = True;
        else:
            self.laneBoundaryLeft.isDetectedLastFrame = False;
        if(len(self.laneBoundaryRight.x) > 0):
            self.laneBoundaryRight.fit  = np.polyfit(self.laneBoundaryRight.y, self.laneBoundaryRight.x, 2)
            self.laneBoundaryRight.isDetectedLastFrame = True;
        else:
            self.laneBoundaryLeft.isDetectedLastFrame = False;

        #for the final report
        #plt.imshow(out_img)
        #ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        #plt.plot(self.laneBoundaryLeft.evaluateX(ploty), ploty, color='yellow')
        #plt.plot(self.laneBoundaryRight.evaluateX(ploty), ploty, color='yellow')
        #plt.xlim(0, 1280)
        #plt.ylim(720, 0)
        #plt.imsave('output_images/afterPoly.jpg', out_img)
        #plt.show()
        return out_img

    def BoundingRegionSearch(self, binary_warped):
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])
        margin = 100

        self.laneBoundaryLeft.lane_inds = ((self.nonzerox > (self.laneBoundaryLeft.evaluateX(self.nonzeroy) - margin))
                                         & (self.nonzerox < (self.laneBoundaryLeft.evaluateX(self.nonzeroy) + margin)))

        self.laneBoundaryRight.lane_inds = ((self.nonzerox > (self.laneBoundaryRight.evaluateX(self.nonzeroy) - margin))
                                         &  (self.nonzerox < (self.laneBoundaryRight.evaluateX(self.nonzeroy) + margin)))

        # Again, extract left and right line pixel positions
        self.laneBoundaryLeft.x  = self.nonzerox[self.laneBoundaryLeft.lane_inds]
        self.laneBoundaryLeft.y  = self.nonzeroy[self.laneBoundaryLeft.lane_inds]
        self.laneBoundaryRight.x = self.nonzerox[self.laneBoundaryRight.lane_inds]
        self.laneBoundaryRight.y = self.nonzeroy[self.laneBoundaryRight.lane_inds]
        # Fit a second order polynomial to each
        if((len(self.laneBoundaryLeft.y) > 3) & (len(self.laneBoundaryLeft.y) > 3)):
            self.laneBoundaryLeft.fit  = np.polyfit(self.laneBoundaryLeft.y, self.laneBoundaryLeft.x, 2)
        else:
            self.laneBoundaryLeft.isDetectedLastFrame = False;
        if((len(self.laneBoundaryRight.y) > 3) & (len(self.laneBoundaryRight.y) > 3)):
            self.laneBoundaryRight.fit = np.polyfit(self.laneBoundaryRight.y, self.laneBoundaryRight.x, 2)
        else:
            self.laneBoundaryRight.isDetectedLastFrame = False;

        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

        out_img[self.laneBoundaryLeft.y, self.laneBoundaryLeft.x] = [255, 0, 0]
        out_img[self.laneBoundaryRight.y, self.laneBoundaryRight.x] = [0, 0, 255]

        """#for the final report
        plt.imshow(out_img)
        #plt.plot(self.laneBoundaryLeft.evaluateX(ploty), ploty, color='yellow')
        #plt.plot(self.laneBoundaryRight.evaluateX(ploty), ploty, color='yellow')
        #plt.xlim(0, 1280)
        #plt.ylim(720, 0)
        #plt.show()"""
        return out_img;

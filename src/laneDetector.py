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
    
    def evaluateX(self, y):
        #evaluate the polynomial
        if(len(self.fit) == 3):
            return (self.fit[0]*(y**2) + self.fit[1]*y +self.fit[2]);
        else:
            return 0


class LaneLinesFinder:
    laneBoundaryLeft = LaneBoundary();
    laneBoundaryRight = LaneBoundary();
    plausiModule      = plaus.PFilter();
    nonzerox = [];
    nonzeroy = [];

    def showHist(self, img):
        histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
        plt.plot(histogram)
        plt.show()

    def findLane(self, binary_warped):
        output = self.slidingWindowSearch(binary_warped)
        #if:
        #    return self.BoundingRegionSearch(binary_warped)
        #else:
        #    return self.slidingWindowSearch(binary_warped)
        #self.plausiModule.filter(self.laneBoundaryLeft,
        #                         self.laneBoundaryRight)
        return output;




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
        self.laneBoundaryLeft.fit  = np.polyfit(self.laneBoundaryLeft.y, self.laneBoundaryLeft.x, 2)
        self.laneBoundaryRight.fit = np.polyfit(self.laneBoundaryRight.y, self.laneBoundaryRight.x, 2)


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
            if((len(self.laneBoundaryRight.y) > 3) & (len(self.laneBoundaryRight.y) > 3)):
                self.laneBoundaryLeft.fit  = np.polyfit(self.laneBoundaryLeft.y, self.laneBoundaryLeft.x, 2)
                self.laneBoundaryRight.fit = np.polyfit(self.laneBoundaryRight.y, self.laneBoundaryRight.x, 2)


                #return self.slidingWindowSearch(binary_warped)
        # Generate x and y values for plotting
        #ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        #left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        #right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        #print('tracked')


    def composeOutputFrame(self, binary_warped, out_img, Minv, undist):
            # Generate x and y values for plotting
            #cv2.imshow('s', out_img)
            #plt.imshow(out_img)
            #plt.show()
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
            #left_fit = self.left_fit;
            #right_fit = self.right_fit;

            left_fitx  = self.laneBoundaryLeft.evaluateX(ploty)
            right_fitx = self.laneBoundaryRight.evaluateX(ploty)

            leftIdx = self.laneBoundaryLeft.lane_inds;
            rightIdx= self.laneBoundaryRight.lane_inds;
            #out_img[self.nonzeroy[leftIdx], self.nonzerox[leftIdx]] = [255, 0, 0]
            #out_img[self.nonzeroy[rightIdx], self.nonzerox[rightIdx]] = [0, 0, 255]
            #plt.imshow(out_img)
            #plt.plot(left_fitx, ploty, color='yellow')
            #plt.plot(right_fitx, ploty, color='yellow')
            #plt.xlim(0, 1280)
            #plt.ylim(720, 0)
            #plt.show()

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
            newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
            # Combine the result with the original image
            result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
            return result;

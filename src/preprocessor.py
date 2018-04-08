import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Preprocessor:
    thresh_sobelx_max = {'gray': 100, 'sat': 100}
    thresh_sobelx_min = {'gray': 50 , 'sat': 20 }

    def crop(self, image):
        #Clip the ROI
        xLength = image.shape[1];
        yLength = image.shape[0];
        resultImg = np.copy(image);

        shiftUp = 100/720 * yLength#75/540 * yLength; #75
        shiftSideUp = 550/1280 * xLength;#400/960 *xLength; #400
        BoundaryUp = image.shape[0]/2 +shiftUp;
        BoundaryDown = yLength;
        BoundaryUpLX = shiftSideUp;
        BoundaryUpRX = xLength - shiftSideUp;
        LeftUp = [BoundaryUpLX, BoundaryUp];
        DownIdent = 0;
        LeftDown = [DownIdent, BoundaryDown];
        RightUp = [BoundaryUpRX, BoundaryUp];
        RightDown = [xLength-DownIdent, BoundaryDown];

        BoundaryL = np.polyfit([LeftDown [0], LeftUp [0]], [LeftDown [1], LeftUp [1]], 1);
        BoundaryR = np.polyfit([RightDown[0], RightUp[0]], [RightDown[1], RightUp[1]], 1);

        XX, YY = np.meshgrid(np.arange(0, image.shape[1]),                          np.arange(0, image.shape[0]))

        GoodIndecesR = (YY >= (XX * BoundaryR[0] + BoundaryR[1])) & (YY > RightUp[1])
        GoodIndecesL = (YY >= (XX * BoundaryL[0] + BoundaryL[1])) & (YY > LeftUp[1])
        GoodIndeces = GoodIndecesL & GoodIndecesR

        badIndeces = ~GoodIndeces
        resultImg[badIndeces] = 0;
        controlPts = np.float32([LeftDown, LeftUp, RightUp, RightDown])
        return {'imageR': resultImg, 'controlPts': controlPts}

    def extractChannel(self, img, mode):
        if(mode == 'gray'):
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return gray
        else:
            if(mode == 'sat'):
                hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
                sat = hls[:,:,2]
                return sat

    def extractEdges(self, img, mode):
        if(mode == 'all'):
            gray = self.extractChannel(img, 'gray')
            grayBinary = self.applySobel(gray, self.thresh_sobelx_min['gray'], self.thresh_sobelx_max['gray'])
            sat  = self.extractChannel(img, 'sat')
            satBinary  = self.applySobel(sat, self.thresh_sobelx_min['sat'], self.thresh_sobelx_max['sat'])
            result = grayBinary | satBinary;
            return result;
        else:
            channel = self.extractChannel(img, mode)
            channelBinary = self.applySobel(channel, self.thresh_sobelx_min[mode], self.thresh_sobelx_max[mode])
            return channelBinary

    def applySobel(self, img, thresh_min, thresh_max):
        sobel_kernel = 5
        # Take the gradient in x
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, sobel_kernel)
        # Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled_sobel = np.uint8(255 * sobelx / np.max(sobelx));
        # Create a binary mask where mag thresholds are met
        binary_output = np.zeros_like(scaled_sobel);
        binary_output[(scaled_sobel > thresh_min) & (scaled_sobel <= thresh_max)] = 255
        return binary_output

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Camera:
    P = [];
    R = [];
    t = [];
    DistCoeff = [];
    nx = 0;
    ny = 0;
    chessBoardPath = '';
    testImgPath = '';
    input3DCoords = [];
    map_3D = [];
    map_2D = [];
    img_size = [0, 0];



    def init(self, nCalx, nCaly, pathName):
        self.nx = nCalx;
        self.ny = nCaly;
        self.chessBoardPath = pathName;
        self.testImgPath = self.chessBoardPath.replace('*', '1');
        self.init3DCoords();
        self.setImgSize();



    def init3DCoords(self):
        self.input3DCoords = np.zeros((self.ny*self.nx,3), np.float32)
        self.input3DCoords[:,:2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1,2)



    def setImgSize(self):
        img = cv2.imread(self.testImgPath)
        self.img_size = (img.shape[1], img.shape[0])



    def prepare2D3DMap(self):
        # Make a list of calibration images
        images = glob.glob(self.chessBoardPath)

        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.nx,self.ny), None)

            # If found, add object points, image points
            if ret == True:
                self.map_3D.append(self.input3DCoords)
                self.map_2D.append(corners)

                #for the final report
                """
                #Draw and display the corners
                #cv2.drawChessboardCorners(img, (self.nx,self.ny), corners, ret)
                #write_name = 'corners_found'+str(idx)+'.jpg'
                #cv2.imwrite(write_name, img)
                #cv2.imshow('img', img)
                #cv2.waitKey(500)
                #cv2.destroyAllWindows()"""



    def calibrate(self):
        self.prepare2D3DMap()
        ret, self.P, self.DistCoeff, self.R, self.t = cv2.calibrateCamera(self.map_3D,
                                                                          self.map_2D,
                                                                          self.img_size,
                                                                          None,
                                                                          None)



    def undistortImg(self, img):
        dst = cv2.undistort(img, self.P, self.DistCoeff, None, self.P)
        return dst



    def validate(self):
        img     = cv2.imread(self.testImgPath)
        imgDash = self.undistortImg(img);
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(imgDash)
        ax2.set_title('Undistorted Image', fontsize=30)
        plt.show()

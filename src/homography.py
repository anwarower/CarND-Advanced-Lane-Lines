import cv2
import numpy as np
class Homography:
    M  = [];
    Minv = [];
    def rectify(self, cropData):
        image = cropData['imageR']
        cropPts = cropData['controlPts']
        goalPts2D = np.float32([
        #[0, image.shape[0]],
        #[0, 0],
        cropPts[0],
        [cropPts[0][0], 0],
        #[image.shape[1], 0],
        #[image.shape[1], image.shape[0]]
        [cropPts[3][0], 0],
        cropPts[3]
        ])
        img_size =  (image.shape[1], image.shape[0])
        self.M = cv2.getPerspectiveTransform(cropPts, goalPts2D)
        self.Minv = cv2.getPerspectiveTransform(goalPts2D, cropPts)
        warped = cv2.warpPerspective(image, self.M, img_size, flags=cv2.INTER_LINEAR)
        return warped;

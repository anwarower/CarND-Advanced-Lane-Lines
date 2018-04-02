import cv2
import numpy as np
class Homography:
    def rectify(self, cropData):
        image = cropData['imageR']
        cropPts = cropData['controlPts']
        print('imageshape:')
        print(image.shape[1])
        goalPts2D = np.float32([
        [0, image.shape[0]],
        [0, 0],
        #cropPts[0],
        #[cropPts[0][0], 0],
        [image.shape[1], 0],
        [image.shape[1], image.shape[0]]
        #[cropPts[3][0], 0],
        #cropPts[3]
        ])
        img_size =  (image.shape[1], image.shape[0])
        M = cv2.getPerspectiveTransform(cropPts, goalPts2D)
        warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
        return warped;

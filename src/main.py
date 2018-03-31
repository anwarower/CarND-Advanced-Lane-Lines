import cameraModel as cm
import numpy as np
import matplotlib.pyplot as plt
import cv2

cam = cm.Camera();
cam.init(9, 6, 'camera_cal/calibration*.jpg')
cam.calibrate();
cam.validate();

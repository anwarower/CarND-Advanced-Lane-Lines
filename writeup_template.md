## Advanced Lane Finding Project
### This project is part of the Udacity's Self-Driving Cars Nanodegree program.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistortionValidation.PNG "Undistorted"
[image2]: ./output_images/afterUndist.jpg "Road Transformed"
[image3a]: ./output_images/afterSobel.jpg "Binary Example"
[image3b]: ./output_images/afterCropping.jpg "Cropped Example"
[image4]: ./output_images/warpingResult.PNG "Warp Example"
[image5b]: ./output_images/afterFitting.jpg "Fit Visual"
[image5c]: ./output_images/afterPoly.jpg
[image6]: ./output_images/afterWarpingBack.jpg "Output"
[video1]:https://www.youtube.com/watch?v=b7vBtLmvcEQ&feature=youtu.be "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

In the file `src/cameraModel.py`, a simple camera model is built. A class instance of a camera has a list of 2D points and a list of 3D points. Both lists describe the same vertices for a chessboard used for calibration. This 2D/3D map is used to calibrate the camera using the function `calibrate()`. The latter function outputs the projection matrix `P` of the camera, which maps a 3D point to a 2D point on the image, and outputs also the distortion coefficients `DistCoeff`, which are used to rectify the lens distortion of the camera.    

To validate the correct acquisition of the right distortion parameters, I used the function `validate()` in the same file, which shows an example of removing radial distortion.

![removing the radial distortion of the lens ][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Here is one example:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The relevant file in the code for this part is the `src/preprocessor`. The functionality of this class is to
apply a chain of preprocessing steps to the input images to extract the interesting features and eventually output
a binary image. The non-zero elements in this binary images localize candidate features to our saught lane lines. The entry point of this preprocessing is the function `findedges()`. This functions applies the sobel operator to extract strong transitions in the gradient (edges). Since lane lines are almost always vertical in the image domain, I did not use really a gradient, but rather the derrivative in the x-direction. The function can operate on different modes, either `gray`, `sat` or `all`. To ensure a robust performance under varying light conditions, the mode `all` was used, which combines the output from the gray channel and the saturaton channel. For each of both channels I had to play a bit around to find out the optimal thresholds.

![alt text][image3a]

After obtaining reasonably a good binary image, I furthermore cropped the image to a heuristically important region defined by a trapezium. The result after cropping can be visualized here:
![alt Text][image3b]
#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
The relevant part in the code for this is the file `homography.py`. Before processing the video stream, I estimate a matrix called the road homography matrix. This is a planar map that defines how the road plane in the image would correspond to the road plane in the eagle eye view. To find this matrix, I selected some points in the original image containing the interesting region where the lane lines are. Moreover, I selected some points which would map the lane lines to parallel lines. For this sake I used an image where the lane lines are parallel.

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 190, 720      | 190  , 720    |
| 580, 460      | 190  , 0      |
| 700, 460      | 1100, 0       |
| 1100, 720     | 1100, 720     |

I used the source and destination points to estimate the matrix using the function `cv2.getPerspectiveTransform()`.
I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The relevant part of the code for this is the file `laneDetector.py`. Here, 2 different classes are defined:
* `LaneBoundary`: An instance of this class contains lane boundary information, such as its polynomial fit.  
* `LaneLinesFinder`: An instance of this class has all the methods that are used for finding lanes in an image.
An instance of this class has typically 2 instances of `LaneBoundary` for left and right sides .

As a model for the lane lines(also called lane boundaries), we select a 2nd degree polynomial.
To find the offset of this polynomial, we calculate a histogram of the rectified image. typically, the offset place corresponds to maxima in the histogram. Here is an example histogram:
![alt text][image5a]
Finding the offset is equivalent finding the x-coordinates of the lane lines at y = `image.shape[1]`.
Having calculated it, we use this leap of faith to find the x-coordinates in the rest of the image. we divide the height of the image to equal vertical regions. In the region `n` we use the assumption that the sought x coordinates would lie in the vicinity of the ones in region `n - 1`. The tolerance region in the horizontal direction along with the defined region height define our window. Therefore the algorithm is called 'slidingWindowSearch'.The defined windows can be visualized here:
![alt text][image5b]

In each step, points found inside the window are appended to the lists `x` and `y` found in the `LaneBoundary` structure. Finally, our sought polynomial is calculated by fitting x coordinates and the y coordinates together using the function `numpy.polyfit()`. This yields a result similar to this:
![alt text][image5c]

Since such a blind search that does not use the previous measurement would be pretty expensive in terms of computation, we add a lighter variant for finding the lanes. In case we had already a stable detection in the past, we take the assumption that the lane line polynomial would have the same "shape" but could just be shifted a bit. This is equivalent to performing the sliding window algorithm but for just a single window that is as tall as the image height! :)
The function implementing this light version of lane detection is called `BoundingRegionSearch`
#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
For the curvature:

In the structure `LaneBoundary` (see last point), there is a function that calculates the curvature, which is called `evaluateCurvature()`. This function merely use this equation to calculate the radius of curvature at y = y_eval:
```python
rCurv =  ((1 + (2*fit[0]*y_eval + fit[1])**2)**1.5) / np.absolute(2*fit[0])
```
Since this function uses the `x` and `y` attribute lists, the curvature would be in image coordinates. The same would be for finding the position of the vehicle with respect to the center. We thus need a mapping between pixel coordinates and 3D world coordinates.
This problem is solved in the function `evaluateIn3D()`. This function takes as input
The function then creates a scaled version of the calculated fit using the relationship:
```python
scaledVersion.fit = [self.fit[0] * mx / (my ** 2),
                self.fit[1] * (mx/my),
                self.fit[2]
                ]
```
The scaled parabola would then yield the sought values in 3D.
In the above code snippet, `mx` and `my` are ratios that define the scale factor between pixel coordinates and 3D coordinates. To adjust the scale in x direction, I calculated the lane width in pixel coordinates and mapped the width with my good feeling to typical lane width in the U.S (3.3m). For the scale in y direction I just used the values used in the lesson and they yielded pretty good results. Here are the scale values used:
```python
my = 30 / 720 # meters per pixel in y dimension
mx = 3.7 / 900 # meters per pixel in x dimension
```

For the distance from center:
This boils down to the function:
```python
distanceFromCenter = (laneCenter - imgCenter) * mx
```
where `laneCenter` is the midpoint between the right lane offset and the left lane offset and `imgCenter` is half the width of the image.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The relevant part for this in the code is the file  `Visualizer.py`. The function `visualizeFrame()` uses the inverse of homography matrix used earlier for the image rectification to inject the calculated polynomial back into the original image coordinates. Moreover, it overlays the output frame with text information about the radius of curvature, vehicle distance from center and the lane width. An example output can be seen here:  

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://www.youtube.com/watch?v=b7vBtLmvcEQ&feature=youtu.be)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
The most prominent weak point of this algorithm is the way how the projective mapping (homography) is estimated. This is because it assumes two things: The camera mounting is always fixed and the street is always planar!
Moreover, the algorithm does not compensate for the ego motion of the vehicle. For this reason, a change in the pitching angle can already introduce an error factor to the estimation, as if the camera mounting has been changed. Moreover, the simplicity of the ROI selection is pretty dangerous. In scenarios like a lane change or a sharp curving road, the assumption that the lane boundaries would lie in the scope of our heuristically selected trapezium will be violated.

Although a smoothing filtering has been applied between frames, and although the past estimations were used to narrow down the search space in the next frames, the algorithm does not really exploit the inter-frame correspondences. This makes it in fact much less robust.

If I get the time to work on this project further, I will track points of interest along frames. Using this correspondence, I would derive the camera pose dynamically to account for the ego motion. I would then feed the ego motion into a probablistic tracker which would help me predict the new positioning of the lane lines robustly.

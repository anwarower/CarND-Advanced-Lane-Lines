import laneDetector as ld
import numpy as np
class PFilter: #Plausibility Filter
    bufferLeft = []
    bufferRight = []
    N = 15; #frames
    widthBuffer = []
    frameWidth = 0;
    frameHeight = 0;

    def __init__(self, frameWidth, frameHeight):
        self.frameWidth = frameWidth
        self.frameHeight = frameHeight

    def filter(self, LBLeft, LBRight):
        #print('filtering')
        if(self.plausibilize(LBLeft, LBRight)):
            self.smoothen(LBLeft, LBRight)
        else:
            if(len(self.bufferRight) > 1):
                self.calcAverage(LBLeft, LBRight)
                self.forgetOldPast();
            else:
                LBLeft.isDetectedLastFrame = False;
                LBRight.isDetectedLastFrame = False;

    def isLaneWidthOK(self, LBLeft, LBRight):
        widthDevThresh = 0.15
        minimumWidthThresh = 600 #px
        isWidthConsistent = True
        isWidthBigEnough =  False
        #check if the width is consistent with the local history
        currentWidth = LBRight.evaluateX(self.frameHeight) - LBLeft.evaluateX(self.frameHeight)
        if(len(self.widthBuffer) > 0):
            avgWidth     = self.getAverageWidth();
            widthDev     = self.calcDevPerc(avgWidth, currentWidth);
            isWidthConsistent  = widthDev <= widthDevThresh;
        #check if the width is reasonably big enough
        isWidthBigEnough   = currentWidth > minimumWidthThresh;

        isWidthGood = isWidthConsistent & isWidthBigEnough
        return isWidthGood

    def areLaneBoundariesParallel(self, LBLeft, LBRight):
        widthDiffThresh = 0.2;
        farRange  = min(max(LBLeft.y), max(LBRight.y))
        nearRange = max(min(LBLeft.y), min(LBRight.y))
        distanceFar = LBRight.evaluateX(farRange) - LBLeft.evaluateX(farRange)
        distanceNear = LBRight.evaluateX(nearRange) - LBLeft.evaluateX(nearRange)
        deviationPercentage = self.calcDevPerc(distanceFar, distanceNear)
        isParallel  = deviationPercentage <= widthDiffThresh
        return isParallel

    def plausibilize(self, LBLeft, LBRight):
        #sanity checks
        areAvailable = (len(LBLeft.x) > 3) & (len(LBRight.x) > 3)
        if(areAvailable == False):
            return False
        isParallel  = self.areLaneBoundariesParallel(LBLeft, LBRight);
        isWidthGood = self.isLaneWidthOK(LBLeft, LBRight);
        return isWidthGood & isParallel

    def smoothen(self, LBLeft, LBRight):
        self.addToBuffer(LBLeft, LBRight)
        self.calcAverage(LBLeft, LBRight)

    def calcAverage(self, LBLeft, LBRight):
        divisor = min(self.N, len(self.bufferRight))
        newLeftFit = [sum(i)/divisor for i in zip(*self.bufferLeft)]
        newRightFit =[sum(i)/divisor for i in zip(*self.bufferRight)]
        LBLeft.fit = newLeftFit;
        LBRight.fit = newRightFit;

    def getAverageWidth(self):
        divisor = min(self.N, len(self.widthBuffer))
        if(divisor == 0):
            return 0;
        else:
            return sum(self.widthBuffer)/divisor;

    def calcDevPerc(self, val1, val2):
        divisor = max(val1, val2);
        result = -1;
        if(divisor > 0):
            return (np.absolute(val1 - val2)/divisor)
        return result

    def forgetOldPast(self):
        self.bufferLeft = self.bufferLeft[1: len(self.bufferLeft)]
        self.bufferRight= self.bufferRight[1: len(self.bufferRight)]
        self.widthBuffer= self.widthBuffer[1: len(self.widthBuffer)]

    def addToBuffer(self, LBLeft, LBRight):
        self.bufferLeft.append(LBLeft.fit)
        self.bufferRight.append(LBRight.fit)
        self.widthBuffer.append(LBRight.evaluateX(self.frameHeight) - LBLeft.evaluateX(self.frameHeight))

        if(len(self.bufferLeft) > self.N):
            self.forgetOldPast();

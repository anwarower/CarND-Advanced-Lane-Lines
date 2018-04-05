import laneDetector as ld
import numpy as np
class PFilter: #Plausibility Filter
    bufferLeft = []
    bufferRight = []
    N = 15; #frames

    def filter(self, LBLeft, LBRight):
        #print('filtering')
        if(self.plausibilize(LBLeft, LBRight)):
            self.smoothen(LBLeft, LBRight)
        else:
            self.calcAverage(LBLeft, LBRight)


    def plausibilize(self, LBLeft, LBRight):
        #sanity checks
        isParallel = False;
        isWidthGood = True; ## TODO:

        widthDiffThresh = 50;
        farRange  = min(max(LBLeft.y), max(LBRight.y))
        nearRange = max(min(LBLeft.y), min(LBRight.y))
        distanceFar = LBRight.evaluateX(farRange) - LBLeft.evaluateX(farRange)
        distanceNear = LBRight.evaluateX(nearRange) - LBLeft.evaluateX(nearRange)
        isParallel  = np.absolute(distanceFar - distanceNear) <= widthDiffThresh
        #print(isParallel)
        return isParallel & isWidthGood

    def smoothen(self, LBLeft, LBRight):
        self.addToBuffer(LBLeft, LBRight)
        self.calcAverage(LBLeft, LBRight)

    def calcAverage(self, LBLeft, LBRight):
        divisor = min(self.N, len(self.bufferRight))
        newLeftFit = [sum(i)/divisor for i in zip(*self.bufferLeft)]
        newRightFit =[sum(i)/divisor for i in zip(*self.bufferRight)]
        LBLeft.fit = newLeftFit;
        LBRight.fit = newRightFit;

    def addToBuffer(self, LBLeft, LBRight):
        self.bufferLeft.append(LBLeft.fit)
        self.bufferRight.append(LBRight.fit)
        if(len(self.bufferLeft) > self.N):
            self.bufferLeft = self.bufferLeft[1: len(self.bufferLeft)]
            self.bufferRight= self.bufferRight[1: len(self.bufferRight)]

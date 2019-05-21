import numpy as np
from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure
import cv2
import imutils
from imutils import perspective


class LicensePlateDetector:
    def __init__(self, image, minPlateW=60, minPlateH=20, numChars=7, minCharW=40):
        self.image = image
        self.minPlateW = minPlateW
        self.minPlateH = minPlateH
        self.numChars = numChars
        self.minCharW = minCharW

    def detect(self):
        # return self.detectPlates()

        lpRegions = self.detectPlates()

        for lpRegion in lpRegions:

            lp = self.detectCharacterCandidates(lpRegion)

    def detectPlates(self):

        regions = []

        HSV = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV", HSV)

        Upper = np.array([120, 255, 255])
        Lower = np.array([90, 127, 120])

        H = cv2.inRange(HSV, Lower, Upper)
        # cv2.imshow("A", H)

        thresh = cv2.threshold(H, 100, 255, cv2.THRESH_BINARY)[1]
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (7, 7))
        # cv2.imshow("B", thresh)

        thresh = cv2.erode(H, None, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (7, 7))
        thresh = cv2.dilate(thresh, None, iterations=4)
        # thresh = cv2.erode(thresh, None, iterations=1)
        # thresh = cv2.dilate(thresh, None, iterations=1)
        # cv2.imshow("C", thresh)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1]
        hullImage = np.zeros(thresh.shape[:2], dtype="uint8")

        for c in cnts:

            hull = cv2.convexHull(c)

            cv2.drawContours(hullImage, [hull], -1, 255, -1)

            cnts = cv2.findContours(hullImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[1]

            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.01 * peri, True)

                if len(approx) >= 4 and len(approx) <= 6:
                    (x, y, w, h) = cv2.boundingRect(approx)
                    aspectRatio = w / float(h)
                    # print("aspectRatio:", aspectRatio)

                    rect = cv2.minAreaRect(c)
                    box = np.int0(cv2.boxPoints(rect))

                    if (aspectRatio > 2.4 and aspectRatio < 4) and h > self.minPlateH and w > self.minPlateW:
                        regions.append(box)

            cv2.imshow("Convex Hull", hullImage)
        return regions

    def detectCharacterCandidates(self, region):
        chars = []

        plate = perspective.four_point_transform(self.image, region)
        cv2.imshow("Perspective Transform", imutils.resize(plate, width=400))

        HSV = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
        Upper = np.array([120, 255, 255])
        Lower = np.array([90, 127, 120])
        H = cv2.inRange(HSV, Lower, Upper)

        thresh = cv2.threshold(H, 100, 255, cv2.THRESH_BINARY_INV)[1]

        plate = imutils.resize(plate, width=400)
        thresh = imutils.resize(thresh, width=400)
        perChar = cv2.erode(thresh, None, iterations=1)
        cv2.imshow("perChar", perChar)

        thresh = cv2.erode(thresh, None, iterations=3)
        cv2.imshow("LP erode", thresh)

        (LPh, LPw) = thresh.shape[:2]
        rectangle = np.zeros((LPh, LPw), dtype="uint8")
        cv2.rectangle(rectangle, (20, 20), (LPw - 20, LPh - 20), 255, -1)
        cv2.imshow("Rectangle", rectangle)

        perBitwiseAnd = cv2.bitwise_and(rectangle, thresh)
        cv2.imshow("LP Mask", perBitwiseAnd)

        bitwiseAnd = cv2.morphologyEx(perBitwiseAnd.copy(), cv2.MORPH_OPEN, (7, 7))
        bitwiseAnd = cv2.morphologyEx(bitwiseAnd, cv2.MORPH_OPEN, (7, 7))
        bitwiseAnd = cv2.morphologyEx(bitwiseAnd, cv2.MORPH_OPEN, (7, 7))
        bitwiseAnd = cv2.dilate(bitwiseAnd, None, iterations=4)
        bitwiseAnd = cv2.threshold(bitwiseAnd, 10, 255, cv2.THRESH_BINARY)[1]
        cv2.imshow("LP mask and", bitwiseAnd)

        # labels = measure.label(thresh, neighbors=8, background=0)
        # charCandidates = np.zeros(thresh.shape, dtype="uint8")

        cnts = cv2.findContours(bitwiseAnd.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1]
        hullImage = np.zeros(bitwiseAnd.shape[:2], dtype="uint8")

        for c in cnts:

            hull = cv2.convexHull(c)

            cv2.drawContours(hullImage, [hull], -1, 255, -1)

            cv2.imshow("Convex Hull", hullImage)

            labels = measure.label(hullImage, neighbors=8, background=0)
            charCandidates = np.zeros(hullImage.shape, dtype="uint8")

            for label in np.unique(labels):

                if label == 0:
                    continue

                labelMask = np.zeros(hullImage.shape, dtype="uint8")
                labelMask[labels == label] = 255
                cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[1]

                if len(cnts) > 0:

                    c = max(cnts, key=cv2.contourArea)
                    (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

                    aspectRatio = boxW / float(boxH)
                    solidity = cv2.contourArea(c) / float(boxW * boxH)
                    heightRatio = boxH / float(plate.shape[0])

                    keepAspectRatio = aspectRatio < 1.0
                    keepSolidity = solidity > 0.15
                    keepHeight = heightRatio > 0.4 and heightRatio < 0.95

                    if keepAspectRatio and keepSolidity and keepHeight:

                        hull = cv2.convexHull(c)
                        cv2.drawContours(charCandidates, [hull], -1, 255, -1)

            cv2.imshow("Chose Convex Hull", charCandidates)

# 当用从蓝底阈值处理出发，一直做到这里的时候，发现最后处理完的每一个字符的convexHull
# 和之前的图去and的时候会有误裁剪的情况
#
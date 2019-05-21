import numpy as np
from collections import namedtuple
from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure
import cv2
import imutils
from imutils import perspective

LicensePlate = namedtuple("LicensePlateRegion", ["success", "plate", "thresh", "candidates"])

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

            if lp.success:
                chars = self.scissor(lp)

                yield (lpRegion, chars)

    def detectPlates(self):

        regions = []

        HSV = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        # cv2.imshow("HSV", HSV)

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

            # cv2.imshow("Convex Hull", hullImage)
        return regions

    def detectCharacterCandidates(self, region):
        chars = []
        count1 = 0
        count2 = 0

        plate = perspective.four_point_transform(self.image, region)
        # cv2.imshow("Perspective Transform", imutils.resize(plate, width=400))

        HSV = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
        Upper = np.array([120, 255, 255])
        Lower = np.array([90, 127, 120])
        H = cv2.inRange(HSV, Lower, Upper)

        thresh = cv2.threshold(H, 100, 255, cv2.THRESH_BINARY_INV)[1]

        plate = imutils.resize(plate, width=400)
        thresh = imutils.resize(thresh, width=400)
        perChar = cv2.erode(thresh, None, iterations=1)
        # cv2.imshow("perChar", perChar)

        thresh = cv2.erode(thresh, None, iterations=3)
        # cv2.imshow("LP erode", thresh)

        (LPh, LPw) = thresh.shape[:2]
        rectangle = np.zeros((LPh, LPw), dtype="uint8")
        cv2.rectangle(rectangle, (20, 20), (LPw - 20, LPh - 20), 255, -1)
        # cv2.imshow("Rectangle", rectangle)

        bitwiseAnd = cv2.bitwise_and(rectangle, thresh)
        perBitwiseAnd = cv2.bitwise_and(rectangle, perChar)
        # cv2.imshow("LP Mask no erode", bitwiseAnd)
        # cv2.imshow("LP Mask eroded", perBitwiseAnd)

        bitwiseAnd = cv2.morphologyEx(bitwiseAnd.copy(), cv2.MORPH_OPEN, (7, 7))
        bitwiseAnd = cv2.morphologyEx(bitwiseAnd, cv2.MORPH_OPEN, (7, 7))
        bitwiseAnd = cv2.morphologyEx(bitwiseAnd, cv2.MORPH_OPEN, (7, 7))
        bitwiseAnd = cv2.dilate(bitwiseAnd, None, iterations=4)
        bitwiseAnd = cv2.threshold(bitwiseAnd, 10, 255, cv2.THRESH_BINARY)[1]
        # cv2.imshow("LP Mask dilate", bitwiseAnd)

        # labels = measure.label(thresh, neighbors=8, background=0)
        # charCandidates = np.zeros(thresh.shape, dtype="uint8")

        cnts = cv2.findContours(bitwiseAnd, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1]
        hullImage = np.zeros(bitwiseAnd.shape[:2], dtype="uint8")

        for c in cnts:

            hull = cv2.convexHull(c)

            cv2.drawContours(hullImage, [hull], -1, 255, -1)

        # cv2.imshow("Convex Hull", hullImage)

        labels = measure.label(hullImage, neighbors=8, background=0)
        charCandidates = np.zeros(hullImage.shape, dtype="uint8")

        count1 += 1

        for label in np.unique(labels):

            if label == 0:
                continue

            labelMask = np.zeros(hullImage.shape, dtype="uint8")
            labelMask[labels == label] = 255
            # cv2.imshow("Label Mask", labelMask)
            # cv2.waitKey(0)
            count2 += 1
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

            # cv2.imshow("Chose Convex Hull", charCandidates)
            charCandidates = segmentation.clear_border(charCandidates)
            cnts = cv2.findContours(charCandidates.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[1]

            lastBitwiseAnd = cv2.bitwise_and(charCandidates, perChar)
            # cv2.imshow("last BitwiseAnd", lastBitwiseAnd)

        # print("1,2:", count1, count2)
        return LicensePlate(success=len(cnts) == self.numChars, plate=plate, thresh=lastBitwiseAnd,
                            candidates=charCandidates)

    def scissor(self, lp):

        cnts = cv2.findContours(lp.candidates.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1]
        boxes = []
        chars = []

        for c in cnts:
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
            dX = min(self.minCharW, self.minCharW - boxW) // 2
            boxX -= dX
            boxW += (dX * 2)

            boxes.append((boxX, boxY, boxX + boxW, boxY + boxH))

        boxes = sorted(boxes, key=lambda b:b[0])

        for (startX, startY, endX, endY) in boxes:
            chars.append(lp.thresh[startY:endY, startX:endX])

        return chars

    @staticmethod
    def preprocessChar(char):

        cnts = cv2.findContours(char.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        if len(cnts) == 0:
            return None
        c = max(cnts, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)
        char = char[y:y + h, x:x + w]

        return char
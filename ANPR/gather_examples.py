from __future__ import print_function
from license_plate.license_plate import LicensePlateDetector
from imutils import paths
import traceback
import imutils
import numpy as np
import cv2
import os
import time
from picamera.array import PiRGBArray
from picamera import PiCamera


camera = PiCamera()
camera.resolution = (500, 500)
camera.framerate = 5
rawCapture = PiRGBArray(camera, size=(500, 500))

time.sleep(0.5)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	labels = []
	boxs = []

	image = frame.array
	rawCapture.truncate(0)
	if image.shape[1] > 640:
		image = imutils.resize(image, width=640)

	lpd = LicensePlateDetector(image, numChars=7)
	plates = lpd.detect()

	for (lpBox, chars) in plates:
		lpBox = np.array(lpBox).reshape((-1, 1, 2)).astype(np.int32)

		plate = image.copy()
		cv2.drawContours(plate, [lpBox], -1, (0, 255, 0), 2)
		cv2.imshow("License Plate", plate)

		for char in chars:
				cv2.imshow("Char", char)
				key = cv2.waitKey(0)

				if key == ord("`"):
					print("[IGNORING] {}".format(imagePath))
					continue

				key = chr(key).upper()
				dirPath = "{}/{}".format("train_char", key)

				if not os.path.exists(dirPath):
					os.makedirs(dirPath)

				count = counts.get(key, 1)
				path = "{}/{}.png".format(dirPath, str(count).zfill(5))
				cv2.imwrite(path, char)

				counts[key] = count + 1
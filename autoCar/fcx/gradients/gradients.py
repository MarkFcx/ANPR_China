import cv2
from skimage import morphology, img_as_float
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths


# image = cv2.imread('../../training_data_3/forward/00012.jpg')
imagePaths = list(paths.list_images("../../training_data_3/forward"))
counts = {}

for imagePath in imagePaths:

	# print("[EXAMINING] {}".format(imagePath))

    image = cv2.imread(imagePath)
    cv2.imshow('image', image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)[1]
    cv2.imshow('A', thresh)

    (maskH, maskW) = thresh.shape[:2]
    rectangle = np.zeros((maskH, maskW), dtype="uint8")
    cv2.rectangle(rectangle, (10, int(maskH/3)), (maskW - 10, maskH - 10), 255, -1)
    cv2.imshow("Rectangle", rectangle)

    bitwiseAnd = cv2.bitwise_and(rectangle, thresh)
    cv2.imshow('B', bitwiseAnd)

    bitwiseAnd = cv2.morphologyEx(bitwiseAnd, cv2.MORPH_OPEN, (7, 7))
    dilated = cv2.dilate(bitwiseAnd, None, iterations=3)
    cv2.imshow("C", dilated)

    cnts = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1]

    clone = image.copy()
    cv2.drawContours(clone, cnts, -1, (255, 255, 0), 2)
    cv2.imshow("D", clone)

    x = []
    area = 1

    for (i, cnt) in enumerate(cnts):
        max_area = cv2.contourArea(cnt)
        print("maxArea:{}".format(i), max_area)

        if max_area > area:
            x = []

            for (j, c) in enumerate(cnt):
                x.append(c[0][0])

            area = max_area
            No_cont = i

    x.sort()
    # print("x", x)
    left_point_x = min(x)
    right_point_x = max(x)
    # point_y = 0

    for (i, c) in enumerate(cnts[No_cont]):
        if c[0][0] == left_point_x:
            left_point_y = c[0][1]

        if c[0][0] == right_point_x:
            right_point_y = c[0][1]

    gradient = (left_point_y - right_point_y)/(right_point_x - left_point_x)

    if gradient >= 0.4:
        print("Forward")
        cv2.putText(image, "Forward", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 0), 2)
        # ser.write(chr(1).encode())

    elif gradient <= 0:
        print("Forward Left")
        cv2.putText(image, "Left", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 0), 2)
        # ser.write(chr(7).encode())

    elif gradient >= 0 and gradient <= 0.4:
        print("Forward Right")
        cv2.putText(image, "Right", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 0), 2)
        # ser.write(chr(6).encode())

    cv2.circle(image, (left_point_x, left_point_y), 5, (0, 0, 255), -1)
    cv2.circle(image, (right_point_x, right_point_y), 5, (0, 255, 0), -1)
    cv2.putText(image, "K: {}".format(gradient), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 0), 2)
    cv2.imshow("E", image)

    cv2.waitKey(0)
# USAGE
# python3 driver.py

from picamera.array import PiRGBArray
from picamera import PiCamera
from fcx.preprocessing import ImageToArrayPreprocessor
from fcx.preprocessing import SimplePreprocessor
from fcx.datasets import SimpleDatasetLoader
from fcx.utils.simple_obj_det import image_pyramid
from fcx.utils.simple_obj_det import sliding_window
from imutils.object_detection import non_max_suppression
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import serial
import time
import cv2

serial_port = "/dev/ttyACM0"
ser = serial.Serial(serial_port, 115200, timeout=1)

classLabels_cls = ["forward", "left", "right"]
classLabels_det = ["greenlight", "nothing", "people", "redlight"]
INPUT_SIZE = (250, 250)
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = (64, 64)
BATCH_SIZE = 32
# load the pre-trained network
print("[INFO] loading pre-trained network...")
model_cls = load_model("model/CompanyData_blackfloor_wh64_e100_b32_sgd005.hdf5")
model_det = load_model("model/detction_1.hdf5")
data_size = 64

camera = PiCamera()
camera.resolution = (500, 500)
camera.framerate = 5
rawCapture = PiRGBArray(camera, size=(500, 500))

time.sleep(0.5)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    labels = []
    boxs = []

    image = frame.array

    #cv2.imshow("Frame", image)
    rawCapture.truncate(0)
    resized = cv2.resize(image.copy(), (250, 250), interpolation=cv2.INTER_CUBIC)

    start = time.time()

    for (x, y, roi) in sliding_window(resized, WIN_STEP, ROI_SIZE):
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = model_det.predict(roi, batch_size=32).argmax(axis=1)
        label = classLabels_det[max(preds)]


        if label != "nothing":
            clone = resized.copy()
            boxs.append((x, y, x + 64, y + 64))
            labels.append(label)

    boxes = np.array(boxs)
    boxes = non_max_suppression(boxes)

    for (x0, y0, x1, y1) in boxes:
        cv2.rectangle(clone, (x0, y0), (x1, y1), (0, 0, 255), 2)

    if label != None:
        if  labels[0] == "redlight":
            print("redlight")
            ser.write(chr(0).encode())

        elif labels[0] == "greenlight":
            print("greenlight")
            ser.write(chr(1).encode())

        elif labels[0] == "people":
            print(people)
            ser.write(chr(0).encode())
    end = time.time()
    print("[INFO] detections took {:.4f} seconds".format(end - start))

    start_cls = time.time()
    gray = cv2.resize(image, (data_size, data_size), interpolation=cv2.INTER_AREA)
    gray = gray.astype("float") / 255.0
    gray = img_to_array(gray)
    gray = np.expand_dims(gray, axis=0)

    preds = model_cls.predict(gray, batch_size=32).argmax(axis=1)

    if classLabels_cls[max(preds)] == "forward":
        print("Forward")
        ser.write(chr(1).encode())

    elif classLabels_cls[max(preds)] == "left":
        print("Forward Left")
        ser.write(chr(7).encode())

    elif classLabels_cls[max(preds)] == "right":
        print("Forward Right")
        ser.write(chr(6).encode())

    end_cls = time.time()
    print("[INFO] class took {:.4f} seconds".format(end_cls - start_cls))

    cv2.imshow("Sliding Window", clone)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()



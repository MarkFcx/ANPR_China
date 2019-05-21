# USAGE
# python3 driver.py

from picamera.array import PiRGBArray
from picamera import PiCamera
from fcx.preprocessing import ImageToArrayPreprocessor
from fcx.preprocessing import SimplePreprocessor
from fcx.datasets import SimpleDatasetLoader
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import serial
import time
import cv2

serial_port = "/dev/ttyACM0"
ser = serial.Serial(serial_port, 115200, timeout=1)

classLabels = ["forward", "left", "right"]
# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model("model/CompanyData_blackfloor_wh64_e100_b32_sgd005.hdf5")
data_size = 64

camera = PiCamera()
camera.resolution = (500, 500)
camera.framerate = 15
rawCapture = PiRGBArray(camera, size=(500, 500))

time.sleep(0.1)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    image = frame.array

    #cv2.imshow("Frame", image)
    rawCapture.truncate(0)

    gray = cv2.resize(image, (data_size, data_size), interpolation=cv2.INTER_AREA)
    gray = gray.astype("float") / 255.0
    gray = img_to_array(gray)
    gray = np.expand_dims(gray, axis=0)

    preds = model.predict(gray, batch_size=32).argmax(axis=1)

    if classLabels[max(preds)] == "forward":
        print("Forward")
        ser.write(chr(1).encode())

    elif classLabels[max(preds)] == "left":
        print("Forward Left")
        ser.write(chr(7).encode())

    elif classLabels[max(preds)] == "right":
        print("Forward Right")
        ser.write(chr(6).encode())


    print("[INFO] putText...", preds)
    cv2.putText(image, "Label: {}".format(classLabels[max(preds)]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    print("[INFO] show...")
    cv2.imshow("Image", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()



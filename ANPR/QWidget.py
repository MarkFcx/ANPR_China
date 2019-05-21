from ui import Ui_MainWidget
from PyQt5.QtCore import QTimer
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog,QTabWidget
from PyQt5.QtGui import QPixmap, QImage
import sys
import cv2
import time

from license_plate.license_plate import LicensePlateDetector
from descriptors.ml import BlockBinaryPixelSum
import numpy as np
from imutils import paths
from imutils.video import VideoStream
import imutils
import pickle
from collections import namedtuple

from picamera.array import PiRGBArray
from picamera import PiCamera
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)



result = namedtuple("result", ["image",
                               "provincial_capital",
                               "char_1",
                               "char_2",
                               "char_3",
                               "char_4",
                               "char_5",
                               "char_6"])

class GUI(QTabWidget, Ui_MainWidget):
    def __init__(self):
        super(GUI, self).__init__()
        self.setupUi(self)
        self.CarNums = 0
        self.RemainingParkingSpace = 100

        self.timer_camera = QTimer()
        # self.cap = cv2.VideoCapture(0)
        self.vs =  VideoStream(usePiCamera = 1, framerate=5).start()
        time.sleep(2.0)
        self.OpenGateButton.clicked.connect(self.slotStart)
        self.CloseGateButton.clicked.connect(self.slotStop)
        self.timer_camera.start(100)
        self.timer_camera.timeout.connect(self.openFrame)

    def slotStart(self):
        self.ctrlG90(180)
        

    def slotStop(self):
        self.ctrlG90(90)

    def run(self, image):

        charModel = pickle.loads(open("model/char.cpickle", "rb").read())

        blockSizes = ((5, 5), (5, 10), (10, 5), (10, 10))
        desc = BlockBinaryPixelSum(targetSize=(30, 15), blockSizes=blockSizes)

        # read image
        # for imagePath in sorted(list(paths.list_images("../ANPR_image"))):
        #     image = cv2.imread(imagePath)

        # read camera
        # camera = cv2.VideoCapture("../ANPR_video/04.mp4")
        # while True:
        # (grabbed, image) = camera.read()
        # if not grabbed:
        #     break

        if image.shape[1] > 640:
            image = imutils.resize(image, width=640)

        lpd = LicensePlateDetector(image)
        plates = lpd.detect()

        count1 = 0
        count2 = 0
        text = ""

        for (lpBox, chars) in plates:
            lpBox = np.array(lpBox).reshape((-1, 1, 2)).astype(np.int32)

            count1 += 1

            for (i, char) in enumerate(chars):
                char = LicensePlateDetector.preprocessChar(char)
                if char is None:
                    continue

                features = desc.describe(char).reshape(1, -1)

                prediction = charModel.predict(features)[0]

                text += prediction.upper()
                # text.append(prediction.upper())

                # print("prediction", prediction)
                count2 += 1

            # print("1, 2:", count1, count2)

            M = cv2.moments(lpBox)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cv2.drawContours(image, [lpBox], -1, (0, 255, 0), 2)
            # cv2.putText(image, text, (cX - (cX // 5), cY - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            #             (0, 0, 255), 2)
        print("LP: ", text)

        # cv2.imshow("detectImage", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # result.append(image)
        # result.append(text)
        if text == "":
            text = "无000000"

        return result(image=image,
                      provincial_capital=text[0],
                      char_1=text[1],
                      char_2=text[2],
                      char_3=text[3],
                      char_4=text[4],
                      char_5=text[5],
                      char_6=text[6])

    def openFrame(self):
        LP = []
        
        image = self.vs.read()
        result = self.run(image)
        if result.provincial_capital != "无" and result.char_1 != "0":
            LP.append((result.provincial_capital,
                                    result.char_1,
                                    result.char_2,
                                    result.char_3,
                                    result.char_4,
                                    result.char_5,
                                    result.char_6))

            self.LicensePlateInformation_0.setText(result.provincial_capital)
            self.LicensePlateInformation_0.setAlignment(QtCore.Qt.AlignCenter)
            self.LicensePlateInformation_0.setStyleSheet("color:green")
            self.LicensePlateInformation_1.setText(result.char_1)
            self.LicensePlateInformation_1.setAlignment(QtCore.Qt.AlignCenter)
            self.LicensePlateInformation_1.setStyleSheet("color:red")
            self.LicensePlateInformation_2.setText(result.char_2)
            self.LicensePlateInformation_2.setAlignment(QtCore.Qt.AlignCenter)
            self.LicensePlateInformation_2.setStyleSheet("color:red")
            self.LicensePlateInformation_3.setText(result.char_3)
            self.LicensePlateInformation_3.setAlignment(QtCore.Qt.AlignCenter)
            self.LicensePlateInformation_3.setStyleSheet("color:red")
            self.LicensePlateInformation_4.setText(result.char_4)
            self.LicensePlateInformation_4.setAlignment(QtCore.Qt.AlignCenter)
            self.LicensePlateInformation_4.setStyleSheet("color:red")
            self.LicensePlateInformation_5.setText(result.char_5)
            self.LicensePlateInformation_5.setAlignment(QtCore.Qt.AlignCenter)
            self.LicensePlateInformation_5.setStyleSheet("color:red")
            self.LicensePlateInformation_6.setText(result.char_6)
            self.LicensePlateInformation_6.setAlignment(QtCore.Qt.AlignCenter)
            self.LicensePlateInformation_6.setStyleSheet("color:red")

            self.ctrlG90(180)
            self.RemainingParkingSpace = self.RemainingParkingSpace - 1

            self.LicensePlateInformation_7.setText("当前剩余车位："+str(self.RemainingParkingSpace))

        else:
            self.ctrlG90(90)

        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, bytesPerComponent = frame.shape
        bytesPerLine = bytesPerComponent * width
        q_image = QImage(frame.data, width, height, bytesPerLine,
                            QImage.Format_RGB888).scaled(self.LPVideo.width(), self.LPVideo.height())
        self.LPVideo.setPixmap(QPixmap.fromImage(q_image))

        return LP

    def ctrlG90(self, angle):
        pwm = GPIO.PWM(17, 50)
        pwm.start(8)
        dutyCycle = angle/ 18. + 3.
        pwm.ChangeDutyCycle(dutyCycle)
        time.sleep(1)
        pwm.stop()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = GUI()
    window.show()
    sys.exit(app.exec_())

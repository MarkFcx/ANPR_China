import cv2

class Preprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.heiht = height
        self.inter = inter

    def preprocess(self, image):
        return cv2.resize(image, (self.width, self.heiht), interpolation=self.inter)
from __future__ import print_function
from license_plate.license_plate import LicensePlateDetector
from descriptors.ml import BlockBinaryPixelSum
from sklearn.svm import LinearSVC
from imutils import paths
import pickle
import glob
import cv2

blockSizes = ((5, 5), (5, 10), (10, 5), (10, 10))
desc = BlockBinaryPixelSum(targetSize=(30, 15), blockSizes=blockSizes)

Data = []
Labels = []

for samplePath in sorted(glob.glob("train_char" + "/*")):

    sampleName = samplePath[samplePath.rfind("/") + 1:]
    imagePaths = list(paths.list_images(samplePath))

    for imagePath in imagePaths:
        char = cv2.imread(imagePath)
        char = cv2.cvtColor(char, cv2.COLOR_BGR2GRAY)
        char = LicensePlateDetector.preprocessChar(char)
        features = desc.describe(char)

        Data.append(features)
        Labels.append(sampleName)

print("[INFO] fitting character model...")
charModel = LinearSVC(C=1.0, random_state=42)
charModel.fit(Data, Labels)

f = open("model/char.cpickle", "wb")
f.write(pickle.dumps(charModel))
f.close()

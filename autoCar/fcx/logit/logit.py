from __future__ import print_function
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from fcx.preprocessing import ImageToArrayPreprocessor
from fcx.preprocessing import SimplePreprocessor
from fcx.datasets import SimpleDatasetLoader
from imutils import paths
import numpy as np
import imutils
import cv2
import sklearn

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

data_size = 64
sp = SimplePreprocessor(data_size, data_size)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, random_state=42)


print("[INFO] training model...")
model = LogisticRegression()
model.fit(trainX, trainY)

print("[INFO] serializing network...")
model.save(args["model"])



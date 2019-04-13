from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from fcx.preprocessing.Preprocessor import Preprocessor
from fcx.preprocessing.ImageToArray import ImageToArray
from fcx.datasets.DatasetLoader import DatasetLoader
from fcx.nn import conv_1_layer

from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np

print("[INFO] loading images...")
imagePaths = list(paths.list_images("training_images/datasets"))

rs = Preprocessor(32, 32)
# 1.导包需要连同文件名字加上, 哎
i2a = ImageToArray()

dl = DatasetLoader(preprocessors=[rs, i2a])
(data, labels) = dl.load(imagePaths, verbose=100)
data = data.astype("float") / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] compiling model ...")
opt = SGD(lr=0.005)
model = conv_1_layer.build(width=32, height=32, depth=3, classes=3)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] train network ...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=1)

print("[INFO] evaluating network ...")
# predictions = model.predict(testY, batch_size=32)
# 3. 错误3又是打错了，哎->testY, 应该是testX。

predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=["forward", "left", "right"]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
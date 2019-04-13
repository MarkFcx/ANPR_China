import numpy as np
import cv2
import os

class DatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        data = []
        labels = []
        
        for (i, imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            # label = imagePaths.split(os.path.sep)[-2]
            # 2. 错误在imagePaths, 一个路径列表怎么分割啊。哎

            if self.preprocessors is not None:

                for p in self.preprocessors:
                    image = p.preprocess(image)

            data.append(image)
            labels.append(label)

            if verbose > 0 and i > 0 and (i + i) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))

        return (np.array(data), np.array(labels))


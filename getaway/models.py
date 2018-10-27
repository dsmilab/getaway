from keras.models import load_model
from PIL import Image
import numpy as np


class CNN:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def reload_model(self, model_path):
        self.model = load_model(model_path)

    def summary(self):
        return self.model.summary()

    def predict(self, img):
        if self.model is None:
            print("haven't load model yet.")

        img = Image.fromarray(np.uint8(img))
        img = img.reshape(-1, 360, 240, 1)
        print(img.shape)
        result = self.model.predict(img)
        print(result)

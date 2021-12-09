from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from tensorflow.python.keras.applications.efficientnet import EfficientNetB0 as  PretrainedModel, preprocess_input

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np



class FeatureExtractor(object):
    def __init__(self, imageShape = [200,200,3], weights = 'imagenet' ):
        self.imageShape = imageShape

        self.ptm = PretrainedModel(
        input_shape = imageShape,
        weights = weights,
        include_top = False
            )

        self.ptm.trainable = False
        x = Flatten()(self.ptm.output)
        self.model = Model(inputs = self.ptm.input , outputs = x)
        

    def extract(self,image):
        image = np.expand_dims(cv2.resize(image, (self.imageShape[0] , self.imageShape[1])) , 0)
        result = self.model.predict(image).squeeze()
        return result


if __name__ == "__main__":
    extractor = FeatureExtractor()
    image  = cv2.imread("images/car.jpg")
    print(extractor.extract(image))

    

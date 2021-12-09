from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.keras.applications.efficientnet import EfficientNetB0 as  PretrainedModel,preprocess_input

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np

IMAGE_SIZE = [200,200]

ptm = PretrainedModel(
        input_shape = IMAGE_SIZE + [3],
        weights = 'imagenet',
        include_top = False

)

ptm.trainable = False

x = Flatten()(ptm.output)
model = Model(inputs = ptm.input , outputs = x)
print(model.summary())
image = cv2.imread("images/car.jpg")
image2 = cv2.imread("images/car2.jpeg")
image = np.expand_dims(cv2.resize(image, (200,200)) , 0)
image2 = cv2.resize(image2, (200,200)).reshape(1,200,200,3)

data = model.predict(image).squeeze()
data2 = model.predict(image2).squeeze()

print(np.mean(data-data2))




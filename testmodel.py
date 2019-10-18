import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras import models, layers
import numpy as np

model1 = models.Sequential()
model1.add(layers.Conv2D(32, 3, input_shape=(32, 32, 3)))
model1.add(layers.BatchNormalization())
model1.add(layers.Dense(2, activation='softmax'))
model1.save_weights('test.h5')

model2 = models.Sequential()
model2.add(layers.Conv2D(32, 3, input_shape=(32, 32, 3)))
model2.add(layers.Dense(2, activation='softmax'))

model2.layers[0].set_weights(model1.layers[0].get_weights())
model2.layers[1].set_weights(model1.layers[2].get_weights())

x = np.random.rand(1, 32, 32, 3).astype('float32')
y1 = model1.predict(x)
y2 = model2.predict(x)

print(y2 == y1)

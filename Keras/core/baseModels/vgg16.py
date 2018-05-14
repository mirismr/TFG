'''
# Reference:
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
'''

import numpy as np
from keras.layers import Flatten, Dense, Input, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.data_utils import get_file
from keras.models import Sequential
from classes import modelType

WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
class VGG16(object):
	def __init__(self):
		self.baseModel = Sequential()
		self.topModel = None
		self.topType = None

		#Block1
		layer = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=(224,224,3))
		self.baseModel.add(layer)
		layer = Conv2D(64, (3, 3),  activation='relu', padding='same', name='block1_conv2')
		self.baseModel.add(layer)
		layer = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')
		self.baseModel.add(layer)

		# Block2
		layer = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')
		self.baseModel.add(layer)
		layer = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')
		self.baseModel.add(layer)
		layer = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')
		self.baseModel.add(layer)

		# Block 3
		layer = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')
		self.baseModel.add(layer)
		layer = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')
		self.baseModel.add(layer)
		layer = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')
		self.baseModel.add(layer)
		layer = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')
		self.baseModel.add(layer)
		
		# Block 4
		layer = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')
		self.baseModel.add(layer)
		layer = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')
		self.baseModel.add(layer)
		layer = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')
		self.baseModel.add(layer)
		layer = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')
		self.baseModel.add(layer)
		
		# Block 5
		layer = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')
		self.baseModel.add(layer)
		layer = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')
		self.baseModel.add(layer)
		layer = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')
		self.baseModel.add(layer)
		layer = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')
		self.baseModel.add(layer)

		weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
										WEIGHTS_PATH_NO_TOP,
										cache_subdir='models')
		self.baseModel.load_weights(weights_path)

	def buildTopModel(self, topType, numClasses):
		self.topModel = Sequential()

		if isinstance(topType, str):
			self.topType = modelType[topType]
		else:
			self.topType = topType

		if self.topType == modelType.typeA:
			layer = Flatten(name='flatten', input_shape=self.baseModel.output_shape[1:])
			self.topModel.add(layer)
			layer = Dense(4096, activation='relu', name='fc1')
			self.topModel.add(layer)
			layer = Dense(4096, activation='relu', name='fc2')
			self.topModel.add(layer)
			layer = Dense(numClasses, activation='softmax', name='predictions')
			self.topModel.add(layer)
		
		elif self.topType == modelType.typeB:
			layer = Flatten(name='flatten', input_shape=self.baseModel.output_shape[1:])
			self.topModel.add(layer)
			layer = Dense(256, activation='relu',name='fc1')
			self.topModel.add(layer)
			layer = Dropout(0.5, name='dropout')
			self.topModel.add(layer)
			layer = Dense(numClasses, activation='softmax', name='predictions')
			self.topModel.add(layer)


if __name__ == '__main__':
 
	model = VGG16()     
	print(model.summary())
	print(len(model.layers))

   
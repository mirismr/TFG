'''
# Reference:
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
'''
import numpy as np
from keras.layers import Flatten, Dense, AveragePooling2D, Dropout, Activation, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Merge, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.utils.data_utils import get_file
from keras import layers
from keras.models import Sequential, Model
from classes import modelType
import keras

WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

class InceptionV3(object):
    def __init__(self):
        self.topType = None
        self.topModel = None       

        # Create model.
        self.baseModel = keras.applications.inception_v3.InceptionV3(weights='imagenet', input_shape=(299,299,3), include_top=False)


    def buildTopModel(self, topType, numClasses):
        if isinstance(topType, str):
            self.topType = modelType[topType]
        else:
            self.topType = topType

        x = GlobalAveragePooling2D()(self.baseModel.output)

        if self.topType == modelType.typeA:
            # Classification block       
            x = Dense(512, activation='relu')(x)
            x = Dense(numClasses, activation='softmax', name='predictions')(x)        
       
        elif self.topType == modelType.typeB:
            x = Dense(256, activation='softmax', name='final_dense')(x)        
            x = Dense(numClasses, activation='softmax', name='predictions')(x) 

        elif self.topType == modelType.typeC:
            x = Dense(4096, activation='softmax', name='final_dense')(x)        
            x = Dense(numClasses, activation='softmax', name='predictions')(x)

        self.topModel = Model(self.baseModel.input, x)

        for layer in self.baseModel.layers:
            layer.trainable = False


#if __name__ == '__main__':
#    model = InceptionV3("a", 10)
#    print(model.baseModel.summary())
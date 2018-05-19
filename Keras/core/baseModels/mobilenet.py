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

class Mobilenet(object):
    def __init__(self):
        self.topType = None
        self.topModel = None       

        # Create model.
        self.baseModel = keras.applications.mobilenet.MobileNet(weights='imagenet', input_shape=(224,224,3), include_top=False)


    def buildTopModel(self, topType, numClasses):
        if isinstance(topType, str):
            self.topType = modelType[topType]
        else:
            self.topType = topType

        x = GlobalAveragePooling2D()(self.baseModel.output)

        if self.topType == modelType.typeA:
            # Classification block       
            shape = (1, 1, int(1024 * 1.0))

            x = GlobalAveragePooling2D()(x)
            x = Reshape(shape, name='reshape_1')(x)
            x = Dropout(1e-3, name='dropout')(x)
            x = Conv2D(numClasses, (1, 1), padding='same', name='conv_preds')(x)
            x = Activation('softmax', name='act_softmax')(x)
            x = Reshape((numClasses,), name='reshape_2')(x)

            #x = Dense(512, activation='relu')(x)
            #x = Dense(numClasses, activation='softmax', name='predictions')(x)        
       
        elif self.topType == modelType.typeB:
            x = Dense(256, activation='softmax', name='final_dense')(x)        
            x = Dense(numClasses, activation='softmax', name='predictions')(x) 

        elif self.topType == modelType.typeC:
            x = Dense(4096, activation='softmax', name='final_dense')(x)        
            x = Dense(numClasses, activation='softmax', name='predictions')(x)

        self.topModel = Model(self.baseModel.input, x)

        for layer in self.baseModel.layers:
            layer.trainable = False
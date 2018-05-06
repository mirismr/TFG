# -*- coding: utf-8 -*-
'''VGG16 model for Keras.

# Reference:

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

'''
from __future__ import print_function

import numpy as np
import warnings

from keras.models import Model
from keras.layers import Flatten, Dense, Input, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras import applications
from keras.utils.data_utils import get_file
from keras import backend as K
from enum_models import enum_models
from keras.models import Sequential

WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

def VGG16(type_top=enum_models.typeA, fine_tune=False, num_classes=10):
    '''Instantiate the VGG16 architecture,
    optionally loading weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        type_top: type of top to include (typeA, typeB, typeC)
        fine_tune: use fine_tune or not
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        num_classes: number of clases that we want to train
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.

    # Returns
        A Keras model instance.
    '''
    
    model = Sequential()
    
    #Block1
    layer = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=(224,224,3))
    model.add(layer)
    layer = Conv2D(64, (3, 3),  activation='relu', padding='same', name='block1_conv2')
    model.add(layer)
    layer = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')
    model.add(layer)

    # Block2
    layer = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')
    model.add(layer)
    layer = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')
    model.add(layer)
    layer = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')
    model.add(layer)

    #modelo hay que guardarlo como h5 (model.save)
    #en java MultiLayerNetwork network = KerasModelImport.importKerasSequentialModelAndWeights("PATH TO YOUR H5 FILE")
    # Block 3
    layer = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')
    model.add(layer)
    layer = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')
    model.add(layer)
    layer = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')
    model.add(layer)
    layer = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')
    model.add(layer)
    
    # Block 4
    layer = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')
    model.add(layer)
    layer = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')
    model.add(layer)
    layer = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')
    model.add(layer)
    layer = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')
    model.add(layer)
    
    # Block 5
    layer = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')
    model.add(layer)
    layer = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')
    model.add(layer)
    layer = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')
    model.add(layer)
    layer = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')
    model.add(layer)

    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
    model.load_weights(weights_path)

    return model


if __name__ == '__main__':

    model = VGG16(type_top=enum_models.typeA, fine_tune=False, num_classes=3)     
    print(model.summary())

   
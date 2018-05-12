'''
# Reference:
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
'''

import numpy as np
from keras.layers import Flatten, Dense, Input, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.data_utils import get_file
from keras.models import Sequential

WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

def VGG16():
    '''
    Instantiate the VGG16 architecture loading weights pre-trained on ImageNet. 
    
    Parameters
    ----------
    None

    Returns
    -------
    model : Sequential
        A Keras sequential instance.
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
 
    model = VGG16()     
    print(model.summary())
    print(len(model.layers))

   
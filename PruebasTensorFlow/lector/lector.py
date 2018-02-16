from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#import furbito as figura


#constantes
IMAGE_HEIGHT=56
IMAGE_WIDTH=56


#funcion a aplicar con map
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    #¿resized o cropped?
    image_resized = tf.image.resize_images(image_decoded, [IMAGE_HEIGHT, IMAGE_WIDTH], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image_resized, label

#devuelve un dataset con las imagenes y los labels
def read_images(filenames, labels):
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)

    return dataset

#devuelve lista con los nombres de la imagen con formato:directory/imagen.JPEG
def get_images_name(directory):
    name_list = os.listdir(directory)
    #depende del directorio desde donde se lance el programa habra que cambiar la formacion de la ruta
    name_list = [directory + '/' + name for name in name_list]

    return name_list

def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


#########################################################################

filenames = tf.constant(get_images_name('images'))
# `labels[i]` is the label for the image in `filenames[i].
labels = tf.constant([0, 0])

dataset = read_images(filenames, labels)

sess = tf.Session()
list_images = []
list_labels = []
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
while True:
    try:
        image, label = sess.run(next_element)
        list_images.append(image)
        list_labels.append(label)
    except tf.errors.OutOfRangeError:
        break

show_images(list_images, titles=list_labels)
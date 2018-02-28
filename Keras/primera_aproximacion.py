#https://keras.io/preprocessing/image/
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np

#devuelve lista con los nombres de la imagen con formato:directory/imagen.JPEG
def get_images_name(directory):
    name_list = os.listdir(directory)
    #depende del directorio desde donde se lance el programa habra que cambiar la formacion de la ruta
    name_list = [directory + '/' + name for name in name_list]

    return name_list

#obtiene las etiquetas asociadas a un synset dado su wnid
def get_words(wnid):
    link = "http://www.image-net.org/api/text/wordnet.synset.getwords?wnid="+wnid
    import urllib.request
    data = urllib.request.urlopen(link)

    words = ''
    for line in data:
        line = line.decode("utf-8")
        words=words+line

    words = words.split('\n')
    words.pop()

    string = ''
    for x in words:
        string=string+x+", "

    string = string[:-2]

    return string

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

def build_model(width, height, channels, n_classes):
    model = Sequential()
    #input_shape tamaño imagen y canales
    model.add(Conv2D(32, (3, 3), input_shape=(height, width, channels)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # the model so far outputs 3D feature maps (height, width, features)
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(n_classes))
    model.add(Activation('softmax'))


    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

    return model


def train_and_validate(model, batch_size, path_train, path_val, height, width, file_save_model, file_dictionary):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
            path_train,  # this is the target directory
            target_size=(height, width),  # all images will be resized to 64x64
            batch_size=batch_size,
            class_mode='categorical')  


    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            path_val,
            target_size=(height, width),
            batch_size=batch_size,
            class_mode='categorical')

    model.fit_generator(
            train_generator,
            steps_per_epoch=1000 // batch_size,
            epochs=10,
            validation_data=validation_generator,
            validation_steps=800 // batch_size)

    model.save_weights(file_save_model)
    class_dictionary = train_generator.class_indices

    #str para que sea compatible con map_class
    class_dictionary = {str(val):str(key) for (key, val) in class_dictionary.items()}

    import json     
    json = json.dumps(class_dictionary)
    f = open(file_dictionary,"w")
    f.write(json)
    f.close()

    return class_dictionary

def load_model(model, file_model, file_dictionary):
    model.load_weights(file_model, by_name=False)
    import json

    with open(file_dictionary) as json_data:
        d = json.load(json_data)
        return d


def predict_image(model, img_path, hot_encoding=False):
    img = load_img(img_path)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    if (hot_encoding):
        prediction = model.predict(x)
    else:
        prediction = model.predict_classes(x)

    return prediction

def map_class(prediction, dictionary):
    return dictionary[str(prediction[0])]


###############################################################################

model = build_model(64,64,3,3)
#class_dictionary = train_and_validate(model, 16, 'data/train', 'data/val', 64, 64, 'model_exported.h5', 'class_dictionary.json')
class_dictionary = load_model(model, 'model_exported.h5', 'class_dictionary.json')
print(class_dictionary)

img = 'data/predict/n01443537_203.JPEG'
img2 = 'data/predict/n01629819_409.JPEG'
img3 = 'data/predict/n02094433_494.JPEG'

print("Imagen ",img)
print("Prediccion: ",map_class(predict_image(model, img), class_dictionary))
print("Imagen ",img2)
print("Prediccion: ",map_class(predict_image(model, img2), class_dictionary))
print("Imagen ",img3)
print("Prediccion: ",map_class(predict_image(model, img3), class_dictionary))

# calculate predictions de un directorio completo
#las imagenes de cada clase deben estar en un 
#subdirectorio con el label como nombre del directorio
'''
pred_datagen = ImageDataGenerator(rescale=1./255)

pred_generator = pred_datagen.flow_from_directory(
        'data/predict',
        target_size=(64, 64),
        batch_size=1,
        shuffle=False,
        class_mode='categorical')


filenames = pred_generator.filenames
nb_samples = len(filenames)
prediction = model.predict_generator(pred_generator,steps = nb_samples)
print(filenames)
print(prediction)
'''



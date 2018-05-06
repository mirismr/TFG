import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input
from keras import applications
from keras import optimizers
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from keras import backend as K
import gc
import matplotlib.pyplot as plt
import math
import cv2


def load_data(path, target_size):
    from keras.preprocessing import image
    from keras.applications.vgg16 import preprocess_input
    import os
    data = []
    labels = []
    class_dictionary = {}

    print('Read data images')
    #folders = ['n02085374', 'n02121808', 'n02159955', 'n01910747', 'n02317335', 'n01909906', 'n01922303']
    folders = ['n02085374', 'n02121808', 'n02159955']

    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        class_dictionary[index] = fld

        path_sysnet = path+fld+"/"
        files = os.listdir(path_sysnet)
        for fl in files:
            img = image.load_img(path_sysnet+fl, target_size=target_size)
            img = image.img_to_array(img)
            #img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            data.append(img)
            labels.append(index)

    import json  
    contentJson = json.dumps(class_dictionary, indent=4, sort_keys=True)
    f = open('class_dictionary.json', "w")
    f.write(contentJson)
    f.close()


    return np.array(data), np.array(labels)


def k_fold_cross_validation(data, labels):

    folds = list(StratifiedKFold(n_splits=5, shuffle=False).split(data, labels))
    
    for j, (train_idx, val_idx) in enumerate(folds):
        
        print('\nFold ',j)
        x_train = data[train_idx]
        y_train = labels[train_idx]
        x_valid = data[val_idx]
        y_valid = labels[val_idx]

        # prepare data augmentation configuration
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)       
        generator = train_datagen.flow(x_train, y_train, shuffle=False, batch_size = batch_size)

        #constantes
        nb_train_samples = len(train_idx)
        num_classes = len(np.unique(labels))
        predict_size_train = int(math.ceil(nb_train_samples / batch_size))

        model = applications.VGG16(include_top=False, weights='imagenet')
        

        bottleneck_features_train = model.predict_generator(generator, predict_size_train, verbose=1)
        np.save('bottlenecks/bottleneck_features_train_fold_'+str(j)+'.npy', bottleneck_features_train)


        test_datagen = ImageDataGenerator(rescale=1. / 255)
        generator = test_datagen.flow(x_valid, y_valid, shuffle=False, batch_size = batch_size)
        nb_validation_samples = len(val_idx)
        predict_size_validation = int(math.ceil(nb_validation_samples / batch_size))

        bottleneck_features_validation = model.predict_generator(generator, predict_size_validation, verbose=1)
        np.save('bottlenecks/bottleneck_features_validation_fold_'+str(j)+'.npy',bottleneck_features_validation)

        ################################################top model###############################################

        train_data = np.load('bottlenecks/bottleneck_features_train_fold_'+str(j)+'.npy')
        train_labels = y_train
        train_labels = to_categorical(train_labels, num_classes=num_classes)

        validation_data = np.load('bottlenecks/bottleneck_features_validation_fold_'+str(j)+'.npy')
        validation_labels = y_valid
        validation_labels = to_categorical(validation_labels, num_classes=num_classes)
        
        inputs = Input(shape=train_data.shape[1:])
        x = Flatten(name='flatten')(inputs)
        x = Dense(256, activation='relu') (x)
        x = Dropout(0.5)(x)
        x = Dense(num_classes, activation='softmax')(x)
        top_model = Model(inputs=inputs, outputs=x, name='top_model')

        top_model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy', metrics=['accuracy'])

        historyGenerated = top_model.fit(train_data, train_labels,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(validation_data, validation_labels))
        

        top_model.save('top_model/top_model_exported_fold_'+str(j)+'.h5')

        import json  
        contentJson = json.dumps(historyGenerated.history,indent=4, sort_keys=True)
        f = open('top_model/top_model_history_fold_'+str(j)+'.json', "w")
        f.write(contentJson)
        f.close()

        ###################################test data######################################################
        # build the VGG16 network revisar, para la parte del top, el base_model tiene que ser losbottlenecks
        base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
        print('Model loaded.')

        inputs = Input(shape=base_model.output_shape[1:])
        x = Flatten(name='flatten') (inputs)
        x = Dense(256, activation='relu') (x)
        x = Dropout(0.5)(x)
        x = Dense(num_classes, activation='softmax')(x)

        top_model = Model(inputs=inputs, outputs=x, name='top_model')
        # note that it is necessary to start with a fully-trained
        # classifier, including the top classifier,
        # in order to successfully do fine-tuning
        top_model.load_weights('top_model/top_model_exported_fold_'+str(j)+'.h5')

        # add the model on top of the convolutional base
        model = Model(input=base_model.input, output=top_model(base_model.output))
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        generator_test = test_datagen.flow(x_valid, validation_labels, shuffle=False, batch_size = batch_size)

        nb_test_samples = nb_validation_samples

        test_loss, test_accuracy = model.evaluate_generator(generator_test, steps=nb_test_samples/batch_size)

        print("[INFO] TEST accuracy: {:.2f}%".format(test_accuracy * 100))
        print("[INFO] Test loss: {}".format(test_loss))

        import json  
        contentJson = json.dumps({'test_accuracy':test_accuracy, 'test_loss':test_loss, 'model':'top_model/top_model_exported_fold_'+str(j)+'.h5'})
        f = open('top_model/top_model_test_fold_'+str(j)+'.json', "w")
        f.write(contentJson)
        f.close()

        #guardar modelo final para importar
        json_string = model.to_json()
        f = open('top_model/final_top_model_fold_'+str(j)+'.json', "w")
        f.write(json_string)
        f.close()

        #liberar memoria
        del model
        del x_train
        del y_train
        del x_valid
        del y_valid
        K.clear_session()
        gc.collect()



data, labels = load_data('/home/mirismr/Descargas/aws/', (224, 224))
k_fold_cross_validation(data, labels)


K.clear_session()









































































# dimensions of our images.
img_width, img_height = 224, 224

fine_tune_model = 'fine_tune_model.h5'
fine_tune_json_weights = 'fine_tune_weights.json'
history_path_fine_tune = 'fine_tune_log.json'
history_path_top = 'top_log.json'
data_test_path = 'data_test.json'
train_data_dir = '/home/mirismr/Descargas/mini-aws/train'
validation_data_dir = '/home/mirismr/Descargas/mini-aws/val'
test_data_dir = '/home/mirismr/Descargas/mini-aws/test'


# number of epochs to train top model
epochs = 10
# batch size used by flow_from_directory and predict_generator
batch_size = 16


def save_bottleneck_features():
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    print(len(generator.filenames))
    print(generator.class_indices)
    print(len(generator.class_indices))

    nb_train_samples = len(generator.filenames)
    num_classes = len(generator.class_indices)

    predict_size_train = int(math.ceil(nb_train_samples / batch_size))

    bottleneck_features_train = model.predict_generator(
        generator, predict_size_train, verbose=1)

    np.save('bottleneck_features_train.npy', bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator.filenames)

    predict_size_validation = int(
        math.ceil(nb_validation_samples / batch_size))

    bottleneck_features_validation = model.predict_generator(
        generator, predict_size_validation, verbose=1)

    np.save('bottleneck_features_validation.npy',
            bottleneck_features_validation)


def train_top_model():
    datagen_top = ImageDataGenerator(rescale=1. / 255)
    generator_top = datagen_top.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    nb_train_samples = len(generator_top.filenames)
    num_classes = len(generator_top.class_indices)

    # save the class indices to use use later in predictions
    np.save('class_indices.npy', generator_top.class_indices)

    # load the bottleneck features saved earlier
    train_data = np.load('bottleneck_features_train.npy')

    # get the class lebels for the training data, in the original order
    train_labels = generator_top.classes

    # https://github.com/fchollet/keras/issues/3467
    # convert the training labels to categorical vectors
    train_labels = to_categorical(train_labels, num_classes=num_classes)

    generator_top = datagen_top.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator_top.filenames)

    validation_data = np.load('bottleneck_features_validation.npy')

    validation_labels = generator_top.classes
    validation_labels = to_categorical(
        validation_labels, num_classes=num_classes)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    historyGenerated = model.fit(train_data, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_data, validation_labels))

    model.save_weights(top_model_weights_path)

    import json  
    contentJson = json.dumps(historyGenerated.history)
    f = open(history_path_top, "w")
    f.write(contentJson)
    f.close()

    plt.figure(1)

    # summarize historyGenerated for accuracy

    plt.subplot(211)
    plt.plot(historyGenerated.history['acc'])
    plt.plot(historyGenerated.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss

    plt.subplot(212)
    plt.plot(historyGenerated.history['loss'])
    plt.plot(historyGenerated.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def fine_tuning():

    # build the VGG16 network
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
    print('Model loaded.')

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(3, activation='softmax'))

    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    top_model.load_weights(top_model_weights_path)

    # add the model on top of the convolutional base
    model = Model(input=base_model.input, output=top_model(base_model.output))

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:25]:
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')


    nb_train_samples = len(train_generator.filenames)
    nb_validation_samples = len(validation_generator.filenames)

    # fine-tune the model
    historyGenerated = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples/batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples/batch_size)

    model.save_weights(fine_tune_model)

    import json  
    contentJson = json.dumps(historyGenerated.history)
    f = open(history_path_fine_tune, "w")
    f.write(contentJson)
    f.close()

#test para el modelo normal, no el fine tune
def test():
    # build the VGG16 network
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
    print('Model loaded.')

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(3, activation='softmax'))


    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    top_model.load_weights(top_model_weights_path)

    # add the model on top of the convolutional base
    model = Model(input=base_model.input, output=top_model(base_model.output))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    #test data
    datagen_test = ImageDataGenerator(rescale=1. / 255)
    generator_test = datagen_test.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    nb_test_samples = len(generator_test.filenames)

    test_loss, test_accuracy = model.evaluate_generator(generator_test, steps=nb_test_samples/batch_size)

    print("[INFO] TEST accuracy: {:.2f}%".format(test_accuracy * 100))
    print("[INFO] Test loss: {}".format(test_loss))

    import json  
    contentJson = json.dumps({'test_accuracy':test_accuracy, 'test_loss':test_loss})
    f = open(data_test_path, "w")
    f.write(contentJson)
    f.close()


def predict():
    # load the class_indices saved in the earlier step
    class_dictionary = np.load('class_indices.npy').item()

    num_classes = len(class_dictionary)

    # add the path to your test image below
    #image_path = '/home/mirismr/Descargas/choosen/n02085374/n02086240/n02086240_3914.JPEG'
    #image_path = '/home/mirismr/Descargas/choosen/n01905661/n02317335/n02317335_82.JPEG'
    image_path = '/home/mirismr/Descargas/descarga3.jpg'
    #image_path = '/home/mirismr/Descargas/choosen/n01922303/n01922303/n01922303_64.JPEG'

    orig = cv2.imread(image_path)

    print("[INFO] loading and preprocessing image...")
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)

    # important! otherwise the predictions will be '0'
    #Remember that in our ImageDataGenerator we set rescale=1. / 255, which means all data is re-scaled from a [0 - 255] range to [0 - 1.0]. 
    image = image / 255

    image = np.expand_dims(image, axis=0)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    # get the bottleneck prediction from the pre-trained VGG16 model
    bottleneck_prediction = model.predict(image)

    # build top model
    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.load_weights(top_model_weights_path)

    # use the bottleneck prediction on the top model to get the final
    # classification
    class_predicted = model.predict_classes(bottleneck_prediction)

    probabilities = model.predict_proba(bottleneck_prediction)

    inID = class_predicted[0]

    inv_map = {v: k for k, v in class_dictionary.items()}

    label = inv_map[inID]

    # get the prediction label
    print("Image ID: {}, Label: {}".format(inID, label))
'''
    # display the predictions with the image
    cv2.putText(orig, "Predicted: {}".format(label), (10, 30),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2)

    cv2.imshow("Classification", orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
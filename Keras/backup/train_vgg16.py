import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input
from keras import optimizers
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from keras import backend as K
from vgg16 import VGG16
import gc
import matplotlib.pyplot as plt
import math
import cv2
from enum_models import enum_models

# number of epochs to train top model
epochs = 5
# batch size
batch_size = 16

def load_data(path, target_size):
    from keras.preprocessing import image 
    from keras.applications.vgg16 import preprocess_input
    import os
    data = []
    labels = []
    class_dictionary = {}

    print('Read data images')
    folders = ['n02085374', 'n02121808', 'n01910747','n02317335', 'n01922303', 'n01816887', 'n01859325', 'n02484322', 'n02504458', 'n01887787']

    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        class_dictionary[index] = fld

        path_sysnet = path+fld+"/"
        files = os.listdir(path_sysnet)
        for fl in files:
            img = image.load_img(path_sysnet+fl, target_size=target_size)
            img = image.img_to_array(img)
            img = preprocess_input(img)
            data.append(img)
            labels.append(index)

    import json  
    contentJson = json.dumps(class_dictionary, indent=4, sort_keys=True)
    f = open('class_dictionary.json', "w")
    f.write(contentJson)
    f.close()


    return np.array(data), np.array(labels)


def k_fold_cross_validation(data, labels, type_top):

    folds = list(StratifiedKFold(n_splits=5, shuffle=True).split(data, labels))
    
    for j, (train_idx, val_idx) in enumerate(folds):
        
        print('\nFold ',j)
        x_train = data[train_idx]
        y_train = labels[train_idx]
        x_valid = data[val_idx]
        y_valid = labels[val_idx]

        num_classes = len(np.unique(labels))

        # prepare data augmentation configuration
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)       
        generator = train_datagen.flow(x_train, y_train, shuffle=False, batch_size = batch_size)

        nb_train_samples = len(train_idx)
        num_classes = len(np.unique(labels))
        predict_size_train = int(math.ceil(nb_train_samples / batch_size))

        model = VGG16()
        
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
        
        top_model = Sequential()

        if type_top == enum_models.typeA:
            layer = Flatten(name='flatten', input_shape=train_data.shape[1:])
            top_model.add(layer)
            layer = Dense(4096, activation='relu', name='fc1')
            top_model.add(layer)
            layer = Dense(4096, activation='relu', name='fc2')
            top_model.add(layer)
            layer = Dense(num_classes, activation='softmax', name='predictions')
            top_model.add(layer)
        
        elif type_top == enum_models.typeB:
            layer = Flatten(name='flatten', input_shape=train_data.shape[1:])
            top_model.add(layer)
            layer = Dense(256, activation='relu',name='fc1')
            top_model.add(layer)
            layer = Dropout(0.5, name='dropout')
            top_model.add(layer)
            layer = Dense(num_classes, activation='softmax', name='predictions')
            top_model.add(layer)
        
        elif type_top == enum_models.typeC:
            layer = Flatten(name='flatten', input_shape=train_data.shape[1:])
            top_model.add(layer)
            layer = Dense(4096, activation='relu', name='fc1')
            top_model.add(layer)
            layer = Dense(4096, activation='relu', name='fc2')
            top_model.add(layer)
            layer = Dropout(0.5, name='dropout')
            top_model.add(layer)
            layer = Dense(num_classes, activation='softmax', name='predictions')
            top_model.add(layer)


        top_model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy', metrics=['accuracy'])

        import time
        t = time.process_time()
        historyGenerated = top_model.fit(train_data, train_labels,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(validation_data, validation_labels))
        
        training_time = time.process_time() - t

        ###################################test data######################################################
        model = VGG16()
        # build the final model
        for layer in top_model.layers:
           model.add(layer)

        test_datagen = ImageDataGenerator(rescale=1. / 255)
        generator_test = test_datagen.flow(x_valid, validation_labels, shuffle=False, batch_size = batch_size)

        nb_test_samples = nb_validation_samples

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        test_loss, test_accuracy = model.evaluate_generator(generator_test, steps=nb_test_samples/batch_size)

        #print("[INFO] TEST accuracy: {:.2f}%".format(test_accuracy * 100))
        #print("[INFO] Test loss: {}".format(test_loss))

        ####################################save info#######################################################
        

        top_model.save('top_model/top_model_exported_fold_'+str(j)+'.h5')

        #guardar modelo final para importar -> pa qué si esto es solo para decidir que top es mejor
        #model.save('top_model/final_model_exported_fold_'+str(j)+'.h5')

        import json  
        contentJson = json.dumps(historyGenerated.history, indent=4, sort_keys=True)
        f = open('top_model/top_model_history_fold_'+str(j)+'.json', "w")
        f.write(contentJson)
        f.close()

        #top_type meterlo aqui
        contentJson = json.dumps({'test_accuracy':test_accuracy, 'test_loss':test_loss,  'training_time': training_time, 'model':'top_model/top_model_exported_fold_'+str(j)+'.h5'})
        f = open('top_model/general_data_fold_'+str(j)+'.json', "w")
        f.write(contentJson)
        f.close()

        #liberar memoria
        del model
        del top_model
        del x_train
        del y_train
        del x_valid
        del y_valid
        K.clear_session()
        gc.collect()

def fine_tune(data, labels, best_top_model, type_top, layers_fine_tune):
    folds = list(StratifiedKFold(n_splits=5, shuffle=True).split(data, labels))
    
    for j, (train_idx, val_idx) in enumerate(folds):
        
        print('\nFold ',j)
        x_train = data[train_idx]
        y_train = labels[train_idx]
        x_valid = data[val_idx]
        y_valid = labels[val_idx]

        num_classes = len(np.unique(labels))

        model = VGG16()

        top_model = Sequential()

        if type_top == enum_models.typeA:
            layer = Flatten(name='flatten', input_shape=model.output_shape[1:])
            top_model.add(layer)
            layer = Dense(4096, activation='relu', name='fc1')
            top_model.add(layer)
            layer = Dense(4096, activation='relu', name='fc2')
            top_model.add(layer)
            layer = Dense(num_classes, activation='softmax', name='predictions')
            top_model.add(layer)
        
        elif type_top == enum_models.typeB:
            layer = Flatten(name='flatten', input_shape=model.output_shape[1:])
            top_model.add(layer)
            layer = Dense(256, activation='relu',name='fc1')
            top_model.add(layer)
            layer = Dropout(0.5, name='dropout')
            top_model.add(layer)
            layer = Dense(num_classes, activation='softmax', name='predictions')
            top_model.add(layer)
        
        elif type_top == enum_models.typeC:
            layer = Flatten(name='flatten', input_shape=model.output_shape[1:])
            top_model.add(layer)
            layer = Dense(4096, activation='relu', name='fc1')
            top_model.add(layer)
            layer = Dense(4096, activation='relu', name='fc2')
            top_model.add(layer)
            layer = Dropout(0.5, name='dropout')
            top_model.add(layer)
            layer = Dense(num_classes, activation='softmax', name='predictions')
            top_model.add(layer)

        top_model.load_weights(best_top_model)
        # build the final model
        for layer in top_model.layers:
           model.add(layer)

        for layer in model.layers[:layers_fine_tune]:
            layer.trainable = False

        model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

        train_labels = y_train
        train_labels = to_categorical(train_labels, num_classes=num_classes)
        validation_labels = y_valid
        validation_labels = to_categorical(validation_labels, num_classes=num_classes)

        # prepare data augmentation configuration
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)       
        train_generator = train_datagen.flow(x_train, train_labels, shuffle=False, batch_size = batch_size)

        #constantes
        nb_train_samples = len(train_idx)
        num_classes = len(np.unique(labels))
        predict_size_train = int(math.ceil(nb_train_samples / batch_size))

        test_datagen = ImageDataGenerator(rescale=1. / 255)
        validation_generator = test_datagen.flow(x_valid, validation_labels, shuffle=False, batch_size = batch_size)
        nb_validation_samples = len(val_idx)
        predict_size_validation = int(math.ceil(nb_validation_samples / batch_size))

        import time
        t = time.process_time()
        historyGenerated = model.fit_generator(
                            train_generator,
                            steps_per_epoch=predict_size_train,
                            epochs=epochs,
                            validation_data=validation_generator,
                            validation_steps=predict_size_validation)
        
        training_time = time.process_time() - t

        #este si porq el fine tune es el que usaremos
        model.save('fine_tune/final_model_exported_fold_'+str(j)+'.h5')

        import json  
        contentJson = json.dumps(historyGenerated.history,indent=4, sort_keys=True)
        f = open('fine_tune/fine_tune_history_fold_'+str(j)+'.json', "w")
        f.write(contentJson)
        f.close()
        ########################################test##############################################
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        generator_test = test_datagen.flow(x_valid, validation_labels, shuffle=False, batch_size = batch_size)

        nb_test_samples = nb_validation_samples
        test_loss, test_accuracy = model.evaluate_generator(generator_test, steps=nb_test_samples/batch_size)

        print("[INFO] TEST accuracy: {:.2f}%".format(test_accuracy * 100))
        print("[INFO] Test loss: {}".format(test_loss))

        import json  
        contentJson = json.dumps({'test_accuracy':test_accuracy, 'test_loss':test_loss, 'training_time':training_time, 'model':'fine_tune/fine_tune_exported_fold_'+str(j)+'.h5'})
        f = open('fine_tune/fine_tune_test_fold_'+str(j)+'.json', "w")
        f.write(contentJson)
        f.close()

        #liberar memoria
        del model
        del top_model
        del x_train
        del y_train
        del x_valid
        del y_valid
        K.clear_session()
        gc.collect()


data, labels = load_data('/home/mirismr/Descargas/data/', (224, 224))
k_fold_cross_validation(data, labels, enum_models.typeB)
#fine_tune(data, labels, 'top_model/top_model_exported_fold_0.h5',enum_models.typeB, 15)


K.clear_session()

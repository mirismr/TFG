from classes.logger import *
import classes.utils as utils
import numpy as np
from classes import modelType
from keras.preprocessing.image import ImageDataGenerator


class Model(object):
	"""
	Model that represent a concrete neuronal network.

	Attributes
	----------
	model : Model of baseModels
		Base model from my personal library.
	
	inputSize : tuple
		Base model's input size.

	logger : Logger
		Logger object that logs the train and test process.

	batchSize : int
		Batch's size.

	epochs : int
		Number of epochs that the model will train.

	numClasses : int
		Number of classes that the model will learn.

	Methods
	-------
	loadAndPreprocessData(path)
		Load and preprocess data for the appropiate Keras model.
	"""
	def __init__(self, model, inputSize, pathToModel):
		self.model = model #cuando se use hacer copia
		self.inputSize = inputSize
		self.logger = Logger(pathToModel)
		self.batchSize = 16
		self.epochs = 1
		self.numClasses = -1

		self.logger.createStructureLogs()

	def loadAndPreprocessData(self, path, log=True):
		"""
		Load and preprocess data for the appropiate Keras model.

		Parameters
		----------
		path : string
			Path where we found the data

		log : boolean
			True if we want log about loading process

		Returns
		-------
		data : numpy array
			Image data
		labels : numpy array
			Class for each image data
		"""
		from keras.preprocessing import image 
		from keras.applications.imagenet_utils import preprocess_input
		import os
		data = []
		labels = []
		classDictionary = {}

		if (log): 
			print('[INFO] Reading data images')
		folders = os.listdir(path)
		folders = ['n02085374', 'n02121808', 'n01910747']#,'n02317335', 'n01922303', 'n01816887', 'n01859325', 'n02484322', 'n02504458', 'n01887787']
		for fld in folders:
			index = folders.index(fld)
			if (log): 
				print('  Loading folder {} (Class: {})'.format(fld, index))
			classDictionary[index] = fld

			path_sysnet = path+fld+"/"
			files = os.listdir(path_sysnet)
			for fl in files:
				img = image.load_img(path_sysnet+fl, target_size=self.inputSize)
				img = image.img_to_array(img)
				img = preprocess_input(img)
				data.append(img)
				labels.append(index)

		self.numClasses = len(folders)
		self.logger.saveClassDictionary(classDictionary)
		return np.array(data), np.array(labels), len(folders)

	def generateBottlenecks(self, trainData, trainLabels, validData, validLabels, fold):
		import math

		print('[INFO] Generating bottlenecks')
			
		# Data augmentation
		trainDatagen = ImageDataGenerator(
			rescale=1. / 255,
			shear_range=0.2,
			zoom_range=0.2,
			horizontal_flip=True)       
		generator = trainDatagen.flow(trainData, trainLabels, shuffle=False, batch_size = self.batchSize)

		trainSamples = len(trainLabels)
		predictSizeTrain = int(math.ceil(trainSamples / self.batchSize))
		
		bottlenecksFeaturesTrain = self.model.baseModel.predict_generator(generator, predictSizeTrain, verbose=1)

		testDatagen = ImageDataGenerator(rescale=1. / 255)
		generator = testDatagen.flow(validData, validLabels, shuffle=False, batch_size = self.batchSize)
		
		validationSamples = len(validLabels)
		predictSizeValidation = int(math.ceil(validationSamples / self.batchSize))

		bottlenecksFeaturesValidation = self.model.baseModel.predict_generator(generator, predictSizeValidation, verbose=1)

		self.logger.saveBottlenecks(bottlenecksFeaturesTrain, bottlenecksFeaturesValidation, fold)

		del bottlenecksFeaturesValidation
		del bottlenecksFeaturesTrain

		import gc
		gc.collect()


	def buildTopModel(self, topType):
		self.model.buildTopModel(topType, self.numClasses)

	def trainTopModel(self, trainLabels, validLabels, fold):
		from keras import optimizers
		print('[INFO] Training top model type '+(str(self.model.topType)))
		trainData, validationData = self.logger.loadBottlenecks(fold)
		trainLabels = utils.transformLabelsToCategorical(trainLabels, self.numClasses)
		validationLabels = utils.transformLabelsToCategorical(validLabels, self.numClasses)

		self.model.topModel.compile(optimizer=optimizers.RMSprop(lr=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])

		import time
		t = time.process_time()
		historyGenerated = self.model.topModel.fit(trainData, trainLabels,
							epochs=self.epochs,
							batch_size=self.batchSize,
							validation_data=(validationData, validationLabels)
							)
		
		trainingTime = time.process_time() - t

		self.logger.saveWeightsTopModel(self.model, fold)

		return trainingTime, historyGenerated


	def addTopModel(self):
		from keras.models import Sequential

		fullModel = Sequential()
		for layer in self.model.baseModel.layers:
			fullModel.add(layer)

		for layer in self.model.topModel.layers:
		   fullModel.add(layer)

		return fullModel

	def testTopModel(self, validationData, validationLabels, trainingTime, historyGenerated, fold):
		print('[INFO] testing top model type '+(str(self.model.topType)))
		fullModel = self.addTopModel()
		validationSamples = len(validationLabels)

		validationLabels = utils.transformLabelsToCategorical(validationLabels, self.numClasses)
		
		testDatagen = ImageDataGenerator(rescale=1. / 255)
		generator = testDatagen.flow(validationData, validationLabels, shuffle=False, batch_size = self.batchSize)

		from keras import optimizers
		fullModel.compile(optimizer=optimizers.RMSprop(lr=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])
		testLoss, testAccuracy = fullModel.evaluate_generator(generator, steps=validationSamples/self.batchSize)

		self.logger.saveDataTopModel(historyGenerated, (testLoss, testAccuracy), trainingTime, fold, self.model.topType)

		import gc
		del fullModel
		gc.collect()

	def fineTune(self, trainData, trainLabels, validData, validLabels, numLayersFreeze, bestTopType, pathBestTop, fold):
		import math

		self.model.topModel = self.logger.loadWeights(self.model.topModel, pathBestTop)
		self.model.topType = modelType.modelType[bestTopType]

		fullModel = self.addTopModel()

		# Freeze layers
		for layer in fullModel.layers[:numLayersFreeze]:
			layer.trainable = False

		import tensorflow as tf
		run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

		from keras import optimizers
		fullModel.compile(loss='categorical_crossentropy',
			optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
			metrics=['accuracy'], options = run_opts)
		

		trainSamples = len(trainLabels)
		validationSamples = len(validLabels)
		predictSizeTrain = int(math.ceil(trainSamples / self.batchSize))
		predictSizeValidation = int(math.ceil(validationSamples / self.batchSize))
		
		trainLabels = utils.transformLabelsToCategorical(trainLabels, self.numClasses)
		validLabels = utils.transformLabelsToCategorical(validLabels, self.numClasses)

		print('[INFO] Fine tuning with '+str(numLayersFreeze)+' layers freeze.')
			
		# Data augmentation
		trainDatagen = ImageDataGenerator(
			rescale=1. / 255,
			shear_range=0.2,
			zoom_range=0.2,
			horizontal_flip=True)       
		generatorTrain = trainDatagen.flow(trainData, trainLabels, shuffle=False, batch_size = self.batchSize)

		valDatagen = ImageDataGenerator(rescale=1. / 255)
		generatorVal = valDatagen.flow(validData, validLabels, shuffle=False, batch_size = self.batchSize)
		
		import time
		t = time.process_time()
		historyGenerated = fullModel.fit_generator(
							generatorTrain,
							steps_per_epoch=predictSizeTrain,
							epochs=self.epochs,
							validation_data=generatorVal,
							validation_steps=predictSizeValidation)
		
		trainingTime = time.process_time() - t

		self.logger.saveWeightsFineTune(fullModel, fold, numLayersFreeze)

		
		import gc
		del fullModel
		gc.collect()
		
		return trainingTime, historyGenerated


	def testFineTuneModel(self, validationData, validationLabels, trainingTime, historyGenerated, fold, numLayersFreeze):
		print('[INFO] testing fine tune model')
		fullModel = self.addTopModel()
		validationSamples = len(validationLabels)

		validationLabels = utils.transformLabelsToCategorical(validationLabels, self.numClasses)
		
		testDatagen = ImageDataGenerator(rescale=1. / 255)
		generator = testDatagen.flow(validationData, validationLabels, shuffle=False, batch_size = self.batchSize)

		from keras import optimizers
		fullModel.compile(optimizer=optimizers.RMSprop(lr=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])
		testLoss, testAccuracy = fullModel.evaluate_generator(generator, steps=validationSamples/self.batchSize)

		self.logger.saveDataFineTuneModel(historyGenerated, (testLoss, testAccuracy), trainingTime, fold, numLayersFreeze, self.model.topType)

		import gc
		del fullModel
		gc.collect()
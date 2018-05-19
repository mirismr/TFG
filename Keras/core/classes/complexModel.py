from classes.logger import *
import classes.utils as utils
import numpy as np
from classes import modelType
from keras.preprocessing.image import ImageDataGenerator


class ComplexModel(object):
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
		self.epochs = 20 #cambiar a 20
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

	def buildTopModel(self, topType):
		self.model.buildTopModel(topType, self.numClasses)

	def trainTopModel(self, trainData, trainLabels, validData, validLabels, fold):
		from keras import optimizers
		import math
		print('[INFO] Training top model type '+(str(self.model.topType)))
		
		self.model.topModel.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'], )	
		
		trainSamples = len(trainLabels)
		validationSamples = len(validLabels)
		predictSizeTrain = int(math.ceil(trainSamples / self.batchSize))
		predictSizeValidation = int(math.ceil(validationSamples / self.batchSize))
		
		trainLabels = utils.transformLabelsToCategorical(trainLabels, self.numClasses)
		validLabels = utils.transformLabelsToCategorical(validLabels, self.numClasses)
			
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
		historyGenerated = self.model.topModel.fit_generator(
							generatorTrain,
							steps_per_epoch=predictSizeTrain,
							epochs=self.epochs,
							validation_data=generatorVal,
							validation_steps=predictSizeValidation)
		
		trainingTime = time.process_time() - t

		self.logger.saveWeightsTopModel(self.model, fold)
		
		return trainingTime, historyGenerated

	def testTopModel(self, validationData, validationLabels, trainingTime, historyGenerated, fold):
		print('[INFO] testing top model type '+(str(self.model.topType)))

		validationSamples = len(validationLabels)

		validationLabels = utils.transformLabelsToCategorical(validationLabels, self.numClasses)
		
		testDatagen = ImageDataGenerator(rescale=1. / 255)
		generator = testDatagen.flow(validationData, validationLabels, shuffle=False, batch_size = self.batchSize)

		testLoss, testAccuracy = self.model.topModel.evaluate_generator(generator, steps=validationSamples/self.batchSize)

		self.logger.saveDataTopModel(historyGenerated, (testLoss, testAccuracy), trainingTime, fold, self.model.topType)

	def fineTune(self, trainData, trainLabels, validData, validLabels, numLayersFreeze, bestTopType, pathBestTop, fold):
		import math
		self.buildTopModel(bestTopType)
		self.model.topModel = self.logger.loadWeights(self.model.topModel, pathBestTop)
		self.model.topType = modelType.modelType[bestTopType]

		# Freeze layers
		for layer in self.model.topModel.layers[:numLayersFreeze]:
			layer.trainable = False

		for layer in self.mode.topModel.layers[numLayersFreeze:]:
			layer.trainable = True

		from keras import optimizers
		self.model.topModel.compile(loss='categorical_crossentropy',
			optimizer=optimizers.SGD(lr=1e-2, momentum=0.9),
			metrics=['accuracy'])


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
		historyGenerated = self.model.topModel.fit_generator(
							generatorTrain,
							steps_per_epoch=predictSizeTrain,
							epochs=10,
							validation_data=generatorVal,
							validation_steps=predictSizeValidation)
		
		trainingTime = time.process_time() - t

		self.logger.saveWeightsFineTune(self.model.topModel, fold, numLayersFreeze)
		
		return trainingTime, historyGenerated

	def testFineTuneModel(self, validationData, validationLabels, trainingTime, historyGenerated, fold, numLayersFreeze):
		print('[INFO] testing fine tune model')
		validationSamples = len(validationLabels)

		validationLabels = utils.transformLabelsToCategorical(validationLabels, self.numClasses)
		
		testDatagen = ImageDataGenerator(rescale=1. / 255)
		generator = testDatagen.flow(validationData, validationLabels, shuffle=False, batch_size = self.batchSize)

		testLoss, testAccuracy = self.model.topModel.evaluate_generator(generator, steps=validationSamples/self.batchSize)

		self.logger.saveDataFineTuneModel(historyGenerated, (testLoss, testAccuracy), trainingTime, fold, numLayersFreeze, self.model.topType)

		import gc
		del self.model.topModel
		gc.collect()
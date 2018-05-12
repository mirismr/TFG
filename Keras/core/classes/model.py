from classes.logger import *
import numpy as np

class Model(object):
	"""
	Model that represent a concrete neuronal network.

	Attributes
	----------
	kerasModel : keras.Sequential()
		Base model from Keras library.
	
	inputSize : tuple
		Base model's input size.

	logger : Logger
		Logger object that logs the train and test process

	batchSize : int
		Batch's size

	epochs : int
		Number of epochs that the model will train.

	Methods
	-------
	loadAndPreprocessData(path)
		Load and preprocess data for the appropiate Keras model.
	"""
	def __init__(self, kerasModel, inputSize, pathToModel):
		self.kerasModel = kerasModel #cuando se use hacer copia
		self.inputSize = inputSize
		self.logger = Logger(pathToModel)
		self.batchSize = 16
		self.epochs = 50

		self.logger.createStructureLogs()

	def loadAndPreprocessData(self, path):
		"""
		Load and preprocess data for the appropiate Keras model.

		Parameters
		----------
		path : string
			Path where we found the data

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

		print('[INFO] Reading data images')
		folders = os.listdir(path)
		folders = ['n02085374', 'n02121808', 'n01910747']#,'n02317335', 'n01922303', 'n01816887', 'n01859325', 'n02484322', 'n02504458', 'n01887787']
		for fld in folders:
			index = folders.index(fld)
			print('Loading folder {} (Class: {})'.format(fld, index))
			classDictionary[index] = fld

			path_sysnet = path+fld+"/"
			files = os.listdir(path_sysnet)
			for fl in files:
				img = image.load_img(path_sysnet+fl, target_size=self.inputSize)
				img = image.img_to_array(img)
				img = preprocess_input(img)
				data.append(img)
				labels.append(index)

		self.logger.saveClassDictionary(classDictionary)
		return np.array(data), np.array(labels)

	def generateBottlenecks(self, trainData, trainLabels, validData, validLabels):
		print('[INFO] Generating bottlenecks')

		model = kerasModel
			
		# Data augmentation
		trainDatagen = ImageDataGenerator(
			rescale=1. / 255,
			shear_range=0.2,
			zoom_range=0.2,
			horizontal_flip=True)       
		generator = trainDatagen.flow(trainData, trainLabels, shuffle=False, batch_size = self.batchSize)
		
		numClasses = len(np.unique(trainLabels))

		trainSamples = len(train_idx)
		predictSizeTrain = int(math.ceil(trainSamples / self.batchSize))
		
		bottlenecksFeaturesTrain = model.predict_generator(generator, predictSizeTrain, verbose=1)

		testDatagen = ImageDataGenerator(rescale=1. / 255)
		generator = testDatagen.flow(validData, validLabels, shuffle=False, batch_size = self.batchSize)
		validationSamples = len(val_idx)
		predictSizeValidation = int(math.ceil(validationSamples / self.batchSize))

		bottlenecksFeaturesValidation = model.predict_generator(generator, predictSizeValidation, verbose=1)

		self.logger.saveBottlenecks(bottlenecksFeaturesTrain, bottlenecksFeaturesValidation)

	def buildTopModel(self, topType):
		pass

	def trainTopModel(self):
		pass

	def buildFinalModel(self, topType, pathTop):
		pass

	def testFinalModel(self):
		pass

	def fineTune(self, data, labels, numLayersFreeze, bestTopType, pathBestTop):
		pass
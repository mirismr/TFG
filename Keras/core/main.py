import classes
import baseModels
from sklearn.model_selection import StratifiedKFold
from keras import backend as K
import gc




###################################Choose images######################################
#srcPath = "/home/mirismr/Descargas/choosen/"
#destPathSelected = "/home/mirismr/Descargas/data/"
#destPathNoSelected = "/home/mirismr/Descargas/noSelectedData/"
destPathSelected = "../data/"
destPathNoSelected = "../noSelectedData/"
#classNames = chooseRandomImages(srcPath, destPathSelected, destPathNoSelected, 1000)



#####################################VGG16###########################################
def trainTopModelsVgg16():
	# Necessary for split data
	print('\n*************************Train top models VGG16*****************************')
	vgg16Model = classes.Model(baseModels.VGG16(), (224, 224), "vgg16/")
	data, labels, numClasses = vgg16Model.loadAndPreprocessData(destPathSelected, log=True)
	numberSplits = classes.obtainNumSplitKFold(len(labels), 20)

	folds = list(StratifiedKFold(n_splits=numberSplits, shuffle=True).split(data, labels))

	for fold, (trainIdx, validIdx) in enumerate(folds):       
		print('\n[INFO] Executing fold ',fold)
		trainData = data[trainIdx]
		trainLabels = labels[trainIdx]
		validData = data[validIdx]
		validLabels = labels[validIdx]

		vgg16Model = classes.Model(baseModels.VGG16(), (224, 224), "vgg16/")
		vgg16Model.numClasses = numClasses
		vgg16Model.generateBottlenecks(trainData, trainLabels, validData, validLabels, fold)

		vgg16Model.buildTopModel(classes.modelType.typeA)
		trainingTime, historyGenerated = vgg16Model.trainTopModel(trainLabels, validLabels, fold)
		vgg16Model.testTopModel(validData, validLabels, trainingTime, historyGenerated, fold)

		
		vgg16Model.buildTopModel(classes.modelType.typeB)
		trainingTime, historyGenerated = vgg16Model.trainTopModel(trainLabels, validLabels, fold)
		vgg16Model.testTopModel(validData, validLabels, trainingTime, historyGenerated, fold)

		vgg16Model.buildTopModel(classes.modelType.typeC)
		trainingTime, historyGenerated = vgg16Model.trainTopModel(trainLabels, validLabels, fold)
		vgg16Model.testTopModel(validData, validLabels, trainingTime, historyGenerated, fold)

		# Free memory
		del vgg16Model
		gc.collect()
		K.clear_session()
		print('\n************END '+str(fold)+'***************')

def fineTuneVgg16():
	print('\n*************************Fine Tune VGG16*****************************')
	vgg16Model = classes.Model(baseModels.VGG16(), (224, 224), "vgg16/")
	data, labels, numClasses = vgg16Model.loadAndPreprocessData(destPathSelected, log=True)
	numberSplits = classes.obtainNumSplitKFold(len(labels), 20)

	folds = list(StratifiedKFold(n_splits=numberSplits, shuffle=True).split(data, labels))
	informationBestModel = classes.obtainBestModel("vgg16/topModel/")


	for fold, (trainIdx, validIdx) in enumerate(folds):
		print('\n[INFO] Executing fold ',fold)
		trainData = data[trainIdx]
		trainLabels = labels[trainIdx]
		validData = data[validIdx]
		validLabels = labels[validIdx]

		vgg16Model = classes.Model(baseModels.VGG16(), (224, 224), "vgg16/")
		vgg16Model.numClasses = numClasses

		vgg16Model.buildTopModel(informationBestModel[1])
		trainingTime, historyGenerated = vgg16Model.fineTune(trainData, trainLabels, validData, validLabels, 15, informationBestModel[1], informationBestModel[0], fold)
		vgg16Model.testFineTuneModel(validData, validLabels, trainingTime, historyGenerated, fold, 15)

		# Free memory
		del vgg16Model
		gc.collect()
		K.clear_session()

		#vgg16Model = classes.Model(baseModels.VGG16(), (224, 224), "vgg16/")
		#vgg16Model.numClasses = numClasses

		#vgg16Model.buildTopModel(informationBestModel[1])
		#trainingTime, historyGenerated = vgg16Model.fineTune(trainData, trainLabels, validData, validLabels, 12, informationBestModel[1], informationBestModel[0], fold)
		#vgg16Model.testFineTuneModel(validData, validLabels, trainingTime, historyGenerated, fold, 12)
		

		print('\n************END Fold '+str(fold)+'***************')


###################################InceptionV3#######################################
def trainTopModelInceptionV3():
	print('\n*************************Train top models InceptionV3*****************************')
	inceptionV3 = classes.ComplexModel(baseModels.InceptionV3(), (299,299), "inceptionV3/")
	data, labels, numClasses = inceptionV3.loadAndPreprocessData(destPathSelected, log=True)
	numberSplits = classes.obtainNumSplitKFold(len(labels), 20)
	folds = list(StratifiedKFold(n_splits=numberSplits, shuffle=True).split(data, labels))

	for fold, (trainIdx, validIdx) in enumerate(folds):       
		print('\n[INFO] Executing fold ',fold)
		trainData = data[trainIdx]
		trainLabels = labels[trainIdx]
		validData = data[validIdx]
		validLabels = labels[validIdx]

		inceptionV3.buildTopModel(classes.modelType.typeA)
		trainingTime, historyGenerated = inceptionV3.trainTopModel(trainData, trainLabels, validData, validLabels, fold)
		inceptionV3.testTopModel(validData, validLabels, trainingTime, historyGenerated, fold)

		print('\n************END '+str(fold)+'***************')

def fineTuneInceptionV3():
	
	inceptionV3 = classes.ComplexModel(baseModels.InceptionV3(), (299,299), "inceptionV3/")
	data, labels, numClasses = inceptionV3.loadAndPreprocessData(destPathSelected, log=True)
	numberSplits = classes.obtainNumSplitKFold(len(labels), 20)

	folds = list(StratifiedKFold(n_splits=numberSplits, shuffle=True).split(data, labels))
	informationBestModel = classes.obtainBestModel("inceptionV3/topModel/")
	print('\n*************************Fine Tune inceptionv3*****************************')
	for fold, (trainIdx, validIdx) in enumerate(folds):       
		print('\n[INFO] Executing fold ',fold)
		trainData = data[trainIdx]
		trainLabels = labels[trainIdx]
		validData = data[validIdx]
		validLabels = labels[validIdx]

		trainingTime, historyGenerated = inceptionV3.fineTune(trainData, trainLabels, validData, validLabels, 279, informationBestModel[1], informationBestModel[0], fold)
		inceptionV3.testFineTuneModel(validData, validLabels, trainingTime, historyGenerated, fold, 15)
		print('\n************END '+str(fold)+'***************')

###################################Xception###########################################
def trainTopModelXception():
	print('\n*************************Train top models Xception*****************************')
	xception = classes.ComplexModel(baseModels.Xception(), (299,299), "xception/")
	data, labels, numClasses = xception.loadAndPreprocessData(destPathSelected, log=True)
	numberSplits = classes.obtainNumSplitKFold(len(labels), 20)
	folds = list(StratifiedKFold(n_splits=numberSplits, shuffle=True).split(data, labels))

	for fold, (trainIdx, validIdx) in enumerate(folds):       
		print('\n[INFO] Executing fold ',fold)
		trainData = data[trainIdx]
		trainLabels = labels[trainIdx]
		validData = data[validIdx]
		validLabels = labels[validIdx]

		xception.buildTopModel(classes.modelType.typeA)
		trainingTime, historyGenerated = xception.trainTopModel(trainData, trainLabels, validData, validLabels, fold)
		xception.testTopModel(validData, validLabels, trainingTime, historyGenerated, fold)

		print('\n************END '+str(fold)+'***************')

def fineTuneXception():
	
	xception = classes.ComplexModel(baseModels.Xception(), (299,299), "xception/")
	data, labels, numClasses = xception.loadAndPreprocessData(destPathSelected, log=True)
	numberSplits = classes.obtainNumSplitKFold(len(labels), 20)

	folds = list(StratifiedKFold(n_splits=numberSplits, shuffle=True).split(data, labels))
	informationBestModel = classes.obtainBestModel("xception/topModel/")
	print('\n*************************Fine Tune Xception*****************************')
	for fold, (trainIdx, validIdx) in enumerate(folds):       
		print('\n[INFO] Executing fold ',fold)
		trainData = data[trainIdx]
		trainLabels = labels[trainIdx]
		validData = data[validIdx]
		validLabels = labels[validIdx]

		trainingTime, historyGenerated = xception.fineTune(trainData, trainLabels, validData, validLabels, 126, informationBestModel[1], informationBestModel[0], fold)
		xception.testFineTuneModel(validData, validLabels, trainingTime, historyGenerated, fold, 15)
		print('\n************END '+str(fold)+'***************')

##################################Mobilenet##########################################
def trainTopModelMobilenet():
	print('\n*************************Train top models Mobilenet*****************************')
	mobilenet = classes.ComplexModel(baseModels.Mobilenet(), (299,299), "mobilenet/")
	data, labels, numClasses = mobilenet.loadAndPreprocessData(destPathSelected, log=True)
	numberSplits = classes.obtainNumSplitKFold(len(labels), 20)
	folds = list(StratifiedKFold(n_splits=numberSplits, shuffle=True).split(data, labels))

	for fold, (trainIdx, validIdx) in enumerate(folds):       
		print('\n[INFO] Executing fold ',fold)
		trainData = data[trainIdx]
		trainLabels = labels[trainIdx]
		validData = data[validIdx]
		validLabels = labels[validIdx]

		mobilenet.buildTopModel(classes.modelType.typeA)
		trainingTime, historyGenerated = mobilenet.trainTopModel(trainData, trainLabels, validData, validLabels, fold)
		mobilenet.testTopModel(validData, validLabels, trainingTime, historyGenerated, fold)

		print('\n************END '+str(fold)+'***************')

def fineTuneMobilenet():
	mobilenet = classes.ComplexModel(baseModels.Mobilenet(), (299,299), "mobilenet/")
	data, labels, numClasses = mobilenet.loadAndPreprocessData(destPathSelected, log=True)
	numberSplits = classes.obtainNumSplitKFold(len(labels), 20)

	folds = list(StratifiedKFold(n_splits=numberSplits, shuffle=True).split(data, labels))
	informationBestModel = classes.obtainBestModel("mobilenet/topModel/")
	print('\n*************************Fine Tune Mobilenet*****************************')
	for fold, (trainIdx, validIdx) in enumerate(folds):       
		print('\n[INFO] Executing fold ',fold)
		trainData = data[trainIdx]
		trainLabels = labels[trainIdx]
		validData = data[validIdx]
		validLabels = labels[validIdx]

		trainingTime, historyGenerated = mobilenet.fineTune(trainData, trainLabels, validData, validLabels, 76, informationBestModel[1], informationBestModel[0], fold)
		mobilenet.testFineTuneModel(validData, validLabels, trainingTime, historyGenerated, fold, 15)
		print('\n************END '+str(fold)+'***************')

######################################################################################
#trainTopModelsVgg16()
#fineTuneVgg16()

#trainTopModelInceptionV3()
#fineTuneInceptionV3()

#cuando tenga los fineTune de vgg16
#informationBestVgg16 = classes.obtainBestModel("vgg16/fineTune/")
#classes.saveFinalModelsPath(informationBestVgg16)

####################################Plotting############################################

'''
plotter = classes.Plotter("/home/mirismr/Descargas/modelsData/vgg16/")
plotter.plotTopHistory(classes.modelType.typeA)
plotter.plotTopHistory(classes.modelType.typeB)
plotter.plotTopHistory(classes.modelType.typeC)
plotter.plotTopGeneralData()
'''

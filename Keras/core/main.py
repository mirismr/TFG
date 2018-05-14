import classes
import baseModels
from sklearn.model_selection import StratifiedKFold
from keras import backend as K
import gc


###################################Choose images######################################
srcPath = "/home/mirismr/Descargas/choosen/"
destPathSelected = "/home/mirismr/Descargas/data/"
destPathNoSelected = "/home/mirismr/Descargas/noSelectedData/"
#classNames = chooseRandomImages(srcPath, destPathSelected, destPathNoSelected, 1000)

#####################################K Fold############################################
# Necessary for split data
vgg16Model = classes.Model(baseModels.VGG16(), (224, 224), "vgg16/")
data, labels, numClasses = vgg16Model.loadAndPreprocessData(destPathSelected, log=True)
numberSplits = classes.obtainNumSplitKFold(len(labels), 20)

folds = list(StratifiedKFold(n_splits=numberSplits, shuffle=True).split(data, labels))

for fold, (trainIdx, validIdx) in enumerate(folds):       
	print('\n[INFO] Executing fold ',fold)
	print('\n************VGG16***************')
	trainData = data[trainIdx]
	trainLabels = labels[trainIdx]
	validData = data[validIdx]
	validLabels = labels[validIdx]

	vgg16Model.generateBottlenecks(trainData, trainLabels, validData, validLabels, fold)

	vgg16Model.buildTopModel(classes.modelType.typeA)
	trainingTime, historyGenerated = vgg16Model.trainTopModel(trainLabels, validLabels, fold)
	vgg16Model.testTopModel(validData, validLabels, trainingTime, historyGenerated, fold)

	vgg16Model.buildTopModel(classes.modelType.typeB)
	trainingTime, historyGenerated = vgg16Model.trainTopModel(trainLabels, validLabels, fold)
	vgg16Model.testTopModel(validData, validLabels, trainingTime, historyGenerated, fold)

	# Free memory
	del vgg16Model
	gc.collect()
	K.clear_session()

	vgg16Model = classes.Model(baseModels.VGG16(), (224, 224), "vgg16/")
	vgg16Model.numClasses = numClasses

	informationBestModel = classes.obtainBestModel("vgg16/topModel/")

	vgg16Model.buildTopModel(informationBestModel[1])
	trainingTime, historyGenerated = vgg16Model.fineTune(trainData, trainLabels, validData, validLabels, 15, informationBestModel[1], informationBestModel[0], fold)
	vgg16Model.testFineTuneModel(validData, validLabels, trainingTime, historyGenerated, fold, 15)

	# Free memory
	del vgg16Model
	gc.collect()
	K.clear_session()

	vgg16Model = classes.Model(baseModels.VGG16(), (224, 224), "vgg16/")
	vgg16Model.numClasses = numClasses

	vgg16Model.buildTopModel(informationBestModel[1])
	trainingTime, historyGenerated = vgg16Model.fineTune(trainData, trainLabels, validData, validLabels, 12, informationBestModel[1], informationBestModel[0], fold)
	vgg16Model.testFineTuneModel(validData, validLabels, trainingTime, historyGenerated, fold, 12)
	

	print('\n************END VGG16 Fold '+str(fold)+'***************')

	# Free memory
	del vgg16Model
	del trainData
	del trainLabels
	del validData
	del validLabels
	K.clear_session()
	gc.collect()



######################################################################################
informationBestVgg16 = classes.obtainBestModel("vgg16/fineTune/")
#informationBestVgg16 = classes.obtainBestModel("vgg16/fineTune/")
#informationBestVgg16 = classes.obtainBestModel("vgg16/fineTune/")
#informationBestVgg16 = classes.obtainBestModel("vgg16/fineTune/")

classes.saveFinalModelsPath(informationBestVgg16)

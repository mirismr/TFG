def chooseRandomImages(srcPath, destPathSelected, destPathNoSelected, numImages):
	"""
	Choose a number of random images from a directory.
	If there is more images than numImages, move to destPathNoSelected.
	It creates the necessary directories. 

	Parameters
	----------
	srcPath : string
		Path from choose the images.
	destPathSelected : string
		Path where we move the selected images.
	destPathNoSelected : string
		Path where we move the non selected images.
	numImages : int
		Number of images to be selected.
		

	Returns
	-------
	sysnets: list
		classes' list that the model will learn
	"""
	import os, random
	from shutil import copyfile

	if not os.path.exists(destPathSelected):
		os.makedirs(destPathSelected)
	if not os.path.exists(destPathNoSelected):
		os.makedirs(destPathNoSelected)
	if not os.path.exists(destPathNoSelected+"all/"):
		os.makedirs(destPathNoSelected+"all/")

	sysnets = os.listdir(srcPath)
	for sysnet in sysnets:
		dirSrcFull = srcPath+"/"+sysnet+"/"

		print("Processing --> "+sysnet)

		if not os.path.exists(destPathSelected+sysnet):
			os.makedirs(destPathSelected+sysnet)
			os.makedirs(destPathNoSelected+"organized/"+sysnet)

			images = os.listdir(dirSrcFull)

			if len(images) >= numImages:
				random.shuffle(images)

				selectedImages = images[:numImages]
				noSelectedImages = images[numImages:]

				for img in selectedImages:
					copyfile(dirSrcFull+img,destPathSelected+sysnet+"/"+img)

				for img in noSelectedImages:
					copyfile(dirSrcFull+img,destPathNoSelected+"organized/"+sysnet+"/"+img)
					copyfile(dirSrcFull+img,destPathNoSelected+"all/"+img)
			else:
				print("Not enough images in "+dirSrcFull)
		else:
			print("Duplicated ",sysnet, ". Not copied.")

	return sysnets

def obtainNumSplitKFold(numExamples, percentageTest):
	"""
	Obtain how many split we should use for K Fold Cross Validation.

	Parameters
	----------
	numExamples : int
		Number of examples per class.
	percentageTest : int
		Percentage's test per class between 0-100

	Returns
	-------
	int
		Number of splits
	"""
	return int(numExamples / (numExamples*percentageTest/100))

def transformLabelsToCategorical(labels, numClasses):
	from keras.utils.np_utils import to_categorical
	
	return to_categorical(labels, num_classes=numClasses)

def obtainBestModel(path):
	import json
	import os
	pathGeneralData = path + "generalData/"
	pathHistory = path + "dataHistory/"
	data = []
	accuracies = []

	dataFiles = os.listdir(pathGeneralData)
	for f in dataFiles:	
		with open(pathGeneralData+f) as jsonData:
			data.append(json.load(jsonData))

	dataFilesHistoryA = os.listdir(pathHistory+"/typeA/")
	dataFilesHistoryB = os.listdir(pathHistory+"/typeB/")
	dataFilesHistoryC = os.listdir(pathHistory+"/typeC/")

	dataFilesHistory = dataFilesHistoryA+dataFilesHistoryB+dataFilesHistoryC

	for d in data:
		accuracies.append(d['testAccuracy'])

	maxAccuracy = max(accuracies)
	indexMax = accuracies.index(maxAccuracy)
	
	pathModel = ''
	topType = ''
	with open(pathGeneralData+dataFiles[indexMax], 'r') as jsonData:
		jsonLoad = json.load(jsonData)
		pathModel = jsonLoad['pathModel']
		topType = jsonLoad['topType']

	# full path to model, topType, path generalData, path history
	return pathModel, topType, pathGeneralData+dataFiles[indexMax], pathHistory+dataFilesHistory[indexMax]

def saveFinalModelsPath(informationBestVgg16):
	from shutil import copyfile
	copyfile(informationBestVgg16[0], "finalModels/weights/final_model_vgg16.h5")
	copyfile(informationBestVgg16[2], "finalModels/generalData/general_data_vgg16.json")
	copyfile(informationBestVgg16[3], "finalModels/dataHistory/history_vgg16.h5")
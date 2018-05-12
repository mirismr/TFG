import classes
import baseModels
from sklearn.model_selection import StratifiedKFold


###################################Choose images######################################
srcPath = "/home/mirismr/Descargas/choosen/"
destPathSelected = "/home/mirismr/Descargas/data/"
destPathNoSelected = "/home/mirismr/Descargas/noSelectedData/"
#classNames = chooseRandomImages(srcPath, destPathSelected, destPathNoSelected, 1000)

#####################################K Fold############################################
print('************VGG16************** ',j)
vgg16Model = classes.Model(baseModels.VGG16(), (224, 224), "vgg16/")
data, labels = vgg16Model.loadAndPreprocessData(destPathSelected)
numberSplits = classes.obtainNumSplitKFold(len(labels), 20)

folds = list(StratifiedKFold(n_splits=numberSplits, shuffle=True).split(data, labels))

for j, (trainIdx, validIdx) in enumerate(folds):       
	
	print('\n [INFO] Executing fold ',j)
	trainData = data[trainIdx]
	trainLabels = labels[trainIdx]
	validData = data[validIdx]
	validLabels = labels[validIdx]

	vgg16Model.generateBottlenecks(trainData, trainLabels, validData, validLabels)

	#modelo.train()
	#modelo.test()
	#modelo.save()


######################################################################################
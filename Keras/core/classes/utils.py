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

def obtainBestTopModel(path):
	pass

def obtainBestFineTuneModel(path):
	pass

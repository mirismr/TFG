def obtainBestTopModel(path):
	pass

def obtainBestFineTuneModel(path):
	pass

def average(valuesList):
	pass

def chooseRandomImages(srcPath, destPathSelected, destPathNoSelected, numImages):
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

		

		#eliminar repetidos porque en las descargas me descargo al propio y a los hijos,
		#entonces puede que un hijo se repita luego como padre
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

if __name__ == '__main__':
	srcPath = "/home/mirismr/Descargas/choosen/"
	destPathSelected = "/home/mirismr/Descargas/data/"
	destPathNoSelected = "/home/mirismr/Descargas/noSelectedData/"
	chooseRandomImages(srcPath, destPathSelected, destPathNoSelected, 1000)
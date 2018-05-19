from . import modelType

class Plotter(object):
	def __init__(self, path):
		self.path = path

	def loadDictionaryData(self, directory):
		import json
		import os

		files = os.listdir(directory)
		data = []

		for f in files:	
			with open(directory+f) as jsonData:
				data.append(json.load(jsonData))

		return data

	def plotTopHistory(self, modelType):
		import matplotlib.pyplot as plt
		from pylab import savefig
		import os

		# Create directory for plot if not exists
		if not os.path.exists(self.path+"topModel/plots/"):
			os.makedirs(self.path+"topModel/plots/")

		if modelType == modelType.typeA:
			directory = self.path+'topModel/dataHistory/typeA/'
		elif modelType == modelType.typeB:
			directory = self.path+'topModel/dataHistory/typeB/'
		elif modelType == modelType.typeC:
			directory = self.path+'topModel/dataHistory/typeC/'

		histories = self.loadDictionaryData(directory)
		numberFolds = len(histories)

		fold_0 = histories[0]
		fold_1 = histories[1]
		fold_2 = histories[2]
		fold_3 = histories[3]
		fold_4 = histories[4]

		lossList = [s for s in fold_0.keys() if 'loss' in s and 'val' not in s]
		valLossList = [s for s in fold_0.keys() if 'loss' in s and 'val' in s]

		epochs = range(1,len(fold_0[lossList[0]]) + 1)
	
		## Loss
		fig = plt.figure(1, figsize=(8,8))
		
		for l in lossList:
			 plt.plot(epochs, fold_0[l], 'b', label='Training loss fold 0(' + str(str(format(fold_0[l][-1],'.5f'))+')'))
		for l in valLossList:
			plt.plot(epochs, fold_0[l], 'b', label='Validation loss fold 0(' + str(str(format(fold_0[l][-1],'.5f'))+')'))
   

		lossList = [s for s in fold_1.keys() if 'loss' in s and 'val' not in s]
		valLossList = [s for s in fold_1.keys() if 'loss' in s and 'val' in s]

		for l in lossList:
			 plt.plot(epochs, fold_1[l], 'g', label='Training loss fold 1(' + str(str(format(fold_1[l][-1],'.5f'))+')'))
		for l in valLossList:
			plt.plot(epochs, fold_1[l], 'g', label='Validation loss fold 1(' + str(str(format(fold_1[l][-1],'.5f'))+')'))
   
		lossList = [s for s in fold_2.keys() if 'loss' in s and 'val' not in s]
		valLossList = [s for s in fold_2.keys() if 'loss' in s and 'val' in s]

		for l in lossList:
			 plt.plot(epochs, fold_2[l], 'y', label='Training loss fold 2(' + str(str(format(fold_2[l][-1],'.5f'))+')'))
		for l in valLossList:
			plt.plot(epochs, fold_2[l], 'y', label='Validation loss fold 2(' + str(str(format(fold_2[l][-1],'.5f'))+')'))
   
		lossList = [s for s in fold_3.keys() if 'loss' in s and 'val' not in s]
		valLossList = [s for s in fold_3.keys() if 'loss' in s and 'val' in s]

		for l in lossList:
			 plt.plot(epochs, fold_3[l], 'tab:pink', label='Training loss fold 3(' + str(str(format(fold_3[l][-1],'.5f'))+')'))
		for l in valLossList:
			plt.plot(epochs, fold_3[l], 'tab:pink', label='Validation loss fold 3(' + str(str(format(fold_3[l][-1],'.5f'))+')'))
   

		lossList = [s for s in fold_4.keys() if 'loss' in s and 'val' not in s]
		valLossList = [s for s in fold_4.keys() if 'loss' in s and 'val' in s]

		for l in lossList:
			 plt.plot(epochs, fold_4[l], 'tab:orange', label='Training loss fold 4(' + str(str(format(fold_4[l][-1],'.5f'))+')'))
		for l in valLossList:
			plt.plot(epochs, fold_4[l], 'tab:orange', label='Validation loss fold 4(' + str(str(format(fold_4[l][-1],'.5f'))+')'))
		


		## Average
		average = {}
		loss = []
		valLoss = []
		acc = []
		valAcc = []
		for i in range(0,len(fold_0['loss'])):
			loss_i = fold_0['loss'][i]+fold_1['loss'][i]+fold_2['loss'][i]+fold_3['loss'][i]+fold_4['loss'][i]
			loss_i = loss_i/numberFolds
			loss.append(loss_i)

			valLoss_i = fold_0['val_loss'][i]+fold_1['val_loss'][i]+fold_2['val_loss'][i]+fold_3['val_loss'][i]+fold_4['val_loss'][i]
			valLoss_i = valLoss_i/numberFolds
			valLoss.append(valLoss_i)

			acc_i = fold_0['acc'][i]+fold_1['acc'][i]+fold_2['acc'][i]+fold_3['acc'][i]+fold_4['acc'][i]
			acc_i = acc_i/numberFolds
			acc.append(acc_i)

			valAcc_i = fold_0['val_acc'][i]+fold_1['val_acc'][i]+fold_2['val_acc'][i]+fold_3['val_acc'][i]+fold_4['val_acc'][i]
			valAcc_i = valAcc_i/numberFolds
			valAcc.append(valAcc_i)

		average['loss'] = loss
		average['val_loss'] = valLoss
		average['acc'] = acc
		average['val_acc'] = valAcc

		lossList = [s for s in average.keys() if 'loss' in s and 'val' not in s]
		valLossList = [s for s in average.keys() if 'loss' in s and 'val' in s]

		for l in lossList:
			 plt.plot(epochs, average[l], 'kx', label='Training loss average(' + str(str(format(average[l][-1],'.5f'))+')'))
		for l in valLossList:
			plt.plot(epochs, average[l], 'rx', label='Validation loss average(' + str(str(format(average[l][-1],'.5f'))+')'))
   
		
		
		plt.title('Loss '+str(modelType.name))
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()

		savefig(self.path+'topModel/plots/'+str(modelType.name)+'_loss.png', bbox_inches='tight')
		plt.close(fig)

		## Accuracy
		fig = plt.figure(2,figsize=(8,8))

		accList = [s for s in fold_0.keys() if 'acc' in s and 'val' not in s]
		valAccList = [s for s in fold_0.keys() if 'acc' in s and 'val' in s]

		for l in accList:
			 plt.plot(epochs, fold_0[l], 'b', label='Training accuracy fold 0(' + str(format(fold_0[l][-1],'.5f'))+')')
		for l in valAccList:    
			plt.plot(epochs, fold_0[l], 'b', label='Validation accuracy fold 0(' + str(format(fold_0[l][-1],'.5f'))+')')
		

		accList = [s for s in fold_1.keys() if 'acc' in s and 'val' not in s]
		valAccList = [s for s in fold_1.keys() if 'acc' in s and 'val' in s]

		for l in accList:
			 plt.plot(epochs, fold_1[l], 'g', label='Training accuracy fold 1(' + str(format(fold_1[l][-1],'.5f'))+')')
		for l in valAccList:    
			plt.plot(epochs, fold_1[l], 'g', label='Validation accuracy fold 1(' + str(format(fold_1[l][-1],'.5f'))+')')
		

		accList = [s for s in fold_2.keys() if 'acc' in s and 'val' not in s]
		valAccList = [s for s in fold_2.keys() if 'acc' in s and 'val' in s]

		for l in accList:
			 plt.plot(epochs, fold_2[l], 'y', label='Training accuracy fold 2(' + str(format(fold_2[l][-1],'.5f'))+')')
		for l in valAccList:    
			plt.plot(epochs, fold_2[l], 'y', label='Validation accuracy fold 2(' + str(format(fold_2[l][-1],'.5f'))+')')
		

		accList = [s for s in fold_3.keys() if 'acc' in s and 'val' not in s]
		valAccList = [s for s in fold_3.keys() if 'acc' in s and 'val' in s]

		for l in accList:
			 plt.plot(epochs, fold_3[l], 'tab:pink', label='Training accuracy fold 3(' + str(format(fold_3[l][-1],'.5f'))+')')
		for l in valAccList:    
			plt.plot(epochs, fold_3[l], 'tab:pink', label='Validation accuracy fold 3(' + str(format(fold_3[l][-1],'.5f'))+')')
		


		accList = [s for s in fold_4.keys() if 'acc' in s and 'val' not in s]
		valAccList = [s for s in fold_4.keys() if 'acc' in s and 'val' in s]


		for l in accList:
			 plt.plot(epochs, fold_4[l], 'tab:orange', label='Training accuracy fold 4(' + str(format(fold_4[l][-1],'.5f'))+')')
		for l in valAccList:    
			plt.plot(epochs, fold_4[l], 'tab:orange', label='Validation accuracy fold 4(' + str(format(fold_4[l][-1],'.5f'))+')')
		


		accList = [s for s in average.keys() if 'acc' in s and 'val' not in s]
		valAccList = [s for s in average.keys() if 'acc' in s and 'val' in s]

		for l in accList:
			 plt.plot(epochs, average[l], 'kx', label='Training accuracy average(' + str(format(average[l][-1],'.5f'))+')')
		for l in valAccList:    
			plt.plot(epochs, average[l], 'rx', label='Validation accuracy average(' + str(format(average[l][-1],'.5f'))+')')
		

		plt.title('Accuracy '+str(modelType.name))
		plt.xlabel('Epochs')
		plt.ylabel('Accuracy')
		plt.legend()

		savefig(self.path+'topModel/plots/'+str(modelType.name)+'_accuracy.png', bbox_inches='tight')
		plt.close(fig)

	def plotTopGeneralData(self):
		import numpy as np
		import matplotlib.pyplot as plt
		from pylab import savefig

		directoryA = self.path+'topModel/generalData/typeA/'
		directoryB = self.path+'topModel/generalData/typeB/'
		directoryC = self.path+'topModel/generalData/typeC/'

		dataA = self.loadDictionaryData(directoryA)
		dataB = self.loadDictionaryData(directoryB)
		dataC = self.loadDictionaryData(directoryC)
		numberFolds = len(dataA)

		fold_0_A = dataA[0]
		fold_1_A = dataA[1]
		fold_2_A = dataA[2]
		fold_3_A = dataA[3]
		fold_4_A = dataA[4]

		fold_0_B = dataB[0]
		fold_1_B = dataB[1]
		fold_2_B = dataB[2]
		fold_3_B = dataB[3]
		fold_4_B = dataB[4]

		fold_0_C = dataC[0]
		fold_1_C = dataC[1]
		fold_2_C = dataC[2]
		fold_3_C = dataC[3]
		fold_4_C = dataC[4]

		averageTestLossA = (fold_0_A['testLoss']+fold_1_A['testLoss']+fold_2_A['testLoss']+fold_3_A['testLoss']+fold_4_A['testLoss'])/numberFolds
		averageTestAccuracyA = (fold_0_A['testAccuracy']+fold_1_A['testAccuracy']+fold_2_A['testAccuracy']+fold_3_A['testAccuracy']+fold_4_A['testAccuracy'])/numberFolds

		averageTestLossB = (fold_0_B['testLoss']+fold_1_B['testLoss']+fold_2_B['testLoss']+fold_3_B['testLoss']+fold_4_B['testLoss'])/numberFolds
		averageTestAccuracyB = (fold_0_B['testAccuracy']+fold_1_B['testAccuracy']+fold_2_B['testAccuracy']+fold_3_B['testAccuracy']+fold_4_B['testAccuracy'])/numberFolds

		averageTestLossC = (fold_0_C['testLoss']+fold_1_C['testLoss']+fold_2_C['testLoss']+fold_3_C['testLoss']+fold_4_C['testLoss'])/numberFolds
		averageTestAccuracyC = (fold_0_C['testAccuracy']+fold_1_C['testAccuracy']+fold_2_C['testAccuracy']+fold_3_C['testAccuracy']+fold_4_C['testAccuracy'])/numberFolds

		# data to plot
		n_groups = 3
		testAccuracy = (averageTestAccuracyA, averageTestAccuracyB, averageTestAccuracyC)
		testLoss = (averageTestLossA, averageTestLossB, averageTestLossC)
		 
		# create plot
		fig, ax = plt.subplots()
		index = np.arange(n_groups)
		bar_width = 0.35
		opacity = 0.8
		 
		rects1 = plt.bar(index, testAccuracy, bar_width,
		                 alpha=opacity,
		                 color='b',
		                 label='Test Accuracy')
		 
		rects2 = plt.bar(index + bar_width, testLoss, bar_width,
		                 alpha=opacity,
		                 color='g',
		                 label='Test Loss')
		 
		plt.xlabel('Models')
		plt.ylabel('Scores')
		plt.title('Testing models')
		plt.xticks(index + bar_width, ('typeA', 'typeB', 'typeC'))
		plt.legend()
		 
		plt.tight_layout()
		
		savefig(self.path+'topModel/plots/testTopModels.png', bbox_inches='tight')
		plt.close(fig)
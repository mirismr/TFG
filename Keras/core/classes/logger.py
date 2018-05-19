import numpy as np

class Logger(object):
    def __init__(self, path):
        self.path = path

   
    def createStructureLogs(self):
        """
        Create the directories structure for logs
        
        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        import os
        if not os.path.exists("finalModels/generalData/"):
            os.makedirs("finalModels/generalData/")
        if not os.path.exists("finalModels/weights/"):
            os.makedirs("finalModels/weights/")
        if not os.path.exists("finalModels/dataHistory/"):
            os.makedirs("finalModels/dataHistory/")

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if not os.path.exists(self.path+"topModel/bottlenecks/"):
            os.makedirs(self.path+"topModel/bottlenecks/")
        if not os.path.exists(self.path+"topModel/dataHistory/typeA/"):
            os.makedirs(self.path+"topModel/dataHistory/typeA/")
        if not os.path.exists(self.path+"topModel/dataHistory/typeB/"):
            os.makedirs(self.path+"topModel/dataHistory/typeB/")
        if not os.path.exists(self.path+"topModel/dataHistory/typeC/"):
            os.makedirs(self.path+"topModel/dataHistory/typeC/")

        if not os.path.exists(self.path+"topModel/weights/"):
            os.makedirs(self.path+"topModel/weights/")
        if not os.path.exists(self.path+"topModel/generalData/"):
            os.makedirs(self.path+"topModel/generalData/")
            
        if not os.path.exists(self.path+"fineTune/dataHistory/15/"):
            os.makedirs(self.path+"fineTune/dataHistory/15/")
        if not os.path.exists(self.path+"fineTune/dataHistory/12/"):
            os.makedirs(self.path+"fineTune/dataHistory/12/")
        if not os.path.exists(self.path+"fineTune/weights/"):
            os.makedirs(self.path+"fineTune/weights/")
        if not os.path.exists(self.path+"fineTune/generalData/"):
            os.makedirs(self.path+"fineTune/generalData/")

    def saveClassDictionary(self, dictionary):	
        import json  
        contentJson = json.dumps(dictionary, indent=4, sort_keys=True)
        f = open(self.path+'classDictionary.json', "w")
        f.write(contentJson)
        f.close()

    def saveBottlenecks(self, bottlenecksFeaturesTrain, bottlenecksFeaturesValidation, fold):
        np.save(self.path+'topModel/bottlenecks/bottleneck_features_train_fold_'+str(fold)+'.npy', bottlenecksFeaturesTrain)
        np.save(self.path+'topModel/bottlenecks/bottleneck_features_validation_fold_'+str(fold)+'.npy',bottlenecksFeaturesValidation)

    def loadBottlenecks(self, fold):
        bottlenecksFeaturesTrain = np.load(self.path+'topModel/bottlenecks/bottleneck_features_train_fold_'+str(fold)+'.npy')
        bottlenecksFeaturesValidation = np.load(self.path+'topModel/bottlenecks/bottleneck_features_validation_fold_'+str(fold)+'.npy')

        return bottlenecksFeaturesTrain, bottlenecksFeaturesValidation

    def saveWeightsTopModel(self, model, fold):
        model.topModel.save(self.path+'topModel/weights/top_model_exported_fold_'+str(fold)+"_"+str(model.topType.value)+'.h5')

    def saveWeightsFineTune(self, model, fold, numLayersFreeze):
        model.save(self.path+'fineTune/weights/final_model_exported_fold_'+str(fold)+"_"+str(numLayersFreeze)+'.h5')

    def loadWeights(self, model, path):
        model.load_weights(path)

        return model

    def saveDataTopModel(self, historyGenerated, resultTest, trainingTime, fold, topType):
        import json  
        contentJson = json.dumps(historyGenerated.history, indent=4, sort_keys=True)
        f = open(self.path+'topModel/dataHistory/'+str(topType.name)+'/top_model_history_fold_'+str(fold)+"_"+str(topType.value)+'.json', "w")
        f.write(contentJson)
        f.close()

        contentJson = json.dumps({'testLoss':resultTest[0], 'testAccuracy':resultTest[1], 'trainingTime': trainingTime, 'pathModel': self.path+'topModel/weights/top_model_exported_fold_'+str(fold)+"_"+str(topType.value)+'.h5', 'topType':str(topType.name)})
        f = open(self.path+'topModel/generalData/general_data_fold_'+str(fold)+"_"+str(topType.value)+'.json', "w")
        f.write(contentJson)
        f.close()

    def saveDataFineTuneModel(self, historyGenerated, resultTest, trainingTime, fold, numLayersFreeze, topType):
        import json  
        contentJson = json.dumps(historyGenerated.history, indent=4, sort_keys=True)
        f = open(self.path+'fineTune/dataHistory/'+str(numLayersFreeze)+'/top_model_history_fold_'+str(fold)+"_"+str(numLayersFreeze)+'.json', "w")
        f.write(contentJson)
        f.close()

        contentJson = json.dumps({'testLoss':resultTest[0], 'testAccuracy':resultTest[1], 'trainingTime': trainingTime, 'pathModel': self.path+'fineTune/weights/final_model_exported_fold_'+str(fold)+"_"+str(numLayersFreeze)+'.h5', 'topType':str(topType.name)})
        f = open(self.path+'fineTune/generalData/general_data_fold_'+str(fold)+"_"+str(numLayersFreeze)+'.json', "w")
        f.write(contentJson)
        f.close()
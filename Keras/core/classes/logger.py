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
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if not os.path.exists(self.path+"topModel/bottlenecks/"):
            os.makedirs(self.path+"topModel/bottlenecks/")
        if not os.path.exists(self.path+"topModel/dataHistory/"):
            os.makedirs(self.path+"topModel/dataHistory/")
        if not os.path.exists(self.path+"topModel/weights/"):
            os.makedirs(self.path+"topModel/weights/")
        if not os.path.exists(self.path+"topModel/generalData/"):
            os.makedirs(self.path+"topModel/generalData/")
        if not os.path.exists(self.path+"topModel/plots/"):
            os.makedirs(self.path+"topModel/plots/")

        if not os.path.exists(self.path+"fineTune/dataHistory/"):
            os.makedirs(self.path+"fineTune/dataHistory/")
        if not os.path.exists(self.path+"fineTune/weights/"):
            os.makedirs(self.path+"fineTune/weights/")
        if not os.path.exists(self.path+"fineTune/generalData/"):
            os.makedirs(self.path+"fineTune/generalData/")
        if not os.path.exists(self.path+"fineTune/plots/"):
            os.makedirs(self.path+"fineTune/plots/")

    def saveClassDictionary(self, dictionary):	
        import json  
        contentJson = json.dumps(dictionary, indent=4, sort_keys=True)
        f = open(self.path+'classDictionary.json', "w")
        f.write(contentJson)
        f.close()

    def saveBottlenecks(bottlenecksFeaturesTrain, bottlenecksFeaturesValidation):
        np.save(self.path+'bottlenecks/bottleneck_features_train_fold_'+str(j)+'.npy', bottlenecksFeaturesTrain)
        np.save(self.path+'bottlenecks/bottleneck_features_validation_fold_'+str(j)+'.npy',bottlenecksFeaturesValidation)

    def saveWeights(self, topModel, finalModel, fold):
    	pass

    def saveData(self, history, resultTest, trainingTime, fold, topType):
    	pass


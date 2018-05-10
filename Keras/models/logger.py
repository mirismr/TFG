class Logger(object):
    def __init__(self, path):
        self.path = path

    """Create the directories structure for logs
    model/top_model/data_history
    model/top_model/weights
    model/top_model/general_data
    
    model/fine_tune/data_history
    model/fine_tune/weights
    model/fine_tune/general_data
    """
    def createStructure(self):
    	pass

    def saveClassDictionary(self, dictionary):
    	pass

    def saveWeights(self, topModel, finalModel, fold):
    	pass

    def saveData(self, history, resultTest, trainingTime, fold, topType):
    	pass


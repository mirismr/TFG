class Model(object):
    def __init__(self, kerasModel, inputSize, logger):
        self.kerasModel = kerasModel #cuando se use hacer copia
        self.inputSize = inputSize
        self.logger = logger
        self.batchSize = 16
        self.epochs = 50
        self.topModel = None

    """Return data and labels used for train and val with appropiate size"""
    def loadAndPreprocessData(self, path):
    	pass

    def generateBottlenecks(self, data, labels):
    	pass

    def buildTopModel(self, topType):
    	pass

    def trainTopModel(self, data, labels)
    	pass

    def buildFinalModel(self, topType, pathTop):
    	pass

    def testFinalModel(self):
    	pass

    def fineTune(self, data, labels, numLayersFreeze, bestTopType, pathBestTop);
    	pass
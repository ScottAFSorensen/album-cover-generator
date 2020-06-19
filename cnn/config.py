#Define paths for files
spectrogramsPath = './data/spec'
slicesPath = './data/slices/'
datasetPath = './data/dataset/'
rawDataPath = './data/raw/'

#Spectrogram resolution
pixelPerSecond = 100

#Slice parameters
sliceSize = 128

#Dataset parameters
filesPerGenre = 1000
validationRatio = 0.1
testRatio = 0.05

#Model parameters
batchSize = 128 
learningRate = 0.0001 
nbEpoch = 30

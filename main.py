import time
from sklearn.metrics import accuracy_score
from commonfunctions import *
##########Global Variables############
formsFolderName = "forms"
windowWidth=14
featuresCount = 13##change here
iterationsCount = 100
accuracy = 0

writers = labelData()
writerForms = labelForms()
candidateWriters = {k:v for k, v in writerForms.items() if len(v) > 2}

total = 0
# create output files
f = open("results.txt", "w+")
f.close()
f = open("time.txt", "w+")
f.close()

for iterationFolder in sorted(glob.glob("data/*")):
    trainingFormIDs, testingFormID, trainingLabels = readData(iterationFolder)

    xTrain = np.empty([0, featuresCount])
    yTrain = []
    xTest = np.empty([0, featuresCount])
    featuresVectors = []
    formsFeaturesVectors = np.empty([0, featuresCount])
    testFeaturesVectors = np.empty([0, featuresCount])

    ##########Processing Training forms#################

    t0 = time.time()
    xTrain, yTrain = getTrainingData(trainingFormIDs, trainingLabels, yTrain, formsFeaturesVectors)
    if xTrain is None:
        continue
    xTrain, mean, std = normalizeFeatures(xTrain)

    ########Processing Testing form###############

    xTest, processedSuccessfully = getTestFeatures(testingFormID, testFeaturesVectors)
    if processedSuccessfully == False:
        continue
    xTest = (xTest - mean) / std

    n_classes = len(np.unique(yTrain))
    uniqueClasses = np.unique(yTrain)

    classification = classify(xTrain, yTrain, xTest, uniqueClasses)
    t1 = time.time()

    writeOutput(classification, t1, t0)

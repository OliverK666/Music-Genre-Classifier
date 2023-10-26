import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from tempfile import TemporaryFile
import os
import math
import pickle
import random
import operator
from collections import defaultdict

def distance(instance1, instance2, k):
    distance = 0
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    distance += (np.dot(np.dot((mm2-mm1).transpose(), np.linalg.inv(cm2)), mm2-mm1))
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance


def getNeighbors(trainingset, instance, k):
    distances = []
    for x in range(len(trainingset)):
        dist = distance(trainingset[x], instance, k) + distance(instance,trainingset[x],k)
        distances.append((trainingset[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def nearestclass(neighbors):
    classVote = {}

    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1

    sorter = sorted(classVote.items(), key=operator.itemgetter(1), reverse=True)
    return sorter[0][0]


def getAccuracy(testSet, prediction):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == prediction[x]:
            correct += 1
    return 1.0 * correct / len(testSet)


def loadDataset(filename, split, trset, teset):
    with open('mydataset.dat','rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break
    for x in range(len(dataset)):
        if random.random() < split:
            trset.append(dataset[x])
        else:
            teset.append(dataset[x])

def featureExtraction():
    directory = 'C:/Users/Oli/PycharmProjects/audioClass/archive/Data/genres_original'
    f = open("mydataset.dat", "wb")
    i = 0

    for folder in os.listdir(directory):
        #print(folder)
        i += 1
        if i == 11:
            break
        for file in os.listdir(directory+"/"+folder):
            #print(file)
            try:
                (rate, sig) = wav.read(directory+"/"+folder+"/"+file)
                mfcc_feat = mfcc(sig, rate, winlen = 0.020, appendEnergy=False)
                covariance = np.cov(np.matrix.transpose(mfcc_feat))
                mean_matrix = mfcc_feat.mean(0)
                feature = (mean_matrix, covariance, i)
                pickle.dump(feature, f)
            except Exception as e:
                print("Got an exception: ", e, 'in folder: ', folder, ' filename: ', file)
    f.close()

    return feature


def determineAccuracy():
    length = len(testSet)
    predictions = []

    for x in range(length):
        predictions.append(nearestclass(getNeighbors(trainingSet, testSet[x], 5)))

    accuracy1 = getAccuracy(testSet, predictions)
    print(accuracy1)


# feature = featureExtraction()
dataset = []
trainingSet = []
testSet = []

loadDataset('mydataset.dat', 0.68, trainingSet, testSet)

# determineAccuracy()

results = defaultdict(int)

directory = "C:/Users/Oli/PycharmProjects/audioClass/archive/Data/genres_original"

i = 1
for folder in os.listdir(directory):
    results[i] = folder
    i += 1

x = ""

while 1:
    file_name = "C:/Users/Oli/PycharmProjects/audioClass/archive/Data/genres_original/"
    x = input("Input genre name: ")
    if x == "-1":
        break
    file_name += x
    file_name += "/"
    x = input("Input song name: ")
    file_name += x

    (rate, sig) = wav.read(file_name)
    mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
    covariance = np.cov(np.matrix.transpose(mfcc_feat))
    mean_matrix = mfcc_feat.mean(0)
    feature = (mean_matrix, covariance)

    pred = nearestclass(getNeighbors(dataset, feature, 5))
    print(results[pred])

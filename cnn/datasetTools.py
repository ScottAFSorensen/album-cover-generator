# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from PIL import Image
from random import shuffle
import numpy as np
import pickle

from imageFilesTools import getImageData
from config import datasetPath
from config import slicesPath

#Creates name of dataset from parameters
def getDatasetName(nbPerGenre, sliceSize):
    name = "{}".format(nbPerGenre)
    name += "_{}".format(sliceSize)
    #print(name)
    return name

#Creates or loads dataset if it exists
#Mode = "train" or "test"
def getDataset(nbPerGenre, genres, sliceSize, validationRatio, testRatio, mode):
    print("[+] Dataset name: {}".format(getDatasetName(nbPerGenre,sliceSize)))
    if not os.path.isfile(datasetPath+"train_X_"+getDatasetName(nbPerGenre, sliceSize)+".p"):
        print("[+] Creating dataset with {} slices of size {} per genre... ⌛️".format(nbPerGenre,
                                                                                     sliceSize))
        createDatasetFromSlices(nbPerGenre,
                                genres,
                                sliceSize,
                                validationRatio,
                                testRatio)
    else:
        print("[+] Using existing dataset")

    return loadDataset(nbPerGenre, genres, sliceSize, mode)

#Loads dataset
#Mode = "train" or "test"
def loadDataset(nbPerGenre, genres, sliceSize, mode):
    #Load existing
    datasetName = getDatasetName(nbPerGenre, sliceSize)

    n_bytes = 2**31
    max_bytes = 2**31 - 1

    
    if mode == "train":
        print("[+] Loading training and validation datasets... ")

        # train_X
        bytes_in = bytearray(0)
        input_size = os.path.getsize("{}train_X_{}.p".format(datasetPath, datasetName))
    
        with open("{}train_X_{}.p".format(datasetPath, datasetName), 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        train_X = pickle.loads(bytes_in)

        # train_y
        bytes_in = bytearray(0)
        input_size = os.path.getsize("{}train_y_{}.p".format(datasetPath, datasetName))
    
        with open("{}train_y_{}.p".format(datasetPath, datasetName), 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        train_y = pickle.loads(bytes_in)

        # validation_X
        bytes_in = bytearray(0)
        input_size = os.path.getsize("{}validation_X_{}.p".format(datasetPath, datasetName))
    
        with open("{}validation_X_{}.p".format(datasetPath, datasetName), 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        validation_X = pickle.loads(bytes_in)

        # validation_y
        bytes_in = bytearray(0)
        input_size = os.path.getsize("{}validation_y_{}.p".format(datasetPath, datasetName))
    
        with open("{}validation_y_{}.p".format(datasetPath, datasetName), 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        validation_y = pickle.loads(bytes_in)

        #train_X = pickle.load(open("{}train_X_{}.p".format(datasetPath,datasetName), "rb" ))
        #train_y = pickle.load(open("{}train_y_{}.p".format(datasetPath,datasetName), "rb" ))
        #validation_X = pickle.load(open("{}validation_X_{}.p".format(datasetPath,datasetName), "rb" ))
        #validation_y = pickle.load(open("{}validation_y_{}.p".format(datasetPath,datasetName), "rb" ))
        print("    Training and validation datasets loaded! ✅")
        return train_X, train_y, validation_X, validation_y

    else:
        print("[+] Loading testing dataset... ")

         # train_X
        bytes_in = bytearray(0)
        input_size = os.path.getsize("{}test_X_{}.p".format(datasetPath, datasetName))
    
        with open("{}test_X_{}.p".format(datasetPath, datasetName), 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        test_X = pickle.loads(bytes_in)

        # train_y
        bytes_in = bytearray(0)
        input_size = os.path.getsize("{}test_y_{}.p".format(datasetPath, datasetName))
    
        with open("{}test_y_{}.p".format(datasetPath, datasetName), 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        test_y = pickle.loads(bytes_in)
        
        #test_X = pickle.load(open("{}test_X_{}.p".format(datasetPath,datasetName), "rb" ))
        #test_y = pickle.load(open("{}test_y_{}.p".format(datasetPath,datasetName), "rb" ))
        print("    Testing dataset loaded! ✅")
        return test_X, test_y

#Saves dataset
def saveDataset(train_X, train_y, validation_X, validation_y, test_X, test_y, nbPerGenre, genres, sliceSize):
     #Create path for dataset if not existing
    if not os.path.exists(os.path.dirname(datasetPath)):
        try:
            os.makedirs(os.path.dirname(datasetPath))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    #SaveDataset
    print("[+] Saving dataset... ")
    datasetName = getDatasetName(nbPerGenre, sliceSize)

    n_bytes = 2**31
    max_bytes = 2**31 - 1

    bytes_out = pickle.dumps(train_X)
    with open("{}train_X_{}.p".format(datasetPath, datasetName), 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])

    bytes_out = pickle.dumps(train_y)
    with open("{}train_y_{}.p".format(datasetPath, datasetName), 'wb') as f_out:
         for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])
            

    
    bytes_out = pickle.dumps(validation_X)
    with open("{}validation_X_{}.p".format(datasetPath, datasetName), 'wb') as f_out:
         for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])


    
    bytes_out = pickle.dumps(validation_y)
    with open("{}validation_y_{}.p".format(datasetPath, datasetName), 'wb') as f_out:
         for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])
            

    bytes_out = pickle.dumps(test_X)
    with open("{}test_X_{}.p".format(datasetPath, datasetName), 'wb') as f_out:
         for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])

    
    bytes_out = pickle.dumps(test_y)
    with open("{}test_y_{}.p".format(datasetPath, datasetName), 'wb') as f_out:
         for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])
            
    print("    Dataset saved! ✅💾")
 
#Creates and save dataset from slices
def createDatasetFromSlices(nbPerGenre, genres, sliceSize, validationRatio, testRatio):
    data = []
    for genre in genres:
        print("-> Adding {}...".format(genre))
        #Get slices in genre subfolder
        filenames = os.listdir(slicesPath+genre)
        filenames = [filename for filename in filenames if filename.endswith('.png')]
        filenames = filenames[:nbPerGenre]
        #Randomize file selection for this genre
        shuffle(filenames)

        #Add data (X,y)
        for filename in filenames:
            imgData = getImageData(slicesPath+genre+"/"+filename, sliceSize)
            label = [1. if genre == g else 0. for g in genres]
            data.append((imgData,label))

    #Shuffle data
    shuffle(data)

    #Extract X and y
    X,y = zip(*data)

    #Split data
    validationNb = int(len(X)*validationRatio)
    testNb = int(len(X)*testRatio)
    trainNb = len(X)-(validationNb + testNb)

    #Prepare for Tflearn at the same time
    train_X = np.array(X[:trainNb]).reshape([-1, sliceSize, 258, 1])
    train_y = np.array(y[:trainNb])
    validation_X = np.array(X[trainNb:trainNb+validationNb]).reshape([-1, sliceSize, 258, 1])
    validation_y = np.array(y[trainNb:trainNb+validationNb])
    test_X = np.array(X[-testNb:]).reshape([-1, sliceSize, 258, 1])
    test_y = np.array(y[-testNb:])
    print("    Dataset created! ✅")

    #Save
    saveDataset(train_X, train_y, validation_X, validation_y, test_X, test_y, nbPerGenre, genres, sliceSize)

    return train_X, train_y, validation_X, validation_y, test_X, test_y

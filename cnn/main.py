# -*- coding: utf-8 -*-
import random
import string
import os
import sys
import numpy as np

from model import createModel
from datasetTools import getDataset
from config import rawDataPath
from config import slicesPath
from config import batchSize
from config import filesPerGenre
from config import nbEpoch
from config import validationRatio, testRatio
from config import sliceSize

from songToData import createSlicesFromAudio
from imageFilesTools import getImageData

import argparse
import tensorflow as tf

from collections import Counter

def get_most_common(array):
        array = array.tolist()
        #counts = []

        g = [x[0] for x in array]
        
        counts = Counter(g)
        return counts.most_common(3)

        #ret = heapq.nlargest(3, counts)
        #return

genres = {
            "Alternative": 0,
            "Ambient Electronic": 1,
            "Avant-Garde": 2,
            "Blues": 3,
            "Classical": 4,
            "Electronic": 5,
            "Experimental": 6,
            "Folk": 7,
            "Hip-Hop": 8,
            "Indie-Rock": 9,
            "International": 10,
            "Jazz": 11,
            "Lo-Fi": 12,
            "Metal": 13,
            "Pop": 14,
            "Psych-Rock": 15,
            "Punk": 16,
            "Rock": 17,
            "Techno": 18,
            "Trip-Hop": 19,
            }

genres = list(genres)


parser = argparse.ArgumentParser()
parser.add_argument("mode", help="Trains or tests the CNN", nargs='+', choices=["train","test",
                                                                                "slice", "predict"])
args = parser.parse_args()

print("--------------------------")
print("| ** Config ** ")
print("| Validation ratio: {}".format(validationRatio))
print("| Test ratio: {}".format(testRatio))
print("| Slices per genre: {}".format(filesPerGenre))
print("| Slice size: {}".format(sliceSize)) 
print("--------------------------")

if "slice" in args.mode:
        createSlicesFromAudio(rawDataPath, slicesPath)
        sys.exit()

#List genres
nbClasses = 20 

#Create model 
model = createModel(nbClasses, sliceSize)


if "train" in args.mode:
	#Create or load new dataset
	train_X, train_y, validation_X, validation_y = getDataset(filesPerGenre,
                                                                  genres,
                                                                  sliceSize,
                                                                  validationRatio,
                                                                  testRatio,
                                                                  mode="train")

	#Define run id for graphs
	run_id = "MusicGenres - " + str(batchSize) + " " + ''.join(random.SystemRandom().
                                                                   choice(string.ascii_uppercase)
                                                                   for _ in range(10))

	#Train the model
	print("[+] Training the model...")
	model.fit(train_X, train_y, n_epoch=nbEpoch,
                  batch_size=batchSize, shuffle=True,
                  validation_set=(validation_X, validation_y),
                  snapshot_step=100, show_metric=True, run_id=run_id)
        
	print("    Model trained! ✅")
	#Save trained model
	print("[+] Saving the weights...")
	model.save('musicDNN.tflearn')
	print("[+] Weights saved! ✅💾")

if "test" in args.mode:

	#Create or load new dataset
	test_X, test_y = getDataset(filesPerGenre,
                                    genres,
                                    sliceSize,
                                    validationRatio,
                                    testRatio,
                                    mode="test")

	#Load weights
	print("[+] Loading weights...")
	model.load('musicDNN.tflearn')
	print("    Weights loaded! ✅")

	testAccuracy = model.evaluate(test_X, test_y)[0]
	print("[+] Test accuracy: {} ".format(testAccuracy))


if "predict" in args.mode:
        # create slices from audio
        path = './data/input/'
        s_path = './data/input/slices/'
        createSlicesFromAudio(path)

        slices = os.listdir(s_path)

        # song slices
        data = []

        [data.append(getImageData(s_path+f, 128)) for f in slices if f.endswith('.png')]
        X = np.array(data).reshape([-1, 128, 258, 1])

        #Load weights
        print("[+] Loading weights...")
        model.load('musicDNN.tflearn')
        print("    Weights loaded!")

                  
        pred = model.predict_label(X)

        [first, second, third] = [x[0] for x in get_most_common(pred)]


        #[print(f.astype(np.uint8)) for f in pred]
        print("Detected genre: " + genres[first])
        print("Second most common detection: " + genres[second])

        
        





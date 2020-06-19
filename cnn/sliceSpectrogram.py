# Import Pillow:
from PIL import Image
import os.path
import cv2
from config import spectrogramsPath, slicesPath
import songToData
import os
import numpy as np

#Slices all spectrograms
def createSlicesFromSpectrograms(filename, desiredSize, path):
        spectrogramsPath = path+'spectrogram/'

        l = os.listdir(spectrogramsPath)

        for i in range(len(l)):
                if l[i].endswith('.png'):
                         sliceSpectrogram(filename,desiredSize, path+'slices/')



#Creates slices from spectrogram
#TODO Improvement - Make sure we don't miss the end of the song
def sliceSpectrogram(filename, desiredSize, path):
        slicePath = path
 
        #genre = filename.split("_")[0] 	#Ex. Dubstep_19.png
        spectrogramsPath = '/Users/henrikforberg/Desktop/project_/cnn/data/input/spectrogram/'
        # Load the full spectrogram
        #img = Image.open(spectogramsPath+filename)

        #print(spectrogramsPath+filename)
        # Load the full spectrogram
        img = cv2.imread(spectrogramsPath+filename, cv2.IMREAD_GRAYSCALE)
        height, width = img.shape
        img = cv2.resize(img, (width, 129)) 
        tempogram = songToData.get_tempogram(os.getcwd()+'/data/input/x.mp3')
        
        tempogram = cv2.resize(tempogram, (width, 129))

        
        img = img/255
        img = np.vstack((img, tempogram))

        height, width = img.shape
        nbSamples = int(width/desiredSize)
        width - desiredSize

        #Create path if not existing
        if not os.path.exists(os.path.dirname(slicePath)):
                try:
                        os.makedirs(os.path.dirname(slicePath))
                except OSError as exc: # Guard against race condition
                        if exc.errno != errno.EEXIST:
                                raise

        img = img*255
        #For each sample
        for i in range(nbSamples):
                #print ("Creating slice: ", (i+1), "/", nbSamples, "for", filename)
                startPixel = i*desiredSize
                imgTmp = img[:, startPixel:desiredSize*(i+1)]
                cv2.imwrite(slicePath+"{}_{}.png".format(filename[:-4],i), imgTmp)
   

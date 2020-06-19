# -*- coding: utf-8 -*-
from subprocess import Popen, PIPE, STDOUT
import os
from PIL import Image
import eyed3

from sliceSpectrogram import createSlicesFromSpectrograms
from audioFilesTools import isMono, getGenre
from config import rawDataPath
from config import spectrogramsPath
from config import pixelPerSecond
import librosa
import cv2

#Tweakable parameters
desiredSize = 128

#Define
currentPath = os.path.dirname(os.path.realpath(__file__))

#Remove logs
eyed3.log.setLevel("ERROR") 


def get_tempogram(filename):
        y, sr = librosa.load(filename)
        hop_length = 512
        oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        tempogram = librosa.feature.tempogram(onset_envelope=oenv,
                                              sr=sr, hop_length=hop_length)
        return tempogram


#Create spectrogram from mp3 files
def createSpectrogram(filename, newFilename):

        command = "sox '{}' '{}.mp3' remix 1,2".format(filename, currentPath+"/data/input/"+newFilename)
        p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE,
                  stderr=STDOUT, close_fds=True, cwd=currentPath+"/")
        output, errors = p.communicate()


        #print(output)
        if errors:
                print (errors)

        #print(output)
        #Create spectrogram
        filename.replace(".mp3","")


        pa = '/Users/henrikforberg/Desktop/project_/cnn/data/input/spectrogram/'
        command = "sox '/{}.mp3' -n spectrogram -Y 200 -X {} -m -r -o '{}.png'".format(currentPath+
                                                                                       "/data/input/"+
                                                                                       newFilename,
                                                                                       pixelPerSecond,
                                                                                       pa + newFilename)

        #print(command)
        p = Popen(command,
                  shell=True,
                  stdin=PIPE,
                  stdout=PIPE,
                  stderr=STDOUT,
                  close_fds=True,
                  cwd=currentPath+"/")

        output, errors = p.communicate()

        #print(output)

        if errors:
                print (errors)

	#print("___________")
	#print(command)
	#print(currentPath)
	#print("___________")
	#Remove tmp mono track
	#os.remove("/tmp/{}.mp3".format(newFilename))

#Creates .png whole spectrograms from mp3 files
def createSpectrogramsFromAudio(path, spec_path):
        rawDataPath = path
        spectrogramsPath = spec_path
        genresID = dict()
        #files = os.listdir(rawDataPath)
        #files = [file for file in files if file.endswith(".mp3")]
        nbFiles = 1 #len(files)

        #print("TEST 1")
        #Create path if not existing
        if not os.path.exists(os.path.dirname(spectrogramsPath)):
                try:
                        os.makedirs(os.path.dirname(spectrogramsPath))
                except OSError as exc: # Guard against race condition
                        if exc.errno != errno.EEXIST:
                                raise 

        createSpectrogram(path, "x")


#Whole pipeline .mp3 -> .png slices
def createSlicesFromAudio(path):

        filename = [f for f in os.listdir(path) if f.endswith('.mp3')]

        print ("Creating spectrograms...")
        createSpectrogramsFromAudio(path+str(filename)[2:-2], path+'spectrogram/')
        print ("Spectrograms created!")

        print ("Creating slices...") 
        createSlicesFromSpectrograms('x.png', 128, path)
        print ("Slices created!")

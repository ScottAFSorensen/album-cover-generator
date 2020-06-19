import numpy as np
import os
from PIL import Image, ImageOps
from pathlib import Path
from random import shuffle, randint
import sys
import scipy.misc
import glob
#from utils import *

class covers:
    def __init__(self, folder, batch_size, cat_dim):
        

        self.data_dir = folder
        self.directory = './genres/'
        self.batch_size = batch_size
        self.list = []
        self.num_images = 0
        self.cat_dim = cat_dim

        self.generate_list()
        self.generate_categories()

    def generate_categories(self):
        self.cat = np.zeros([self.num_images])
        num = 0
        rand = False



        if 'all covers' in self.directory:
            rand = True
        for l in self.list:
            if rand:
                print("SOMETHING GOT WRONG")
                
            elif 'Alternative' in l:
                self.cat[num] = 0
            elif 'Ambient Electronic' in l:
                self.cat[num] = 1
            elif 'Avant-Garde' in l:
                self.cat[num] = 2
            elif 'Blues' in l:
                self.cat[num] = 3
            elif 'Classical' in l:
                self.cat[num] = 4
            elif 'Electronic' in l:
                self.cat[num] = 5
            elif 'Experimental' in l:
                self.cat[num] = 6
            elif 'Folk' in l:
                self.cat[num] = 7
            elif 'Hip-Hop' in l:
                self.cat[num] = 8
            elif 'Indie-Rock' in l:
                self.cat[num] = 9
            elif 'International' in l:
                self.cat[num] = 10
            elif 'Jazz' in l:
                self.cat[num] = 11
            elif 'Lo-Fi' in l:
                self.cat[num] = 12
            elif 'Metal' in l:
                self.cat[num] = 13
            elif 'Pop' in l:
                self.cat[num] = 14
            elif 'Psych-Rock' in l:
                self.cat[num] = 15
            elif 'Punk' in l:
                self.cat[num] = 16
            elif 'Rock' in l:
                self.cat[num] = 17
            elif 'Techno' in l:
                self.cat[num] = 18
            elif 'Trip-Hop' in l:
                self.cat[num] = 19

            num += 1

    def generate_list(self):
        txt = open('image_list.txt', 'w')
        txt.truncate(0)
        files = glob.glob(self.data_dir + '*/*')
        
        for filename in files:
            
            out = (str(filename))
            
            # Check if the file is a valid image
            try:
                img = Image.open(filename)
                self.list.append(out)
                txt.write(out + '\n')
            except Exception as e:
                print (e)
        txt.close()
        self.num_images = len(self.list)


    def get_image(self, start_idx=None):
        for i in range(len(self.list)):
            if start_idx is None or start_idx <= i:
                img = Image.open(self.list[i])
                category = self.cat[i]

                try:
                    img2 = ImageOps.fit(img, [100, 100], Image.ANTIALIAS)

                    rgbimg = img2.convert('RGB')
                    rgbimg = np.array(rgbimg, dtype=np.float32)/127.5 - 1
                except Exception as e:
                    continue
                yield i, rgbimg, category


    # get next batch of photos
    def batched_images(self, start_idx=None):
        batch, next_idx, data_counter = None, None, 0
        for idx, image, img_cat in self.get_image(start_idx):
            if batch is None:
                batch = np.empty((self.batch_size, 100, 100, 3))
                batch_cat = np.empty((self.batch_size, self.cat_dim), dtype = int)
                next_idx = 0
                data_counter = 0
            if image is not None:
                batch[data_counter] = image
                batch_cat[data_counter] = img_cat
                data_counter += 1
            else:
                print("exception")
            next_idx += 1
            if data_counter == self.batch_size:
                yield next_idx + 1, batch, batch_cat
                batch = None

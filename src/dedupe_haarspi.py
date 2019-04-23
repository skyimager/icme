#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
To know more on HaarPSI: http://www.haarpsi.org/
"""
import os
import numpy as np
from keras.preprocessing import image

from src.dedupe.haarPSI import haar_psi_numpy


def prepare_images(files_list):
    image_list = []
    np_files_list = []
    for file_path in files_list:
        
        try:
            img = image.load_img(file_path, target_size=(200, 128,3))
            np_img = image.img_to_array(img)
            image_list.append(np_img)
            np_files_list.append(file_path)
        except:
            print(file_path)
    
    np_image_list = np.array(image_list)
    np_files_list = np.array(np_files_list)
    
    
    return np_image_list, np_files_list


def find_duplicates_using_haarpsi(files_list):
    
    np_image_list, np_files_list = prepare_images(files_list)
    
    duplicates = dict()
    for i in range(len(np_image_list)):
        for j in range(i,len(np_image_list)):
            if i !=j:
                similarity = haar_psi_numpy(np_image_list[i], np_image_list[j], True)
                if similarity >0.80:
                    if np_files_list[i] in duplicates:
                        duplicates[os.path.basename(np_files_list[i])[:-4].upper()].append([os.path.basename(np_files_list[j]).upper(), str(similarity)])
                        del np_image_list[j]
                    else:
                        duplicates[os.path.basename(np_files_list[i])[:-4].upper()] = [os.path.basename(np_files_list[j]).upper(), str(similarity)]
                        del np_image_list[j]
    
    return duplicates
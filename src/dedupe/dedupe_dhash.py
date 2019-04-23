#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
from hashlib import md5


def resize(image, height=200, width=128):
    
    row_res = cv2.resize(image,(height, width), interpolation = cv2.INTER_AREA).flatten()
    col_res = cv2.resize(image,(height, width), interpolation = cv2.INTER_AREA).flatten('F')
    
    return row_res, col_res


def intensity_diff(row_res, col_res):
    
    difference_row = np.diff(row_res)
    difference_col = np.diff(col_res)
    difference_row = difference_row > 0
    difference_col = difference_col > 0
    
    return np.vstack((difference_row, difference_col)).flatten()

def file_hash(array):
    return md5(array).hexdigest()

def difference_score(image, height = 200, width = 128):
    
    row_res, col_res = resize(image, height, width)
    difference = intensity_diff(row_res, col_res)
    
    return difference

def find_duplicates_using_dhash(file_list):
    ds_dict = {}
    duplicates = []
    hash_ds = []
    
    for full_path in file_list:
        
        image = cv2.imread(full_path,0)
        
        if image is None:
            continue
        
        ds = difference_score(image)
        hash_ds.append(ds)
        filehash = md5(ds).hexdigest()
        
        if filehash not in ds_dict:
            ds_dict[filehash] = os.path.basename(full_path)[:-4]
        else:
            try:
                duplicates[ds_dict[filehash].upper()].append(os.path.basename(full_path)[:-4].upper())
            except:
                duplicates[ds_dict[filehash].upper()] = []
                duplicates[ds_dict[filehash].upper()].append(os.path.basename(full_path)[:-4].upper())
    
    return  duplicates
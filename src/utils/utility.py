#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from skimage.measure import compare_ssim

def save_json(file_name, dict_to_dump):
    with open(file_name, 'w') as outfile:
        json.dump(dict_to_dump, outfile)
        
        
def ssim_similarity(img1, img2):
    return compare_ssim(img1, img2, multichannel = True)
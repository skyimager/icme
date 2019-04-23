#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import itertools
from collections import defaultdict
from skimage.measure import compare_ssim 

from src.dedupe.kmeans import KMeans
from src.utils.utility import ssim_similarity


def duplicated(img_file1, img_file2, thresh = 0.9):
    """
    Measures duplication using SSIM metric (because that is what we used in clustering)
    We should use the same distance function in clustering step and finding duplication
    step. Returns a Boolean of whether or not img1 and img2 are same (Default Threshold 0.9)
    """
    img1 = cv2.resize(cv2.imread(img_file1), (256, 256))
    img2 = cv2.resize(cv2.imread(img_file2), (256, 256))
    
    return compare_ssim(img1, img2, multichannel = True) > thresh


def find_duplicates_using_kmeans(img_files):
    _, clusters = KMeans(img_files, distance_function = ssim_similarity).k_means_clustering()
    
    #images_in_clusters is a dict of all images in a cluster
    images_in_clusters = defaultdict(list)

    for i, img_file in enumerate(img_files):
        images_in_clusters[clusters[i]].append(img_file)

#    duplicates = {}
    total_duplicates = defaultdict(list)
    for clust in list(set(clusters)):
        images_in_cluster = images_in_clusters[clust]
        for comb in itertools.combinations(images_in_cluster, 2):
            file1, file2 = comb[0], comb[1]
            if duplicated(file1, file2):
                total_duplicates[os.path.basename(file1[:-4]).upper()].append(os.path.basename(file2[:-4]).upper())
    
    return total_duplicates


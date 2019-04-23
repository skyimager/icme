#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Standard imports
import os
import importlib
import numpy as np
from keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

#Custom imports
from src import config

def load_model(model_name, weights_path):
    
    build = getattr(importlib.import_module(model_name),"build")
    model = build(input_shape=(200, 128, 3))
    model.load_weights(weights_path, by_name=True)
    
    preprocess = getattr(importlib.import_module(model_name),"preprocess_input")
    
    return model, preprocess


def extract_features(model, preprocess, files_list, save=True):
    feature_list = []
    new_list = []
    
    for file in files_list:
        try:
            img = image.load_img(file, target_size=(200, 128,3))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess(img_data)
    
            feature_vector = model.predict(img_data)
            feature_np = np.array(feature_vector)
            
            new_list.append(file)
            feature_list.append(feature_np.flatten())
        except:
            print(file)
            
    feature_list_np = np.array(feature_list)
    files_list_np = np.array(new_list)
    
    if save:
        np.save("data/files_list", files_list_np , allow_pickle=True)
        np.save("data/features_list", feature_list_np , allow_pickle=True)
    
    return files_list_np, feature_list_np


def find_duplicates_using_fe(img_files):
    
    model, preprocess = load_model(config.model, config.weights_path)        
    files_list_np, feature_list_np = extract_features(model, preprocess, img_files)
    
    duplicates = dict()
    for i in range(len(feature_list_np)):
        for j in range(i,len(feature_list_np)):
            if i !=j:
                similarity = cosine_similarity(feature_list_np[i].reshape(1,-1),feature_list_np[j].reshape(1,-1))
                if similarity >0.80:
                    if files_list_np[i] in duplicates:
                        duplicates[os.path.basename(files_list_np[i])[:-4].upper()].append([os.path.basename(files_list_np[j]).upper(), str(similarity)])
                        del feature_list_np[j]
                    else:
                        duplicates[os.path.basename(files_list_np[i])[:-4].upper()] = [os.path.basename(files_list_np[j]).upper(), str(similarity)]
                        del feature_list_np[j]
    return duplicates
                        

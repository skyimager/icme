#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def extract_category(dataset, category="Tops"):
    
    dataset = dataset.dropna(subset = ['categories']) 
    cropped_dataset = dataset.loc[dataset['categories'].str.endswith(str('>')+str(category))]
    
    return cropped_dataset

def extract_primaryimageurlstr(dataset):
    
    dataset = dataset.dropna(subset = ['imageUrlStr']) 
    dataset['primaryImageUrlStr'] = [str(item).split(';')[0] for item in dataset.imageUrlStr]
    
    return dataset


def df_drop_duplicates(dataset):
    
    dataset = dataset.drop_duplicates(subset='productId', keep="first")
    dataset = dataset.drop_duplicates(subset='productUrl', keep="first")
    dataset = dataset.drop_duplicates(subset='primaryImageUrlStr', keep="first")
    
    return dataset


def df_remove_unwanted(dataset, unwanted):
    
    dataset.drop(unwanted,axis = 1,inplace = True)
    
    return dataset
    
    
    
    
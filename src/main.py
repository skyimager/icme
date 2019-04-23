#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Standard imports
import os, sys, glob
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
sys.path.append(os.path.abspath('./src/networks'))

import pandas as pd

#Custom imports
from src import config
from src.huew.pre_process import extract_category, extract_primaryimageurlstr, df_drop_duplicates, df_remove_unwanted
from src.utils.genrate_json import get_json_of_duplicates
from src.utils.utility import save_json
from src.utils.download_images import multi_download_images
from src.dedupe.dedupe_kmeans import find_duplicates_using_kmeans
from src.dedupe.dedupe_dhash import find_duplicates_using_dhash
from src.dedupe.dedupe_fe import find_duplicates_using_fe
from src.dedupe.dedupe_haarspi import find_duplicates_using_haarpsi

if __name__ == "__main__":
    
    dataset = pd.read_csv(config.dataset_file)
    
    #Extracting subset tops
    tops_dataset = extract_category(dataset, category="Tops") 
    
    #Extracting primary/first image url from image urls under "imageUrlStr"
    tops_dataset = extract_primaryimageurlstr(tops_dataset)

    #extracting primary dublicate list based on primary image url
    dub_list = get_json_of_duplicates(tops_dataset, key_col='productId', val_col='primaryImageUrlStr')    
    save_json(dub_list,'data/priliminary_duplicates.json')
    
    #Data pre-processing
    tops_dataset = df_drop_duplicates(tops_dataset)
    
    if not os.path.exists('data/tops.csv'):
        tops_dataset.to_csv('data/tops.csv', index = False) #Original 5GB reduced to 113MB
    
    #Columns to be removed as they do not add value to differentiability between data points. 
    #for now this was manual and crude. A better way has to be identified. 
    unwanted_cols = ['description', 'categories','sellingPrice', 'specialPrice', 
                     'inStock', 'codAvailable', 'offers', 'discount',
                     'shippingCharges', 'deliveryTime', 'sizeUnit','storage', 
                     'displaySize','detailedSpecsStr','specificationList','sellerName', 
                     'sellerAverageRating', 'sellerNoOfRatings', 'sellerNoOfReviews', 
                     'sleeve', 'neck', 'idealFor']
    
    #not sure if other image urls are required or not. So kept it seperate. For now removing it. 
    tops_dataset.drop(['imageUrlStr'],axis = 1,inplace=True)
    tops_dataset = df_remove_unwanted(tops_dataset, unwanted_cols) 
    
    if not os.path.exists('data/refine_tops.csv'):
        tops_dataset.to_csv('data/refine_tops.csv', index = False) #Original 5GB reduced to 49MB    
    
    #Downloading images using mutliprocessing
    urls = list(tops_dataset.primaryImageUrlStr)
    names = list(config.img_dir + name.lower() +'.jpg' for name in tops_dataset.productId)
    iter_list = [(urls[i], names[i]) for i in range(len(urls))]
    if config.to_download:
        multi_download_images(iter_list) #86,022 images downloaded
    
    #Run this from terminal: find . -name "*.jpg" -size -1k -delete
    
    #finding duplicates
    imgs_dir = config.img_dir + '*.jpg'
    img_files = glob.glob(imgs_dir)[:10000] #taking only the first 100 images
    
    
    #Strategy-1
    duplicates_kmeans = find_duplicates_using_kmeans(img_files)
    save_json(duplicates_kmeans, 'data/duplicates_with_kmeans.json')
    
    #Strategy-2
    duplicates_dhash = find_duplicates_using_dhash(img_files)
    save_json(duplicates_dhash, 'data/duplicates_with_dhash.json')
    
    #Strategy-3
    duplicates_fe = find_duplicates_using_fe(img_files)
    save_json(duplicates_fe, 'data/duplicates_with_squeezenet.json')
    
    #Strategy-4
    duplicates_fe = find_duplicates_using_haarpsi(img_files)
    save_json(duplicates_fe, 'data/duplicates_with_haarspi.json')
    
    
    
    
    
    

    
    
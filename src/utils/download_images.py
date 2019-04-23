#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import requests
import multiprocessing


def download_files(web_link, file_name):
    if os.path.isfile(file_name):
        pass
    else:
        try:
            image = requests.get(web_link, timeout=10, allow_redirects=True)
            open(file_name, 'wb').write(image.content)
        except:
            print("Could not download file at {}".format(web_link))
            

def multi_download_images(iter_list):
    pool = multiprocessing.Pool()
    pool.starmap(download_files, iter_list)

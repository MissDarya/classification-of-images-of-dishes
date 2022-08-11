#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 19:19:36 2022

@author: darya
"""

import os
import numpy as np
import random
import shutil

def getFilePath(path, fileSubName):
    f = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        f.extend(filenames)
        break
    f_ts_SubName = list(filter(lambda k: fileSubName in k, f))
    return f

path = '/media/darya/Queen/BMSTU/CV/Lab2/Food'
pathNewTrain = '/media/darya/Queen/BMSTU/CV/Lab2/SplitFood/train'
pathNewVal = '/media/darya/Queen/BMSTU/CV/Lab2/SplitFood/val'

listNames = ['Bread', 'Dessert', 'Meat', 'Soup']

for i in range(len(listNames)):

    listImg = getFilePath(path = os.path.join(path, listNames[i]), fileSubName = '-')
    
    tarinImg = random.sample(listImg, k = np.int(0.8*len(listImg)))
    valImg = list(set(listImg) - set(tarinImg))
    
    
    for j in range(len(tarinImg)):
        shutil.copy(os.path.join(path, listNames[i], tarinImg[j]), os.path.join(pathNewTrain, listNames[i]))
        
    for k in range(len(valImg)):
        shutil.copy(os.path.join(path, listNames[i], valImg[k]), os.path.join(pathNewVal, listNames[i]))
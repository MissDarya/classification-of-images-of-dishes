#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 02:42:06 2022

@author: darya
"""
import torch
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

class_names = ['Bread', 'Dessert', 'Meat', 'Soup']
labelList = torch.load('/media/darya/Queen/BMSTU/CV/Lab2/label.pt',map_location=torch.device('cpu') )
outputList = torch.load('/media/darya/Queen/BMSTU/CV/Lab2/outut.pt',map_location=torch.device('cpu') )


#%%Матрица ошибок
cf_matrix = confusion_matrix(labelList, outputList)
categories = class_names
sns.heatmap(cf_matrix, annot=True,cmap='Blues',cbar=False,fmt="d",xticklabels = categories,yticklabels=categories)


#%%Функция потерь, точность
df = pd.read_csv('/media/darya/Queen/BMSTU/CV/Lab2/df.csv')

fig,ax = plt.subplots(figsize=(7,5))
plt.plot(df['trainLoss'], label='Тренировочный набор данных')
plt.plot( df['valLoss'], label='Тестовый набор данных')
plt.legend(loc='upper right')
ax.set_xlabel('Эпоха')
ax.set_ylabel('Функция потерь')


fig,ax = plt.subplots(figsize=(7,5))
plt.plot(df['trainAcc'], label='Тренировочный набор данных')
plt.plot( df['valAcc'], label='Тестовый набор данных')
plt.legend(loc='lower right')
ax.set_xlabel('Эпоха')
ax.set_ylabel('Точность')
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 16:15:49 2021

@author: ichamseddine
"""


#%% Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import warnings
import os
import seaborn as sns
from scipy.stats import mannwhitneyu, fisher_exact, chi2_contingency
from sklearn.preprocessing import StandardScaler


warnings.simplefilter(action='ignore', category=FutureWarning)
    


    
#%% data

filename = 'feature correlation.png'

# Prepare data
df = pd.read_excel("../dataset.xlsx")

features = ['sex', 'age', 
            'cirrhosis', 'cirrhosis_etiology', 'liversize', 'PVT', 'CP0',
            'newDx',  'lesion_size', 'lesion_number', 'GTV', 
            'ALB0', 'BIL0', 'PLT0', 'AFP0', 'ALC0', 
            'proton', 'Fx', 'TD', 'EUD','EQD2_MLD', 
            'EQD2_V5', 'EQD2_V10', 'EQD2_V15', 'EQD2_V20', 'EQD2_V25',
            'EQD2_V30', 'EQD2_V35', 'EQD2_V40', 'EQD2_V45', 'EQD2_V50']

X=df[features]

conts = []
for xx in features:
    if len(X[xx].unique())>5:
        conts.append(xx)
        
std = StandardScaler()
X[conts] = std.fit_transform(X[conts])



#%% 
endpoints = ['SRVy1', 'NLFy1', 'CP2plus', 'ALBI1plus', 'RIL', 'LF', 'LFLRF']
# endpoints = ['SRVy1', 'NLFy1', 'LRFy1', 'DMy1', 'CP2plus', 'ALBI1plus', 'RIL']
pmap = pd.DataFrame(np.ones((len(endpoints)*len(features), 3)), columns = ['feature', 'outcome', 'p'])


ii = -1                     
for y_name in endpoints:
    
    df = df.dropna(subset = [y_name])
    df = df.reset_index(drop=True)
    y = df[y_name]
    for x_name in features:
        ii += 1
        x = df[x_name]
        if len(x.unique())>10:
            gr0 = x[y==0]
            gr1 = x[y==1]
            _, p = mannwhitneyu(gr0,gr1)
            # print(f"{x_name:10} \t MannWhitney \t {p}")
            
        elif len(x.unique())==2:
            tbl = pd.crosstab(x, y, margins = False)
            _, p = fisher_exact(tbl)
            # print(f"{x_name:10} \t Fisher Exact \t {p}")
            
        else:
            tbl = pd.crosstab(x, y, margins = False)
            _, p, _, _ = chi2_contingency(tbl)
            # print(f"{x_name:10} \t Pearson Chi2 \t {p}")
            
        pmap.feature[ii] = x_name
        pmap.outcome[ii] = y_name
        pmap.p[ii]       = p

pmap = pmap.pivot(index="feature", columns = 'outcome', values = 'p')
#%% plot heatmap
fig = plt.figure(figsize=(8,8))
sns.heatmap(pmap, cmap='gray', annot = True)   
plt.savefig("univariate_pvals.png", format = 'png', dpi = 600) 
    
    
    
    
    
    
    
    


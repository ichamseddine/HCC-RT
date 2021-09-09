#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 16:15:49 2021

@author: ichamseddine
"""


# ============================================================================
#                                 LIBRARIES
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import warnings
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler

#from xgboost import XGBClassifier
#from sklearn.svm import SVC
#from sklearn.model_selection import StratifiedKFold
#from sklearn.metrics import roc_auc_score
from scipy import stats

warnings.simplefilter(action='ignore', category=FutureWarning)
    


    
# ============================================================================
#                                 MAIN
# ============================================================================

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

fig = plt.figure(figsize=(8, 7))
corr = X.corr()
# plot the heatmap
sns.heatmap(abs(corr),
xticklabels=corr.columns,
yticklabels=corr.columns,
cmap="Greys", 
vmin=0, vmax=1)
plt.savefig(filename, format='png', dpi=600, bbox_inches='tight')
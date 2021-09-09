#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 11:30:29 2021

@author: ichamseddine
"""
#%%
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#
import ibrahim_functions as ibr
import hcc_models as hcc

#
from sklearn.preprocessing import StandardScaler
from sklearn.impute import IterativeImputer
#
import pickle

#%%
def get_rate(df):

    df = df.dropna(subset = ['CODy1'])

    N = df.shape[0] # number of patients
    rate = [None]*6
    for ii in range(6):
        cat = ii
        rate[ii] = sum(df.CODy1==cat)/N
    
    return rate



#%% Read data


df = pd.read_excel('../../mda_dataset.xlsx')

dfout = pd.DataFrame(columns = ['category'], data = range(6))

dfout['Overall\npopulation'] = get_rate(df)

print('(overall, %i)'%df.shape[0])


dfout1 = pd.DataFrame(columns = ['category'], data = range(6))

dfout1['overall'] = get_rate(df)


#%% get risk groups of survival
endpoint = 'SRVy1'
clfname = 'Logit'

features = hcc.optimal_feature_set(endpoint, clfname)

contvars = []
for xx in features:
    if len(df[xx].unique()) > 10:
        contvars.append(xx)

filename = f"FinalModels/{endpoint}_{clfname}_tuned.pkl"
clf = pickle.load(open(filename,'rb'))
        
        
X = df[features]
X = IterativeImputer().fit_transform(X) 
X = pd.DataFrame(X, columns = features)


X[contvars]  = StandardScaler().fit_transform(X[contvars])
pred_prob = clf.predict_proba(X)[:, 1]


risk_group = pd.qcut(pred_prob, 10, labels = False, duplicates = 'drop')
df_hi = df.iloc[risk_group == 0] # high risk of mortality = low risk of survival 
df_lo = df.iloc[risk_group == risk_group.max()] # and vice versa

dfout[f'Low-risk'] = get_rate(df_lo)
dfout[f'High-risk'] = get_rate(df_hi)

PtHi = df_hi.index
PtLo = df_lo.index

dfout = dfout.iloc[[5,1,2,3,4,0]]
#%% Define categories
dfout.loc[dfout['category']==5, 'category']='Alive'
dfout.loc[dfout['category']==0, 'category']='Unknown cause of death'
dfout.loc[dfout['category']==1, 'category']='Liver failure (no liver progression)'
dfout.loc[dfout['category']==2, 'category']='Liver failure (due to liver progression)'
dfout.loc[dfout['category']==3, 'category']='Non-liver related disease progression'
dfout.loc[dfout['category']==4, 'category']='Non-HCC related'
dfout = dfout.set_index('category')

#%% Plot
# clr={"Unknown": [40/255,79/255,118/255], \
#      "Liver failure (no liver progression)"    : [71/255,151/255,209/255],\
#      "Liver failure (due to liver progression)": "slategray", \
#      "Non-liver related disease progression"   : "goldenrod",\
#      "Non-HCC related"                         : [55/255, 118/255, 130/255]}



clr={"Alive"                                   : 'gray',\
     "Unknown cause of death"                  : 'gray', \
     "Liver failure (no liver progression)"    : 'w',\
     "Liver failure (due to liver progression)": "w", \
     "Non-liver related disease progression"   : "w",\
     "Non-HCC related"                         : 'w'}

# hatch
htc={"Alive"                                   : " ",\
     "Unknown cause of death": "x", \
     "Liver failure (no liver progression)"    : " ",\
     "Liver failure (due to liver progression)": " ", \
     "Non-liver related disease progression"   : "/",\
     "Non-HCC related"                         : "o"}

ptr = []
plt.figure(figsize=(4,2))
ax = dfout.T.plot.barh(stacked=True, color = clr, edgecolor='k', width = .8)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlabel('Proportion of cause of death')

bars = ax.patches
patterns =(' ', ' ', '///', '|||','oo','///', 'xxx')
hatches = [p for p in patterns for i in range(dfout.shape[1])]
# hatches = ''.join(h*len(df) for h in 'x/O.')
# hatches = []
# for ii i
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)


plt.legend(loc='center top', bbox_to_anchor=(0.015, 1))
plt.savefig('COD_by_riskgroup.png', format='png', dpi=600, bbox_inches='tight')





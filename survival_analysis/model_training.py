#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 19:27:17 2021

@author: ichamseddine
"""

#%% Libraries

#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

#
from sksurv.util import Surv

#
import ibrahim_functions as ibr
import hcc_survmodels as hcc

#
from scipy.stats import mannwhitneyu

#
import os
import pickle


    
#%% User Input
# endpoint              1 OS    2 PSF
# model	                1 Cox   2 RSF
endpoint = 'OS'
estname  = 'RSF'


#%%
if endpoint == 'OS':
    event = 'death'
    time  = 'followup'
    
elif endpoint == 'PFS':
    event = 'death'
    time  = 'NLF_months'
    
features = hcc.optimal_feature_set(endpoint, estname)
est = hcc.optimal_est(endpoint, estname)


# %% Read data
df = pd.read_excel("../dataset.xlsx")

df = df[features+[time, event]]



df = df.dropna(subset = [event, time])
df = df.reset_index(drop=True)

contvars = []
for xx in features:
    if len(df[xx].unique())>10:
        contvars.append(xx)


#%% train and tune final model
X, y = df[features], Surv.from_dataframe(event, time, df)

X = ibr.impute_data(X)
param_grid = hcc.hp_grid(estname)
estGS = GridSearchCV(est, param_grid)
est = estGS.fit(X, y).best_estimator_

#%% Bootstrap
n_boots = 100

c_index_2 = np.array([None]*n_boots)
c_index_4 = np.array([None]*n_boots)
n_patients = df.shape[0]



for ii in range(n_boots):
    
    train = ibr.BootIndices(list(range(n_patients)), ratio = 1) 
    test  = [x for x in list(range(n_patients)) if x not in train]

    
    # Determine test and train sets
    df_train, df_test = df.iloc[train], df.iloc[test]
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    # Imputation
    df_train[features] = ibr.impute_data(df_train[features])
    df_test[features]  = ibr.impute_data(df_test[features])
    
    
    # Scale continuous features
    stdsc = StandardScaler()
    df_train[contvars]  = stdsc.fit_transform(df_train[contvars])
    df_test[contvars]  = stdsc.transform(df_test[contvars])
    
    
    
    # Train the model

    X_train, y_train = df_train[features], Surv.from_dataframe(event, time, df_train)
    X_test,  y_test =  df_test[features],  Surv.from_dataframe(event, time, df_test)
    
    
    est.fit(X_train, y_train)
    
    # Evaluate the model
    risk = est.predict(X_test)
    for NumGroup in  [2, 4]:
        risk_group = pd.qcut(risk, NumGroup, labels=False, duplicates='drop')
        
        lh_ix = (risk_group==0)  |  (risk_group == np.max(risk_group))
        df_lh = df_test[lh_ix]
        X_lh  = X_test[lh_ix]
        y_lh  = y_test[lh_ix]
        
        exec(f"c_index_{NumGroup}[{ii}] = est.score(X_lh, y_lh)")

    
    if ii%5 == 0:
        print(f"boot {ii:>3d} \t c-index_2 : {c_index_2[ii]:0.2f} \t c-index_4 : {c_index_4[ii]:0.2f}")

print(f"c-index 2 = {np.mean(c_index_2):>0.2f} +/- {np.std(c_index_2):>0.2f}")
print(f"c-index 4 = {np.mean(c_index_4):>0.2f} +/- {np.std(c_index_4):>0.2f}")


#%% Save Results
dfout = pd.DataFrame(data = [[None]*3]*n_boots, columns = ['boot','c_index_2', 'c_index_4'])
dfout.boot      = list(range(n_boots))
dfout.c_index_2 = c_index_2
dfout.c_index_4 = c_index_4

path1 = 'Bootstrapping'
try:
    os.mkdir(path1)
except FileExistsError:
    pass

path2 = f'{path1}/{endpoint}'
try:
    os.mkdir(path2)
except FileExistsError:
    pass

dfout.to_excel(f"{path2}/{estname}.xlsx")


#%% Train on the comeplete data set

X, y = df[features], Surv.from_dataframe(event, time, df)

X = ibr.impute_data(X)
param_grid = hcc.hp_grid(estname)
estGS = GridSearchCV(est, param_grid)
est = estGS.fit(X, y).best_estimator_


# save model
path3 = 'FinalModels'
try:
    os.mkdir(path2)
except FileExistsError:
    pass

# save classifier
filename = f"{path3}/{endpoint}_{estname}.pkl"
pickle.dump(est, open(filename, 'wb'))

plt.text(0.1, 0.5, f"{est}")
plt.savefig(f"{path3}/{endpoint}_{estname}_hpars.png", format = 'png', dpi = 600)





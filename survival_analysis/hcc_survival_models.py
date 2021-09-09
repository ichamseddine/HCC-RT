#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 16:53:20 2021

@author: ichamseddine
"""

#%% Libaries
# from lifelines import KaplanMeierFitter, CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis

#%% 
def est_structure(modelname):
    if   modelname == 'Cox': est = CoxPHSurvivalAnalysis() #CoxPHFitter()
    if   modelname == 'RSF': est = RandomSurvivalForest(oob_score=False, random_state=0)
    return est

#%% 
def hp_grid(modelname):
    if   modelname == 'Cox': param_grid = {'alpha':[0.0001, 0.001, 0.01, 0.1, 1] }
    elif modelname == 'RSF': param_grid = {'n_estimators':[10, 50, 100],
                                           'max_depth': [2, 4, 6]}
    return param_grid


#%%
def optimal_feature_set(endpoint, modelname):
    if endpoint == 'OS':
        if modelname == 'Cox'  : fs = ['AFP0', 'ALB0', 'BIL0', 'CP0', 'EQD2_MLD', 'EQD2_V10', 'GTV', 'lesion_size', 'liversize', 'PLT0', 'proton', 'PVT'] #'EQD2_V5', 'EUD'
        elif modelname == 'RSF': fs = ['CP0', 'BIL0', 'AFP0']

        
    if endpoint == 'PFS':
        if modelname == 'Cox'  : fs = ['ALB0']
        elif modelname == 'RSF': fs = ['ALB0']

        
    return fs
        
#%%
def optimal_est(endpoint, modelname):
    if endpoint == 'OS':
        if modelname == 'Cox': est = CoxPHSurvivalAnalysis( alpha = 1) #CoxPHFitter(alpha = 1)
        if modelname == 'RSF': est = RandomSurvivalForest(n_estimators = 50, max_depth = 2)
        
    return est

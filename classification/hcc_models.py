#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 16:53:20 2021

@author: ichamseddine
"""

#%% Libaries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

#%% 
def clf_structure(modelname):
    if   modelname == 'Logit': classifier = LogisticRegression(random_state=0, max_iter = 10000)
    elif modelname == 'SVM':   classifier = SVC(kernel= 'rbf', probability=True)
    elif modelname == 'XGB':   classifier = XGBClassifier(n_estimators = 10, subsample = 0.5)
    elif modelname == 'MLP':   classifier = MLPClassifier(solver = 'adam', max_iter = 15000, hidden_layer_sizes = (10,))
    return classifier

#%% 
def hp_grid(modelname):
    if   modelname == 'Logit': param_grid = {'C'    : [0.1, 1, 2, 5, 10, 20, 50, 100]}
    elif modelname == 'SVM':   param_grid = {'C'    : [0.1, 1, 2, 5, 10, 20, 50, 100], 
                                             'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]}
    elif modelname == 'XGB':   param_grid = {"eta"      : [ 0.1, 0.3, 0.5, 0.7] ,
                                             "max_depth": [ 2, 4, 6 ]}
    elif modelname == 'MLP':   param_grid = { 'alpha'   : [ 10**c for c in range(-6, 0) ]}
    return param_grid


#%%
def optimal_feature_set(endpoint, modelname):
    if endpoint == 'SRVy1':
        if modelname == 'Logit': fs = ['AFP0', 'ALB0', 'BIL0', 'CP0', 'EQD2_MLD', 'EQD2_V10', 'GTV', 'PLT0', 'PVT', 'cirrhosis', 'lesion_size', 'liversize', 'proton'] #'EQD2_V5', 'EUD'
        
        elif modelname == 'SVM': fs = ['CP0', 'ALB0', 'AFP0', 'TD']
        elif modelname == 'XGB': fs = ['age', 'CP0', 'GTV', 'BIL0', 'EQD2_V10']
        elif modelname == 'MLP': fs = ['ALB0', 'TD', 'EQD2_V10'] # 'EQD2_V5'
        
    if endpoint == 'NLFy1':
        if modelname == 'Logit': fs = ['AFP0', 'ALB0', 'ALC0', 'BIL0', 'EQD2_MLD', 'EQD2_V10', 'GTV', 'TD', 'age' ]
        elif modelname == 'SVM': fs = ['lesion_number', 'GTV', 'ALB0', 'TD']
        elif modelname == 'XGB': fs = ['age', 'liversize', 'EQD2_MLD', 'BIL0']
        elif modelname == 'MLP': fs = ['ALB0', 'TD', 'EQD2_MLD']
        
    # if endpoint == 'LRFy1':
    #     if modelname == 'Logit': fs = ['ALB0', 'BIL0', 'GTV', 'newDx', 'EUD', 'EQD2_MLD', 'EQD2_V5', 'EQD2_V10']
    #     elif modelname == 'SVM': fs = ['CP0']
    #     elif modelname == 'XGB': fs = ['ALB0']
    #     elif modelname == 'MLP': fs = ['ALB0']
        
    # if endpoint == 'DMy1':
    #     if modelname == 'Logit': fs = ['AFP0', 'CP0', 'EQD2_MLD', 'EQD2_V5']
    #     elif modelname == 'SVM': fs = ['CP0']
    #     elif modelname == 'XGB': fs = ['ALB0']
    #     elif modelname == 'MLP': fs = ['ALB0']
    
    if endpoint == 'CP2plus':
        if modelname == 'Logit': fs = ['BIL0', 'PLT0'] 
        elif modelname == 'SVM': fs = ['PLT0', 'proton', 'TD']
        elif modelname == 'XGB': fs = ['BIL0', 'PLT0']
        elif modelname == 'MLP': fs = ['GTV', 'BIL0', 'PLT0', 'EQD2_MLD']
        
    if endpoint == 'ALBI1plus':
        if modelname == 'Logit': fs = ['ALB0', 'BIL0', 'PLT0', 'liversize']
        elif modelname == 'SVM': fs = ['ALB0', 'PLT0', 'proton', 'TD', 'EQD2_V5']
        elif modelname == 'XGB': fs = ['BIL0']
        elif modelname == 'MLP': fs = ['liversize', 'ALB0', 'PLT0', 'proton']
    
    if endpoint == 'RIL':
        if modelname == 'Logit': fs = ['AFP0', 'ALC0', 'BIL0', 'EQD2_MLD', 'EQD2_V10','GTV', 'PLT0'] #'EQD2_V5',
        elif modelname == 'SVM': fs = ['ALC0', 'TD']
        elif modelname == 'XGB': fs = ['ALC0', 'TD', 'liversize']
        elif modelname == 'MLP': fs = ['age', 'liversize', 'ALB0', 'ALC0', 'TD']
        
        
    if endpoint == 'LF':
        if modelname == 'Logit': fs = ['sex', 'age', 'cirrhosis', 'cirrhosis_etiology', 'liversize', 'PVT', 'CP0', 'newDx',  'lesion_size', 'lesion_number', 'GTV', 'ALB0', 'BIL0', 'PLT0', 'AFP0', 'ALC0', 'proton', 'Fx', 'TD', 'EUD','EQD2_MLD', 'EQD2_V5', 'EQD2_V10']
        elif modelname == 'SVM': fs = ['age', 'CP0', 'PLT0', 'Fx']
        elif modelname == 'XGB': fs = ['sex', 'age', 'cirrhosis', 'cirrhosis_etiology', 'liversize', 'PVT', 'CP0', 'newDx',  'lesion_size', 'lesion_number', 'GTV','ALB0', 'BIL0', 'PLT0', 'AFP0', 'ALC0', 'proton', 'Fx', 'TD', 'EUD','EQD2_MLD', 'EQD2_V5', 'EQD2_V10']
        elif modelname == 'MLP': fs = ['age', 'ALB0', 'PLT0', 'ALC0', 'EUD', 'EQD2_V10']
        
    if endpoint == 'LFLRF1st':
        if modelname == 'Logit': fs = ['liversize', 'PVT', 'CP0', 'ALC0', 'EQD2_MLD']
        elif modelname == 'SVM': fs = ['age', 'liversize', 'CP0', 'lesion_size', 'ALC0', 'Fx', 'TD', 'EQD2_MLD']
        elif modelname == 'XGB': fs = ['liversize', 'CP0', 'lesion_size', 'EQD2_MLD']
        elif modelname == 'MLP': fs = ['liversize', 'EQD2_MLD']
        
    if endpoint == 'DM1st':
        if modelname == 'Logit': fs = ['CP0', 'PLT0']
        elif modelname == 'SVM': fs = ['ALC0', 'TD']
        elif modelname == 'XGB': fs = ['age', 'PLT0']
        elif modelname == 'MLP': fs = ['CP0', 'TD']
        
    return fs
        
#%%
def optimal_clf(endpoint, modelname):
    if endpoint == 'SRVy1':
        if modelname == 'Logit': clf = LogisticRegression(random_state=0)
        elif modelname == 'SVM': clf = SVC(C = 20, gamma = 0.01, probability=True)
        elif modelname == 'XGB': clf = XGBClassifier(eta = 0.7, max_depth = 2, n_estimators = 10, subsample = 0.5)
        elif modelname == 'MLP': clf = MLPClassifier(alpha = 0.00001, solver = 'adam', max_iter = 15000, hidden_layer_sizes = (10,))
        
    if endpoint == 'NLFy1':
        if modelname == 'Logit': clf = LogisticRegression(C = 1, max_iter = 10000) 
        elif modelname == 'SVM': clf = SVC(C = 0.1, gamma = 1, probability=True)
        elif modelname == 'XGB': clf = XGBClassifier(eta = 0.3, max_depth = 2, n_estimators = 10, subsample = 0.5)
        elif modelname == 'MLP': clf = MLPClassifier(alpha = 1e-5, solver = 'adam', max_iter = 15000, hidden_layer_sizes = (10,))
        
    if endpoint == 'LRFy1':
        if modelname == 'Logit': clf = LogisticRegression(C = 1, max_iter = 10000) 
        elif modelname == 'SVM': clf = SVC(C = 2, gamma = 1, probability=True)
        elif modelname == 'XGB': clf = XGBClassifier(eta = 0.2, max_depth = 2, reg_lambda = 5)
        elif modelname == 'MLP': clf = MLPClassifier(alpha = 0.1, solver = 'adam', max_iter = 15000, hidden_layer_sizes = (10,))
        
    if endpoint == 'DMy1':
        if modelname == 'Logit': clf = LogisticRegression(C = 1, max_iter = 10000) 
        elif modelname == 'SVM': clf = SVC(C = 2, gamma = 1, probability=True)
        elif modelname == 'XGB': clf = XGBClassifier(eta = 0.2, max_depth = 2, reg_lambda = 5)
        elif modelname == 'MLP': clf = MLPClassifier(alpha = 0.1, solver = 'adam', max_iter = 15000, hidden_layer_sizes = (10,))
        
    if endpoint == 'CP2plus':
        if modelname == 'Logit': clf = LogisticRegression(C = 1, max_iter = 10000) 
        elif modelname == 'SVM': clf = SVC(C = 100, gamma = 0.1, probability=True)
        elif modelname == 'XGB': clf = XGBClassifier(eta = 0.7, max_depth = 4, n_estimators = 10, subsample = 0.5)
        elif modelname == 'MLP': clf = MLPClassifier(alpha = 0.00001, solver = 'adam', max_iter = 15000, hidden_layer_sizes = (10,))
        
    if endpoint == 'ALBI1plus':
        if modelname == 'Logit': clf = LogisticRegression(C = 1, max_iter = 10000) 
        elif modelname == 'SVM': clf = SVC(C = 5, gamma = 0.001, probability=True)
        elif modelname == 'XGB': clf = XGBClassifier(eta = 0.2, max_depth = 2, reg_lambda = 5)
        elif modelname == 'MLP': clf = MLPClassifier(alpha = 0.01, solver = 'adam', max_iter = 15000, hidden_layer_sizes = (10,))
    
    if endpoint == 'RIL':
        if modelname == 'Logit': clf = LogisticRegression(C = 1, max_iter = 10000) 
        elif modelname == 'SVM': clf = SVC(C = 100, gamma = 0.1, probability=True)
        elif modelname == 'XGB': clf = XGBClassifier(eta = 0.3, max_depth = 4, n_estimators = 10, subsample = 0.5)
        elif modelname == 'MLP': clf = MLPClassifier(alpha = 0.001, solver = 'adam', max_iter = 15000, hidden_layer_sizes = (10,))
        
    if endpoint == 'LF':
        if modelname == 'Logit': clf = LogisticRegression(C = 1, max_iter = 10000)
        elif modelname == 'SVM': clf = SVC(C = 100, gamma = 0.1, probability=True)
        elif modelname == 'XGB': clf = XGBClassifier(eta = 0.3, max_depth = 4, n_estimators = 10, subsample = 0.5)
        elif modelname == 'MLP': clf = MLPClassifier(alpha = 0.001, solver = 'adam', max_iter = 15000, hidden_layer_sizes = (10,))
    
    if endpoint == 'LFLRF1st':
        if modelname == 'Logit': clf = LogisticRegression(C = 0.1, max_iter = 10000)
        elif modelname == 'SVM': clf = SVC(C = 20, gamma = 0.01, probability=True)
        elif modelname == 'XGB': clf = XGBClassifier(eta = 0.1, max_depth = 2, n_estimators = 10, subsample = 0.5)
        elif modelname == 'MLP': clf = MLPClassifier(alpha = 0.00001, solver = 'adam', max_iter = 15000, hidden_layer_sizes = (10,))
        
    if endpoint == 'DM1st':
        if modelname == 'Logit': clf = LogisticRegression(C = 0.1, max_iter = 10000)
        elif modelname == 'SVM': clf = SVC(C = 20, gamma = 0.01, probability=True)
        elif modelname == 'XGB': clf = XGBClassifier(eta = 0.1, max_depth = 2, n_estimators = 10, subsample = 0.5)
        elif modelname == 'MLP': clf = MLPClassifier(alpha = 0.00001, solver = 'adam', max_iter = 15000, hidden_layer_sizes = (10,))
    return clf

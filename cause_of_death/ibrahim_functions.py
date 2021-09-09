#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 17:29:33 2021

@author: ichamseddine
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import auc
from random import randrange

#%%
def cv_split(df, kfold, endpoint): 
    dftrain, dftest = train_test_split(df, test_size = 1/kfold, 
                                       random_state = 0, stratify = df[endpoint])
    dftrain = dftrain.reset_index(drop=True)
    dftest = dftest.reset_index(drop=True)
    
    return dftrain, dftest

#%%
def BootIndices(indices, ratio):
    # indices: list of indices from which samples are drawn
    # ratio: size of sampel to dataset
    sample = []
    n_sample = round(len(indices) * ratio)
    while len(sample) < n_sample:
        ii = randrange(len(indices))
        sample.append(ii)
    return sample 


#%%
def impute_data(X):
    features = X.columns
    # build Bayesian ridge regression imputer
    imp = IterativeImputer(random_state=0, max_iter=100)
    imp.fit(X) 
    # impute missing values
    X = imp.transform(X)
    X=pd.DataFrame(X,columns=features)
    return X

#%%
def plot_roc(tprs, mean_fpr, aucs, endpoint, clfname, pathsave):

    fig, ax = plt.subplots(figsize=(4,4))
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k')
    
    mean_tpr = np.nanmean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.nanmean(aucs)
    std_auc = np.nanstd(aucs)
    auc_lo = np.max([0., mean_auc - std_auc]) #np.percentile(aucs, 2.5, axis = 0)
    auc_hi = np.min([1., mean_auc + std_auc]) #np.percentile(aucs, 97.5, axis = 0)
    ax.plot(mean_fpr, mean_tpr, color='k',lw=2, alpha=.8)
    
    std_tpr = np.nanstd(tprs, axis=0)
    tprs_upper = mean_tpr + std_tpr #np.percentile(tprs, 2.5, axis = 0)
    tprs_lower = mean_tpr - std_tpr #np.percentile(tprs, 97.5, axis = 0)
    tprs_upper = np.clip(tprs_upper, 0., 1.)
    tprs_lower = np.clip(tprs_lower, 0., 1.)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)
    
    #ttl = f"{endpoint} {clfname} AUC = {mean_auc:0.2f} CI[{auc_lo:0.2f}, {auc_hi:0.2f}]"
    ttl = f"{endpoint} {clfname} AUC = {mean_auc:0.2f}  SD[{auc_lo:0.2f}, {auc_hi:0.2f}]"
    
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    ax.set_title(ttl, fontsize=10)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    
    
    plt.savefig(f'{pathsave}/{clfname}_roc.png', dpi=600, format = 'png', bbox_inches='tight')
    
#%%
def plot_prec_recall(precs, mean_recall, aps, endpoint, clfname, pathsave, y_true):
    plt.figure(figsize = (4,4)) 
    mean_prec = np.mean(precs, axis=0)
    std_prec = np.std(precs, axis=0)
    prec_up = mean_prec + std_prec
    prec_lo = mean_prec - std_prec
    prec_up = np.clip(prec_up, 0., 1.)
    prec_lo = np.clip(prec_lo, 0., 1.)
    # prec_lo = np.max([0., np.percentile(precs, 2.5, axis = 0))
    ap_mean = np.mean(aps)
    ap_std = np.std(aps)
    ap_up = ap_mean + ap_std
    ap_lo = ap_mean - ap_std
    ap_up = np.min([1., ap_up])
    ap_lo = np.max([0., ap_lo])
    # ap_up = np.min([1., np.percentile(aps, 97.5, axis = 0)])
    # ap_lo = np.max([0., np.percentile(aps, 2.5, axis = 0)])
    plt.fill_between(mean_recall, prec_lo, prec_up, color='grey', alpha=0.3)
    plt.plot(mean_recall, mean_prec, 'k')

    no_skill = len(y_true[y_true==1]) / len(y_true)
    plt.plot([0, 1], [no_skill, no_skill], 'k--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
    plt.title(f"{endpoint} {clfname} AP {np.mean(aps):0.2f} [{ap_lo:0.2f}, {ap_up:0.2f}]")
    plt.savefig(f"{pathsave}/{clfname}_prec_recall.png", format = 'png', dpi = 600, bb_inches = 'tight')


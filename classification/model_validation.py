#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 19:27:17 2021

@author: ichamseddine
"""

#%% Libraries

# Common
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Self-defined modules
import hcc_models as hcc
import ibrahim_functions as ibr

#
from sklearn.metrics import (plot_roc_curve, roc_auc_score, roc_curve, accuracy_score,\
                             average_precision_score, precision_recall_curve, \
                             r2_score, brier_score_loss)
    
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

#
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#
import sys
import warnings
import pickle
import os
warnings.simplefilter(action='ignore', category=FutureWarning)

#%% INPUT
endpoint = 'CP2plus' # SRVy1, NLFy1, CP2plus, ALBI1plus, RIL
clfname = 'MLP'   # Logit, SVM, XGB, MLP 
agg = 1 # an indicator of whether model aggregation is used. Set to 1 for the imbalanced toxicity outcomes. 0 otherwise 
nmodels = 10

#%% User Input
# endpoint              1 SRV1    2 RILD    3 ALC3    4 LRFDM1
# model	                0 Logit   1 SVM     2 XGB     3 MLP     

features = hcc.optimal_feature_set(endpoint, clfname)
clf = hcc.optimal_clf(endpoint, clfname)

try:
    os.mkdir('ExternalValidation')
except FileExistsError:
    pass

pathsave = f'ExternalValidation/{endpoint}_{clfname}_tuned'
try:
    os.mkdir(pathsave)
except FileExistsError:
    pass


#%% Read data
# internal dataset
dfint = pd.read_excel("../dataset.xlsx")
dfint = dfint[features+[endpoint]]
dfint = dfint.dropna(subset = [endpoint])
dfint = dfint.reset_index(drop=True)      
        
Xint = dfint[features]
yint = dfint[endpoint].astype(bool)

# external dataset
dfext = pd.read_excel("../mda_dataset.xlsx")
dfext = dfext[features+[endpoint]]
dfext = dfext.dropna(subset = [endpoint])
dfext = dfext.reset_index(drop=True)      
             
Xext = dfext[features]
yext = dfext[endpoint].astype(bool)

# cont vars
contvars = []
for xx in features:
    if len(dfint[xx].unique())>10:
        contvars.append(xx)
        
 
       
#%% load models

if agg == 0: 
    filename = f'FinalModels/{endpoint}_{clfname}_tuned.pkl'
    clf      = pickle.load(open(filename, 'rb'))
    
elif agg == 1: 
    for ii in range(1,nmodels+1):
        filename = f"FinalModelsAgg/{endpoint}_{clfname}_models/submodel_{ii}.pkl"
        exec(f"clf_{ii} = pickle.load(open(filename, 'rb'))")
    

#%% Imputation
imp = IterativeImputer(random_state=0, max_iter=100)
Xint = imp.fit_transform(Xint)
Xext = imp.fit_transform(Xext) 

Xint = pd.DataFrame(Xint,columns=features)
Xext = pd.DataFrame(Xext,columns=features)

#%%bScale continuous features
stdsc = StandardScaler()
stdsc.fit(Xint[contvars])
Xint[contvars]  = stdsc.transform(Xint[contvars])
Xext[contvars]  = stdsc.fit_transform(Xext[contvars])

#%% Prediction 
# clf.fit(Xint, yint)
if agg == 0:
    y_pred = clf.predict_proba(Xext)[:, 1]
    
elif agg == 1:
    y_pred = 0
    for m in range(1,nmodels+1):
        exec(f's_{m} = clf_{m}.predict_proba(Xext)[:, 1]')
        exec('y_pred += s_%i'%m) 
        
    y_pred = y_pred/(nmodels)

dfout = dfext
dfout[f'{endpoint}_pred'] = y_pred
dfout.to_excel(f"{pathsave}/predicted_probabilities.xlsx")

#%% High and low risk groups
y_true = yext.astype('int')


risk_group = pd.qcut(y_pred, 10, labels=False, duplicates='drop')

ix = risk_group == 9
ac_hi_10 = accuracy_score(y_true[ix].values, np.ones(sum(ix)))

ix = (risk_group == 9) | (risk_group == 8)
ac_hi_20 = accuracy_score(y_true[ix].values, np.ones(sum(ix)))


ix = risk_group == 0
ac_lo_10 = accuracy_score(y_true[ix].values, np.zeros(sum(ix)))

ix = (risk_group == 0) | (risk_group == 1)
ac_lo_20 = accuracy_score(y_true[ix].values, np.zeros(sum(ix)))

#%%# ROC



AUC = roc_auc_score(y_true, y_pred)
fpr, tpr, thr = roc_curve(y_true, y_pred)
plt.figure(figsize=(4,5))
plt.plot(fpr, tpr, lw = 2, color = 'k')
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title(f'{endpoint}  {clfname}  AUC = {AUC:0.2f}')
ttl = f'{endpoint}  {clfname}  AUC = {AUC:0.2f} \n ac_hi_10 = {ac_hi_10:0.2f} ac_hi_20 = {ac_hi_20:0.2f} \n ac_lo_10 = {ac_lo_10:0.2f} ac_lo_20 = {ac_lo_20:0.2f}'
plt.title(ttl, fontsize=12)
plt.savefig(f"{pathsave}/roc.png", format = 'png', dpi = 600)

#%% Precision Recall

AP = average_precision_score(y_true, y_pred)
prec, recall, _ = precision_recall_curve(y_true, y_pred)

prec   = prec[::-1]
recall = recall[::-1]

plt.figure(figsize=(4,4))
plt.plot(recall, prec, 'k', lw = 2)
no_skill = len(y_true[y_true==1]) / len(y_true)
plt.plot([0, 1], [no_skill, no_skill], 'k--')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'{endpoint}  {clfname}  AP = {AP:0.2f}')
plt.savefig(f"{pathsave}/precision_recall.png", format = 'png', dpi = 600)


    
#%% Calibration plot

def efron_rsquare(y, y_pred):
    n = float(len(y))
    t1 = np.sum(np.power(y - y_pred, 2.0))
    t2 = np.sum(np.power((y - (np.sum(y) / n)), 2.0))
    return 1.0 - (t1 / t2)


n_bins = 4

bins = pd.qcut(y_pred, n_bins, labels=False, duplicates='drop')

mean_pred_surv, true_surv = np.ones(n_bins), np.ones(n_bins)
lolims, uplims = np.ones(n_bins), np.ones(n_bins)
yerr = np.ones(n_bins)

for i in range(n_bins):
    mean_pred_surv[i] = np.mean(y_pred[bins == i])
    true_surv[i]       = y_true[bins==i].sum()/y_true[bins==i].shape[0]

    
plt.figure(figsize=[4,4])
plt.scatter( mean_pred_surv, true_surv, marker='o', color='k', lw=1, label='Count R2    %0.3f'%r2_score(y_true.astype(int), y_pred))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.plot([0, 1], [0, 1], linestyle = '--', color = 'k')
plt.xlabel(f'Predicted {endpoint}')
plt.ylabel(f'Observed {endpoint}')
plt.title(f'{endpoint} {clfname} Count $R^2$ {(efron_rsquare(y_true.astype(int), y_pred)):0.2f}, Brier score {brier_score_loss(y_true.astype(int), y_pred):0.2f}') 
plt.savefig(f'{pathsave}/calibration_plot.png', format='png', dpi = 600, bb_inches='tight')


#%% Net benefit analysis
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')

try:
    os.mkdir('NetBenefitAnalysisAgg')
except FileExistsError:
    pass

pts = np.linspace(0.05,.95,100)
net_benefit = -np.ones(len(pts))
true_label = y_true.values.astype(int)
N = len(true_label)
for ii, pt in enumerate(pts):
    pred_label = to_labels(y_pred, pt)
    TP, FP = 0, 0
    for pl, tl in zip(pred_label, true_label):
        if   pl==1 and tl == 1: TP += 1
        elif pl==1 and tl==0: FP += 1
        
    net_benefit[ii] = TP/N - FP/N * pt/(1-pt)

plt.figure(figsize=[4,4])
plt.plot(pts, net_benefit, color = 'k')
plt.plot([0,1], [0,0], color = 'k', linestyle = '--')
plt.xlabel('Threshold probability (%)')
plt.ylabel('Net benefit') 
plt.title(f'{endpoint} {clfname}')
plt.savefig(f'{pathsave}/netbenefit.png', format='png', dpi = 600, bb_inches='tight')







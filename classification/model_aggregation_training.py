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
from sklearn.metrics import (plot_roc_curve, roc_auc_score, roc_curve, \
                             average_precision_score, precision_recall_curve, \
                             r2_score, brier_score_loss)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV

#
import sys
import warnings
import pickle
import os
warnings.simplefilter(action='ignore', category=FutureWarning)

#%% INPUT
ENDPOINT = 'RIL'  # SRVy1, NLFy1, CP2plus, ALBI1plus, RIL, LFLRF1st (refers to LFLRF failure before DM), DMN1st (DM before LFLRF)
CLFNAME = 'Logit'   # Logit, SVM, XGB, MLP 
NMODEL = 10
BagRatio = 0.5

#%% User Input
# ENDPOINT              1 SRV1    2 RILD    3 ALC3    4 LRFDM1
# model	                0 Logit   1 SVM     2 XGB     3 MLP     

features = hcc.optimal_feature_set(ENDPOINT, CLFNAME)
clf = hcc.optimal_clf(ENDPOINT, CLFNAME)

try:
    os.mkdir('AggBootstrapping')
except FileExistsError:
    pass

pathsave = f'AggBootstrapping/{ENDPOINT}'
try:
    os.mkdir(pathsave)
except FileExistsError:
    pass


#%% Read data
df = pd.read_excel("../dataset.xlsx")
df = df[features+[ENDPOINT]]

df = df.dropna(subset = [ENDPOINT])
df = df.reset_index(drop=True)      
        
contvars = []
for xx in features:
    if len(df[xx].unique())>10:
        contvars.append(xx)


X = df[features]
y = df[ENDPOINT].astype(bool)

#%% HPO and training

X = ibr.impute_data(X)
stdsc = StandardScaler()
X[contvars] = stdsc.fit_transform(X[contvars])

# param_grid = hcc.hp_grid(CLFNAME)

# clf = GridSearchCV(hcc.clf_structure(CLFNAME), param_grid, refit=True)

# clf = clf.fit(X, y.values.ravel()).best_estimator_

#%% Crossvalidation

n_cv = 100
aucs, tprs, mean_fpr            = [], [], np.linspace(0, 1, 100)
aps, precs, mean_recall         = [], [], np.linspace(0, 1, 100)


cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=0)

dfout = pd.DataFrame(data = [[None]*(NMODEL*2+5)]*n_cv, \
                     columns = ['TrainIndex', 'ValIndex'] + \
                               ['Bag_%i'%x for x in range(1,NMODEL+1)] + \
                               ['s_%i'%x for x in range(1,NMODEL+1)] + \
                               ['s_avg',\
                                'AUC', 'AP'])
n_patients = df.shape[0]
print('-------------------------------------------')
print(f"{ENDPOINT} \t {CLFNAME}\n")
k = -1
while k < n_cv-1:
    k+=1
    # Train and test indices
    train = ibr.BootIndices(list(range(n_patients)), ratio = 1) 
    test  = [x for x in list(range(n_patients)) if x not in train]
    
    # --- Stop infeasible set
    onesInTrain = y[train].sum()
    onesInTest = y[test].sum()
    if (onesInTrain==0) or (onesInTest==0):
        print("Infeasible split. One or more dataset has one label.")
        k -= 1
        continue
    
    if ENDPOINT == 'NLFy1':
        zerosInTrain = (y[train]==0).sum()
        zerosInTest =  (y[test]==0).sum()
        if (zerosInTrain==0) or (zerosInTest==0):
            print("Infeasible split. One or more dataset has one label.")
            k -= 1
            continue
        
    
    
    
    
    dfout.TrainIndex[k] = train
    dfout.ValIndex[k]   = test
    
    X_train, y_train = X.iloc[train], y.iloc[train]
    X_test, y_test = X.iloc[test], y.iloc[test]
    X_test.reset_index(drop=True)
    y_test.reset_index(drop=True)
    
    # Fill the bags
    b = 1
    bag_len = int(BagRatio * len(train))
    while b<=NMODEL:
        feas = 0
        exec('Bag_%i = ibr.BootIndices(train, BagRatio)'%b)
        exec('y_%i   = y[Bag_%i]'%(b,b))
        # exec('if y_%i.sum() > 3: \n feas = 1'%b)   
        # --- Only include balanced sets
        
        # if ENDPOINT == 'SRVy1':
        #     exec("zerosInYi = (y_%i==0).sum()"%b)
        #     exec("if not 0.4 <= zerosInYi/(zerosInYi+zerosInTest) <= 0.6: feas = 1")
        # else:
        #     exec("onesInYi = y_%i.sum()"%b)
        #     exec("if not 0.4 <= onesInYi/(onesInYi+onesInTest) <= 0.6: feas = 1")
        
        if ENDPOINT == 'SRVy1':
            exec("zerosInYi = (y_%i==0).sum()"%b)
            exec("if 0.4 <= zerosInYi/bag_len <= 0.6: feas = 1")
        else:
            exec("onesInYi = y_%i.sum()"%b)
            exec("if 0.4 <= onesInYi/bag_len <= 0.6: feas = 1")
        
        # continue
        if feas == 1:
            exec('dfout.Bag_%i[k] = Bag_%i'%(b,b))
            b += 1
    
    # Imputation
    X_train = ibr.impute_data(X_train)
    X_test = ibr.impute_data(X_test) 
    
    # Scale continuous features
    stdsc = StandardScaler()
    X_train[contvars]  = stdsc.fit_transform(X_train[contvars].copy())
    X_test[contvars]  = stdsc.transform(X_test[contvars].copy())
    
    for b in range(1,NMODEL+1):
        
        # Determine test and train sets
        exec('X_%i = X_train.iloc[Bag_%i]'%(b,b))
        exec('y_%i = y_train.iloc[Bag_%i].astype("int")'%(b,b))
        exec('X_%i = X_%i.reset_index(drop=True)'%(b,b))
        exec('y_%i = y_%i.reset_index(drop=True)'%(b,b))
        

    # Train, predict, and aggregate
    s_avg = 0
    a = .5 # 0.8
    for m in range(1,NMODEL+1):
        exec('w = a*y_%i + (1-a)'%m)
        exec('clf_%i = hcc.optimal_clf(ENDPOINT, CLFNAME)'%m)
        # exec('clf_%i.fit(X_%i, y_%i, sample_weight = w )'%(m,m,m))
        exec('clf_%i.fit(X_%i, y_%i)'%(m,m,m))
        exec('s_%i = clf_%i.predict_proba(X_test)[:, 1]'%(m,m))
        exec('dfout.s_%i[k] = s_%i'%(m,m))
        exec('s_avg += s_%i'%m) 
        
    s_avg = s_avg/(NMODEL)
    dfout.s_avg[k] = s_avg
    
    # ROC
    y_true = y_test.astype('int')
    y_pred = s_avg
    
    auc_k = roc_auc_score(y_true, y_pred)
    dfout.AUC[k] = auc_k
    aucs.append(auc_k)
    fpr, tpr, thr = roc_curve(y_true, y_pred)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    
    
    # Precision-Recall
    THR = 0.5
    ap = average_precision_score(y_true, y_pred)
    dfout.AP[k] = ap
    aps.append(ap)
    prec, recall, _ = precision_recall_curve(y_true, y_pred)
    # reverse order since recall is in decending order
    prec   = prec[::-1]
    recall = recall[::-1]
    # interpolate
    interp_prec = np.interp(mean_recall, recall, prec)
    precs.append(interp_prec)

    print(f"boot {k:>3d} \t AUC : {aucs[k]:0.3f} \t AP : {ap:0.3f}")
     

    dfout.to_excel(f"{pathsave}/{CLFNAME}.xlsx")

dfout.to_excel(f"{pathsave}/{CLFNAME}.xlsx")


#%% Plot Roc Curve     

ibr.plot_roc(tprs, mean_fpr, aucs, ENDPOINT, CLFNAME, pathsave)


#%% Plot Precision-Recall
ibr.plot_prec_recall(precs, mean_recall, aps, ENDPOINT, CLFNAME, pathsave, y_test)



#%% Fit the model to the compete dataset

path2 = 'FinalModelsAgg'
try:
    os.mkdir(path2)
except FileExistsError:
    pass

path3 = f'FinalModelsAgg/{ENDPOINT}_{CLFNAME}_models'

try:
    os.mkdir(path3)
except FileExistsError:
    pass

X = ibr.impute_data(X)
stdsc = StandardScaler()
X[contvars] = stdsc.fit_transform(X[contvars])

# Fill the bags
ratio = 0.5
b = 1
lenbag = int(y.shape[0]*ratio)
while b<=NMODEL:
    feas = 0
    exec('Bag_%i = ibr.BootIndices(list(range(n_patients)), ratio)'%b)
    exec('y_%i = y.iloc[Bag_%i]'%(b,b))
    exec('X_%i = X.iloc[Bag_%i]'%(b,b))
    exec("onesInYi = y_%i.sum()"%b)
    exec("if 0.4 <= onesInYi/(lenbag) <= 0.6: feas = 1")
    if feas == 1:
        b += 1
# --- now we have indices of the validation set and the bags 1-10 for training
    
#%% HPO and training
        
# Train 10 models and dump them
param_grid = hcc.hp_grid(CLFNAME)
for b in range(1,NMODEL+1):
    exec('print(f"\t training and dumping model {%i}")'%(b))
    exec(f'clf_{b} = GridSearchCV(hcc.clf_structure(CLFNAME), param_grid, refit=True)')
    exec(f'clf_{b} = clf_{b}.fit(X_{b}, y_{b}).best_estimator_')
    exec('pickle.dump(clf_%i, open(f"{path3}/submodel_%i.pkl", "wb"))'%(b,b))
    
    plt.figure()
    exec(f'plt.text(0.1,0.5,clf_{b})')
    exec(f'plt.savefig("{path3}/hpars_submod_{b}.png", format = "png", dpi = 600)')
    
#%% Calibration plot

def efron_rsquare(y, y_pred):
    n = float(len(y))
    t1 = np.sum(np.power(y - y_pred, 2.0))
    t2 = np.sum(np.power((y - (np.sum(y) / n)), 2.0))
    return 1.0 - (t1 / t2)



try:
    os.mkdir('CalibrationPlotsAgg')
except FileExistsError:
    pass

n_bins = 4

pred = 0

for b in range(1,NMODEL+1):
    exec(f'pred_{b} = clf_{b}.predict_proba(X)[:,1]')
    exec(f'pred += pred_{b}')
    
pred /= NMODEL
strategy = 'quantile'

if strategy == 'quantile':
    bins = pd.qcut(pred, n_bins, labels=False, duplicates='drop')
elif strategy == 'uniform':
    bins = np.array([None]*df.shape[0])
    bins[pred<.25] = 0
    bins[(pred>=.25) & (pred<.50)] = 1
    bins[(pred>=.50) & (pred<.75)] = 2
    bins[pred>=.75] = 3



mean_pred_surv, true_surv = np.ones(n_bins), np.ones(n_bins)
lolims, uplims = np.ones(n_bins), np.ones(n_bins)
yerr = np.ones(n_bins)

for i in range(n_bins):
    mean_pred_surv[i] = np.mean(pred[bins == i])
    # lolims[i]       = np.percentile(pred[bins == i], 2.5)
    # uplims[i]       = np.percentile(pred[bins == i], 97.5)
    true_surv[i]       = y[bins==i].sum()/y[bins==i].shape[0]
    # yerr[i]         = np.array([true_surv[i] - lolims[i], uplims[i] - true_surv[i]])
    

    
plt.figure(figsize=[4,4])
# yerr = np.transpose(yerr)
# plt.errorbar( mean_pred_surv, true_surv, yerr=yerr, marker='o', color='k', lw=1,
#              label='Count R2    %0.3f'%r2_score(y.astype(int), pred))
plt.scatter( mean_pred_surv, true_surv, marker='o', color='k', lw=1,
              label='Count R2    %0.3f'%r2_score(y.astype(int), pred))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.plot([0, 1], [0, 1], linestyle = '--', color = 'k')
plt.xlabel(f'Predicted {ENDPOINT}')
plt.ylabel(f'Observed {ENDPOINT}')
plt.title(f'{ENDPOINT} {CLFNAME} Count $R^2$ {(efron_rsquare(y.astype(int), pred)):0.2f}, Brier score {brier_score_loss(y.astype(int), pred):0.2f}') 
plt.savefig(f'CalibrationPlotsAgg/{ENDPOINT}_{CLFNAME}.png', format='png', dpi = 600, bb_inches='tight')


#%% Net benefit analysis
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')

try:
    os.mkdir('NetBenefitAnalysisAgg')
except FileExistsError:
    pass

pts = np.linspace(0.05,.95,100)
net_benefit = -np.ones(len(pts))
true_label = y.values.astype(int)
N = len(true_label)
for ii, pt in enumerate(pts):
    pred_label = to_labels(pred, pt)
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
plt.title(f'{ENDPOINT} {CLFNAME}')
plt.savefig(f'NetBenefitAnalysisAgg/{ENDPOINT}_{CLFNAME}.png', format='png', dpi = 600, bb_inches='tight')


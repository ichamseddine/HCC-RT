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
from sklearn.metrics import plot_roc_curve, r2_score, brier_score_loss
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV

#
from scipy.special import expit

#
import sys
import warnings
import pickle
import os
warnings.simplefilter(action='ignore', category=FutureWarning)

#%% INPUT
ENDPOINT = 'RIL'  # SRVy1, NLFy1, CP2plus, ALBI1plus, RIL
CLFNAME = 'Logit'   # Logit, SVM, XGB, MLP 

#%% User Input
# ENDPOINT              1 SRV1    2 RILD    3 ALC3    4 LRFDM1
# model	                0 Logit   1 SVM     2 XGB     3 MLP     

features = hcc.optimal_feature_set(ENDPOINT, CLFNAME)
clf = hcc.optimal_clf(ENDPOINT, CLFNAME)

try:
    os.mkdir('Bootstrapping')
except FileExistsError:
    pass

pathsave = f'Bootstrapping/{ENDPOINT}_tuned'
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

NBOOTS = 100
tprs = []
mean_fpr = np.linspace(0, 1, 100)
aucs = np.array([None]*NBOOTS)


cv = StratifiedKFold(n_splits=NBOOTS, shuffle=True, random_state=0)

dfout = pd.DataFrame(data = [[None]*2]*NBOOTS, columns = ['boot','AUC'])
n_patients = df.shape[0]

k = -1

while k < NBOOTS-1:
    train = ibr.BootIndices(list(range(n_patients)), ratio = 1) 
    test  = [x for x in list(range(n_patients)) if x not in train]

    X_train, y_train = X.iloc[train], y.iloc[train]
    X_test, y_test = X.iloc[test], y.iloc[test]
    
    X_train.reset_index(drop=True)
    y_train.reset_index(drop=True)
    X_test.reset_index(drop=True)
    y_test.reset_index(drop=True)
    
    # Check Boot feasibility
    if (y_train==1).sum() == 0 or (y_test==1).sum() == 0:
        print("Infeasible sample")
        continue
    
    k += 1  
    # print(f"balance {y_test.sum()/y_test.shape[0]}")
    # Imputation
    X_train = ibr.impute_data(X_train)
    X_test = ibr.impute_data(X_test) 
    
    # Scale continuous features
    stdsc = StandardScaler()
    # stdsc = MinMaxScaler()
    X_train[contvars]  = stdsc.fit_transform(X_train[contvars].copy())
    X_test[contvars]  = stdsc.transform(X_test[contvars].copy())
    
    # Train the model
    # a = 0.5 # expected portion of the major negative group
    # w = a*y_train.values+ (1-a)
    # if (CLFNAME == 'MLP') | (ENDPOINT == 'SRVy1'):
    #     clf.fit(X_train, y_train)
    # else:
    #     clf.fit(X_train, y_train, sample_weight = w)
    
    clf.fit(X_train, y_train)
    # Evaluate the model
    rocFig = plot_roc_curve(clf, X_test, y_test)
    auc_k = rocFig.roc_auc
    interp_tpr = np.interp(mean_fpr, rocFig.fpr, rocFig.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs[k] = auc_k
    plt.close()
    
    # # Plot box train and test sets for troubleshooting purposes
    # var = 'BIL0'
    # path2 = f'StabilityAnalysis/indvBox_{ENDPOINT}_{CLFNAME}_{var}'
    # try:
    #     os.mkdir(path2)
    # except FileExistsError:
    #     pass
    
    # plt.figure(figsize=(4,4))
    # dff = pd.read_excel("../dataset.xlsx")
    # dff = dff.dropna(subset=[ENDPOINT])
    # dff['subset'] = np.array(['train']*dff.shape[0])
    # for ii in test:
    #     dff.subset.iloc[ii] = 'test'
    # sns.swarmplot(data = dff, x = ENDPOINT, y = var, hue = 'subset', palette = 'pastel')
    # plt.title(f"AUC = {auc_k:0.2f}")
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    # plt.savefig(f"{path2}/auc_{auc_k:0.2f}.png", format = 'png', dpi = 600)
    # plt.close()


    # # Plot scatter train and test sets for troubleshooting purposes
    # path3 = f'StabilityAnalysis/indvScatt_{ENDPOINT}_{CLFNAME}_{var}'
    # try:
    #     os.mkdir(path3)
    # except FileExistsError:
    #     pass
    
    # plt.figure(figsize=(4,4))
    # # plt.plot(X_train, clf.coef_*X_train+clf.intercept_, color='k', lw=1)
    # # plt.plot(X_test, clf.coef_*X_test+clf.intercept_, color='b', lw=1)
    # loss_train = expit(clf.coef_*X_train+clf.intercept_)
    # loss_test = expit(clf.coef_*X_test+clf.intercept_)
    # plt.plot(X_train, loss_train, color='k', lw=1)
    # plt.plot(X_test, loss_test, color='b', lw=1)
    # plt.scatter(X_train, y_train, color='k', label='train')
    # plt.scatter(X_test, y_test, color='b', label='test')
    # plt.legend()
    # plt.xlabel(var)
    # plt.ylabel(ENDPOINT)
    # plt.title(f"AUC = {auc_k:0.2f}")
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    # plt.savefig(f"{path3}/auc_{auc_k:0.2f}.png", format = 'png', dpi = 600)
    # plt.close()                

                

    print(f"boot {k:>3d} \t AUC : {aucs[k]:0.3f} \t pos : {y_test.sum()}")
        
    # Save Results
    if k%5==0:

        dfout.fold = k
        dfout.AUC  = aucs[k]
        dfout.to_excel(f"{pathsave}/{CLFNAME}.xlsx")
        
print()
print(f"{k+1:>3d} samples are completed \n")
print(f"Median AUC \t {np.median(aucs):>0.2f}")
print(f"Q1         \t {np.percentile(aucs, 25, interpolation = 'midpoint'):>0.2f}")
print(f"Q3         \t {np.percentile(aucs, 75, interpolation = 'midpoint'):>0.2f}")
print()
print()


#%% Plot ROC
ibr.plot_roc(tprs, mean_fpr, aucs, ENDPOINT, CLFNAME, pathsave)


#%% Fit the model to the compete dataset

path2 = 'FinalModels'
try:
    os.mkdir(path2)
except FileExistsError:
    pass

X = ibr.impute_data(X)
stdsc = StandardScaler()
X[contvars] = stdsc.fit_transform(X[contvars])

param_grid = hcc.hp_grid(CLFNAME)

clf = GridSearchCV(hcc.clf_structure(CLFNAME), param_grid, refit=True)

clf = clf.fit(X, y.values.ravel()).best_estimator_
print("Fitting the model to the complete dataset")

# save classifier
filename = f"{path2}/{ENDPOINT}_{CLFNAME}_tuned.pkl"
pickle.dump(clf, open(filename, 'wb'))

plt.figure()
plt.text(0.1,0.5,f"{clf}")
plt.savefig(f"{path2}/{ENDPOINT}_{CLFNAME}_hpars.png", format = 'png', dpi = 600)
#%% Calibration plot

def efron_rsquare(y, y_pred):
    n = float(len(y))
    t1 = np.sum(np.power(y - y_pred, 2.0))
    t2 = np.sum(np.power((y - (np.sum(y) / n)), 2.0))
    return 1.0 - (t1 / t2)



try:
    os.mkdir('CalibrationPlots')
except FileExistsError:
    pass

n_bins = 4
pred = clf.predict_proba(X)[:,1]
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
plt.savefig(f'CalibrationPlots/{ENDPOINT}_{CLFNAME}.png', format='png', dpi = 600, bb_inches='tight')

#%% Net benefit analysis
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')

try:
    os.mkdir('NetBenefitAnalysis')
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
plt.savefig(f'NetBenefitAnalysis/{ENDPOINT}_{CLFNAME}.png', format='png', dpi = 600, bb_inches='tight')

# #%% Analyze data: whey AUC is sometimes low although p value is small from the univariate analysis
# try:
#     os.mkdir('StabilityAnalysis')
# except FileExistsError:
#     pass

# var = 'BIL0'
# plt.figure(figsize=(4,4))
# dff = pd.read_excel("../dataset.xlsx")
# dff = dff.dropna(subset=[ENDPOINT])
# # sns.violinplot(data = df, x = ENDPOINT, y = var, color='lightgrey')
# sns.swarmplot(data = dff, x = ENDPOINT, y = var, color='k')
# sns.boxplot(data = dff, x = ENDPOINT, y = var, color='w')
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.savefig(f"StabilityAnalysis/box_{ENDPOINT}_{var}.png", format = 'png', dpi = 600)

#%% Plot AUCs distribution
# Can we assume normal distribution?

# hist, bins = np.histogram(aucs, bins=100, normed=True)
# bin_centers = (bins[1:]+bins[:-1])*0.5
# plt.plot(bin_centers, hist)
# plt.xlabel('AUC')
# plt.savefig(f'StabilityAnalysis/Dist_10000Boots_{ENDPOINT}_{CLFNAME}.png', format='png', dpi = 600, bb_inches='tight')







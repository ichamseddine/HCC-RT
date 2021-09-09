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

# Self-defined modules
import hcc_models as hcc
import ibrahim_functions as ibr


# 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

#
import warnings
import os
import dataframe_image as dfi
import seaborn as sns
warnings.simplefilter(action='ignore', category=FutureWarning)


#%% INPUT
ENDPOINT = 'DM1st'  # SRVy1, NLFy1, CP2plus, ALBI1plus, RIL
CLFNAME = 'Logit'   # Logit, SVM, XGB, MLP 

#%% Read data
df = pd.read_excel("../dataset.xlsx")

features = ['sex', 'age', 
            'cirrhosis', 'cirrhosis_etiology', 'liversize', 'PVT', 'CP0',
            'newDx',  'lesion_size', 'lesion_number', 'GTV', 
            'ALB0', 'BIL0', 'PLT0', 'AFP0', 'ALC0', 
            'proton', 'Fx', 'TD', 'EUD','EQD2_MLD', 
            'EQD2_V5', 'EQD2_V10']


df = df[features + [ENDPOINT]]

#%%
# Cirrhosis Etiology
#   1 = Alcohol
#   2 = Hepatitis B
#   3 = Hepatitis C
#   4 = NASH (Non-alcoholic steatohepatitis)
#   5 = Cryptogenic
#   6 = Autoimmune hepatitis
#   7 = alpha-1 antitrypsin deficiency
#   8 = NAFLD

etio = ['alc', 'hepB', 'hepC', 'nash', 'cryp', 'autoimm', 'alpha', 'nafld']
cirrhosis_etiology = ['cir_%s'%ee for ee in etio]

for ee in cirrhosis_etiology:
    df[ee] = np.zeros(df.shape[0])

for ii in range(df.shape[0]):
    if df['cirrhosis_etiology'][ii] == 1:
        df.cir_alc[ii] = 1
    elif df['cirrhosis_etiology'][ii] == 2:
        df.cir_hepB[ii] = 1
    elif df['cirrhosis_etiology'][ii] == 3:
        df.cir_hepC[ii] = 1
    elif df['cirrhosis_etiology'][ii] == 4:
        df.cir_nash[ii] = 1
    elif df['cirrhosis_etiology'][ii] == 5:
        df.cir_cryp[ii] = 1
    elif df['cirrhosis_etiology'][ii] == 6:
        df.cir_autoimm[ii] = 1
    elif df['cirrhosis_etiology'][ii] == 7:
        df.cir_alpha[ii] = 1
    elif df['cirrhosis_etiology'][ii] == 8:
        df.cir_nafld[ii] = 1
    else:
        df.cir_alc[ii],   df.cir_hepB[ii], df.cir_hepC[ii]    = None, None, None
        df.cir_nash[ii],  df.cir_cryp[ii], df.cir_autoimm[ii] = None, None, None
        df.cir_alpha[ii], df.cir_nafld[ii]                    = None, None


#%%    
df = df.drop(columns = 'cirrhosis_etiology')     
features = list(df.columns.values)
features.remove(ENDPOINT)

df = df.dropna(subset = [ENDPOINT])
df = df.reset_index(drop = True)        
        
contfeatures = []
for xx in features:
    if len(df[xx].unique())>10:
        contfeatures.append(xx)
    
#%% nCV

KFOLD = 3 # number of folds per split
NREP = 30 # number of random repeats
niter = KFOLD*NREP
SCORE = 'roc_auc'
ii = -1

internal_auc = np.zeros(niter)
external_auc = np.zeros(niter)


clf = hcc.clf_structure(CLFNAME)
param_grid = hcc.hp_grid(CLFNAME)


try:
    os.mkdir('NestedCV')
except FileExistsError:
    pass

pathsave  = f'NestedCV/{ENDPOINT}_{CLFNAME}'
try:
    os.mkdir(f"{pathsave}")
except FileExistsError:
    pass

    
  
# =============
# Outer CV
# =============

# Initialize arrays for plotting
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)


colnames = ['seed', 'fold']+features+['n_features']+\
    list(param_grid.keys())+['internal_auc', 'external_auc']
LogDF = pd.DataFrame(data = np.zeros((niter, len(colnames))), 
                 columns = colnames)

for seed in range(NREP):
    cv = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=seed)
    X, y = df[features], df[ENDPOINT]
    print(ENDPOINT, CLFNAME, seed)
    
    for k, (train, test) in enumerate(cv.split(X, y)):
        
        print("----------------------------------")
        print(f"seed {seed} \t fold {k}")   
        
        
        ii += 1 # iterable 
        
        LogDF.seed[ii] = seed
        LogDF.fold[ii] = k
        
        X_train, y_train = X.iloc[train], y.iloc[train]
        X_test, y_test = X.iloc[test], y.iloc[test]
        
        X_train.reset_index(drop=True)
        y_train.reset_index(drop=True)
        X_test.reset_index(drop=True)
        y_test.reset_index(drop=True)
        

        # ===============
        # Preprocessing
        # ===============
        print("\t data preprocessing")
        # Imputation
        X_train = ibr.impute_data(X_train)
        X_test  = ibr.impute_data(X_test) # X_train influence the imputation but is not imputed
        
        # Scale continuous features
        stdsc = StandardScaler()
        X_train[contfeatures]  = stdsc.fit_transform(X_train[contfeatures].copy())
        X_test[contfeatures]  = stdsc.transform(X_test[contfeatures].copy())
        
        
        # =================
        # Feature Selection
        # =================
        print("\t feature selection")
        # Backward Feature Elimination
        thisfeatureset = features.copy() # dynamic list of features
        n_features = len(thisfeatureset)
        OptimalSet = features.copy()
        
        clfFS = GridSearchCV(clf, param_grid, scoring=SCORE, cv = KFOLD,
                             refit=True, return_train_score=True)
        clfFS.fit(X_train, np.ravel(y_train))
        Smax = clfFS.best_score_

        
        for kk in range(len(features)-1):
            
            if kk%5==0:
                print(f"\t \t feature set length : {len(thisfeatureset)}")
            this_Xtrain = X_train[thisfeatureset]
            this_Xtest = X_test[thisfeatureset]
            
            # Test current model
            clfFS = GridSearchCV(clf, param_grid, scoring=SCORE, cv= KFOLD,
                             refit=True, return_train_score=True)
            clfFS.fit(this_Xtrain, np.ravel(y_train.astype('bool')))
            Si = clfFS.best_score_
            if Si>=Smax:
                OptimalSet = thisfeatureset.copy()
                Smax = Si
                
            # remove least important feature
            result = permutation_importance(clfFS, this_Xtrain, y_train, scoring=SCORE, 
                                            random_state=0)
            importances = result.importances_mean
            
            idx = np.argmin(importances)
            least_important = thisfeatureset[idx]
            thisfeatureset.remove(least_important)
            
        
        # Record optimal set
        for x in OptimalSet:
            LogDF[x][ii]+=1
        
        LogDF['n_features'][ii]=len(OptimalSet)
            
        # =====================
        # Hyperparameter Tuning
        # =====================
        print("\t hyperparameter tuning")
        X_train = X_train[OptimalSet]
        X_test = X_test[OptimalSet]
        clfGS = GridSearchCV(clf, param_grid, scoring=SCORE, cv=KFOLD,
                             refit=True, return_train_score=True)
        clfGS.fit(X_train, np.ravel(y_train.astype('bool')))
        internal_auc[ii] = clfGS.best_score_
        Model_k = clfGS.best_estimator_
        
        
        
        # Record optimal hyperpars
        for p in list(param_grid.keys()):
            LogDF[p][ii] = clfGS.best_params_[p]
            
        
        # =================================================
        # Evaluation of the optimal internal model M(n*,p*)
        # =================================================
        print("\t model evaluation")
        rocFig = plot_roc_curve(Model_k, X_test, y_test)
        external_auc[ii] = rocFig.roc_auc
        interp_tpr = np.interp(mean_fpr, rocFig.fpr, rocFig.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(external_auc[ii])
        plt.close()
        

            
        LogDF['internal_auc'][ii] = internal_auc[ii]
        LogDF['external_auc'][ii] = external_auc[ii]
        # print('seed %i  fold %i' %(seed, k), '  n_features', len(OptimalSet), 
        #       '  internal %0.4f'%internal_auc[ii], '  external %0.4f'%external_auc[ii])
    
    LogDF.to_excel(pathsave+'/INTER_LOGGING.xlsx', float_format="%.4f")    
    
#%%
# ==========================
# Logging Data
# ==========================
StatDF = pd.DataFrame(data=0, 
                      index = ['STATS'], columns = colnames)
for col in colnames:
    if col in features or ['internal_auc', 'external_auc']:
        StatDF[col]['STATS'] = sum(LogDF[col][0:niter])
    # elif col in list(param_grid.keys())+['internal_'+auc, 'external_'+auc]:
    elif col in ['internal_auc', 'external_auc']:
        StatDF[col]['STATS'] = sum(LogDF[col][0:niter])/niter

LogDF = LogDF.append(StatDF)
LogDF.to_excel(f"{pathsave}/LOG_{ENDPOINT}_{CLFNAME}.xlsx", float_format="%.4f")
    

#%% ROC

ibr.plot_roc(tprs, mean_fpr, aucs, ENDPOINT, CLFNAME, pathsave)


#%% FS heatmap
fsDF = LogDF[features][1:-1]
sns.heatmap(fsDF, cmap='binary', cbar=False)
plt.ylabel('Sample')
plt.title(f"{ENDPOINT} {CLFNAME}")
plt.savefig(pathsave+'/featureset_Heatmap.png', 
                            dpi=600, format = 'png', bbox_inches='tight')
plt.close()

#%%
for hpar in list(param_grid.keys()):
    categories = LogDF[hpar].sort_values().astype(str)
    categories.hist(grid=False, figsize=(4,4), color='gainsboro', 
                    edgecolor='black', rwidth=0.8, lw = 2)
    plt.title(f"{ENDPOINT} {CLFNAME}")    
    plt.xticks(rotation=45)
    plt.xlabel('Optimal value')
    plt.ylabel(hpar)
    plt.savefig(pathsave+'/DistributionOf_'+hpar+'.png', 
                                  dpi=600, format = 'png', bbox_inches='tight')
    plt.close()

#%%
fig = plt.subplots(figsize=(3, 3))
internal = LogDF['internal_auc'][0:-1]
external = LogDF['external_auc'][0:-1]
plt.scatter(internal, external, c='gainsboro', edgecolor='black',s=10)
plt.plot([0, 1], [0, 1], color='black')
plt.xlabel('Internal auc')
plt.ylabel('External auc')
plt.title(f"{ENDPOINT} {CLFNAME}")
plt.savefig(pathsave+'/Internal_vs_External_Scores.png', 
                              dpi=600, format = 'png', bbox_inches='tight')
plt.close()


# ===================
# Unique Optimal Sets
# ===================
df1 = LogDF[features]
df2 = LogDF[features+list(param_grid.keys())+['internal_auc']+['external_auc']]
df1 = df1.iloc[0:-1]
df2 = df2.iloc[0:-1]

UniqueOptimalSets = df1.drop_duplicates().copy()  
UniqueOptimalSets['n_models']       = np.nan
for p in list(param_grid.keys()):
    UniqueOptimalSets['mode_'+p]   = np.nan
    
UniqueOptimalSets['avg_int_auc']   = np.nan
UniqueOptimalSets['avg_ext_auc']   = np.nan
UniqueOptimalSets['min_ext_auc']   = np.nan
UniqueOptimalSets['max_ext_auc']   = np.nan
    
# find unique model
for row1 in range(UniqueOptimalSets.shape[0]):
    subrows = []
    for row2 in range(df2.shape[0]):
        if (UniqueOptimalSets[features].iloc[row1].values == df2[features].iloc[row2].values).all():
            subrows.append(row2)
        
    subset = df2.iloc[subrows]
    UniqueOptimalSets['n_models'].iloc[row1]        = subset.shape[0]
    
    for p in list(param_grid.keys()):
        UniqueOptimalSets['mode_'+p].iloc[row1]     = subset[p].mode().values[0]
        
    UniqueOptimalSets['avg_int_auc'].iloc[row1]    = subset['internal_auc'].mean()
    UniqueOptimalSets['avg_ext_auc'].iloc[row1]    = subset['external_auc'].mean()
    UniqueOptimalSets['min_ext_auc'].iloc[row1]    = subset['external_auc'].min()
    UniqueOptimalSets['max_ext_auc'].iloc[row1]    = subset['external_auc'].max()


# sort by top models       
UniqueOptimalSets = UniqueOptimalSets.sort_values(by=['n_models'], ascending=False)


UniqueOptimalSets.insert(0,'Model', None)
UniqueOptimalSets.insert(1,'nvars', None)
for row in range(UniqueOptimalSets.shape[0]):
    modelVars = []
    for x in features:
        if UniqueOptimalSets[x].iloc[row] == 1:
            modelVars.append(x)
    
    UniqueOptimalSets['Model'].iloc[row] = modelVars
    UniqueOptimalSets['nvars'].iloc[row] = len(modelVars)
    
UniqueOptimalSets = UniqueOptimalSets.drop(columns=features)
totalmodels = UniqueOptimalSets.shape[0]
UniqueOptimalSets.to_excel(pathsave+'/OptimalModels.xlsx')


# save top 10
if totalmodels > 10:
    UniqueOptimalSets = UniqueOptimalSets.iloc[0:10]

UniqueOptimalSets.dfi.export(pathsave+"/Top10Models.png")




















































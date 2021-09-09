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
import hcc_survmodels as hcc
import ibrahim_functions as ibr

#
from sksurv.util import Surv

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
endpoint = 'OS' # OS or RSF
estname  = 'RSF' 

#%%
if endpoint == 'OS':
    event = 'death'
    time  = 'followup'
    
elif endpoint == 'PFS':
    event = 'death'
    time  = 'NLF_months'

#%% 
df = pd.read_excel("../dataset.xlsx")

features = ['sex', 'age', 
            'cirrhosis', 'cirrhosis_etiology', 'liversize', 'PVT', 'CP0',
            'newDx',  'lesion_size', 'lesion_number', 'GTV', 
            'ALB0', 'BIL0', 'PLT0', 'AFP0', 'ALC0', 
            'proton', 'Fx', 'TD', 'EUD','EQD2_MLD', 
            'EQD2_V5', 'EQD2_V10']


df = df[features + [event, time]]

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

   
df = df.drop(columns = 'cirrhosis_etiology')     
features = list(df.columns.values)
features.remove(event)
features.remove(time)

df = df.dropna(subset = [event, time])
df = df.reset_index(drop = True)        
        
contvars = []
for xx in features:
    if len(df[xx].unique())>10:
        contvars.append(xx)
        
        
X = df[features]
y = Surv.from_dataframe(event, time, df)

#%% Prepare nCV

nfold = 10 # number of folds per split
nrep = 30 # number of random repeats
niter = nfold*nrep
ii = -1

internal_c = np.zeros(niter)
external_c = np.zeros(niter)


est = hcc.est_structure(estname)
param_grid = hcc.hp_grid(estname)

try:
    os.mkdir('NestedCV')
except FileExistsError:
    pass

pathsave  = f'NestedCV/{endpoint}_{estname}'
try:
    os.mkdir(f"{pathsave}")
except FileExistsError:
    pass

npatients = df.shape[0]



colnames = ['seed', 'fold']+features+['n_features']+\
    list(param_grid.keys())+['internal_c', 'external_c']
    
LogDF = pd.DataFrame(data = np.zeros((npatients, len(colnames))), 
                 columns = colnames)

#%% Starting the outer loop
for seed in range(nrep):
    cv = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed)
    
    print(endpoint, estname, seed)
    
    for k, (train, test) in enumerate(cv.split(X, df[event].astype(int))):
        
        print("----------------------------------")
        print(f"seed {seed} \t fold {k}")
            
        ii += 1 # iterable 
        
        LogDF.seed[ii] = seed
        LogDF.fold[ii] = k
        
        X_train, y_train = X.iloc[train], y[train]
        X_test, y_test   = X.iloc[test], y[test]
        
        X_train.reset_index(drop=True)
        X_test.reset_index(drop=True)
        
        
        # Imputation
        X_train = ibr.impute_data(X_train)
        X_test  = ibr.impute_data(X_test) 
        
        # Scale continuous features
        stdsc = StandardScaler()
        X_train[contvars]  = stdsc.fit_transform(X_train[contvars])
        X_test[contvars]  = stdsc.transform(X_test[contvars])

        
        
        # =================
        # Feature Selection
        # =================
        # Backward Feature Elimination
        ThisFeatureSet = features.copy() # dynamic list of features
        n_features = len(ThisFeatureSet)
        OptimalSet = features.copy()
        
        estFS = GridSearchCV(est, param_grid, refit=True, return_train_score=True)
        estFS.fit(X_train, y_train)
        Smax = estFS.best_score_
        
        for kk in range(len(features)-1):
            
            if kk%5==0:
                print(f"\t \t feature set length : {len(ThisFeatureSet)}")
                
            this_Xtrain = X_train[ThisFeatureSet]
            this_Xtest = X_test[ThisFeatureSet]
            
            # Test current model
            estFS = GridSearchCV(est, param_grid, cv= nfold, refit=True, return_train_score=True)
            estFS.fit(this_Xtrain, y_train)
            Si = estFS.best_score_
            if Si>=Smax:
                OptimalSet = ThisFeatureSet.copy()
                Smax = Si
                
            # remove least important feature
            result = permutation_importance(estFS, this_Xtrain, y_train, random_state=0)
            importances = result.importances_mean
            
            idx = np.argmin(importances)
            least_important = ThisFeatureSet[idx]
            ThisFeatureSet.remove(least_important)
        
        # Record optimal set
        for x in OptimalSet:
            LogDF[x][ii]+=1
        
        LogDF['n_features'][ii]=len(OptimalSet)
            
        # =====================
        # Hyperparameter Tuning
        # =====================
        X_train = X_train[OptimalSet]
        X_test = X_test[OptimalSet]
        estGS = GridSearchCV(est, param_grid, cv=nfold, refit=True, return_train_score=True)
        estGS.fit(X_train, y_train)
        internal_c[ii] = estGS.best_score_
        Model_k = estGS.best_estimator_
        
        # Record optimal hyperpars
        for p in list(param_grid.keys()):
            LogDF[p][ii] = estGS.best_params_[p]
            
        
        # =================================================
        # Evaluation of the optimal internal model M(n*,p*)
        # =================================================
        if estname == 'Cox':
            pred_curves = Model_k.predict_survival_function(X_test)
        elif estname == 'RSF':
            pred_curves = Model_k.predict_survival_function(X_test, return_array=True)
        
        # risk groups
        risk = Model_k.predict(X_test)
        risk_group = pd.qcut(risk, 2, labels=False, duplicates='drop')   
        
        lh_ix = (risk_group==0)  |  (risk_group == np.max(risk_group))
        X_lh  = X_test[lh_ix]
        y_lh  = y_test[lh_ix]
        
        
        # predictions
        external_c[ii] = Model_k.score(X_lh, y_lh)   
 
    
        # save
        LogDF['external_c'][ii] = external_c[ii]                         
        LogDF['internal_c'][ii] = internal_c[ii]
            
            
        print('seed %i  fold %i' %(seed, k), '  n_features', len(OptimalSet), 
              '  int %0.2f'  %LogDF['internal_c'][ii], 
              '  ext %0.2f' %LogDF['external_c'][ii])

    LogDF.to_excel(pathsave+'/INTER_LOGGING.xlsx', float_format="%.4f")    


#%% Logging Data

StatDF = pd.DataFrame(data=0, 
                      index = ['STATS'], columns = colnames)
for col in colnames:
    if col in features:
        StatDF[col]['STATS'] = sum(LogDF[col][0:npatients])

    elif col in ['internal_c', 'external_c']:
        StatDF[col]['STATS'] = sum(LogDF[col][0:npatients])/npatients

LogDF = LogDF.append(StatDF)
LogDF.to_excel(pathsave+'/LOG__'+endpoint+estname+'.xlsx', float_format="%.4f")
    


 
    
#%% Heatmap of feature selection
fsDF = LogDF[features][1:-1]
sns.heatmap(fsDF, cmap='binary', cbar=False)
plt.ylabel('Sample')
plt.title(event + '  '+ estname + '  %ifoldX%i'%(nfold,nrep))
plt.savefig(pathsave+'/FeatureSet_Heatmap.png', 
                            dpi=600, format = 'png', bbox_inches='tight')
plt.close()

#%% Hyperparameters Histograms

for hpar in list(param_grid.keys()):
    categories = LogDF[hpar].sort_values().astype(str)
    categories.hist(grid=False, figsize=(4,4), color='gainsboro', 
                    edgecolor='black', rwidth=0.8, lw = 2)
    plt.title(event + '  '+ estname + '  %ifoldX%i'%(nfold,nrep))
    plt.xticks(rotation=45)
    plt.xlabel('Optimal value')
    plt.ylabel(hpar)
    plt.savefig(pathsave+'/DistributionOf_'+hpar+'.png', 
                                  dpi=600, format = 'png', bbox_inches='tight')
    plt.close()


#%% External versus internal score
fig = plt.subplots(figsize=(3, 3))
internal = LogDF['internal_c'][0:-1]
external = LogDF['external_c'][0:-1]
plt.scatter(internal, external, c='gainsboro', edgecolor='black',s=10)
plt.plot([0, 1], [0, 1], color='black')
plt.xlabel('Internal_c')
plt.ylabel('External_c')
plt.title(event + '  '+ estname + '  %ifoldX%i'%(nfold,nrep))
plt.savefig(pathsave+'/Internal_vs_External_Scores.png', 
                              dpi=600, format = 'png', bbox_inches='tight')
plt.close()



#%% Unique Optimal Sets

df1 = LogDF[features]
df2 = LogDF[features+list(param_grid.keys())+['internal_c']+['external_c']]
df1 = df1.iloc[0:-1]
df2 = df2.iloc[0:-1]

UniqueOptimalSets = df1.drop_duplicates().copy()  
UniqueOptimalSets['n_models']       = np.nan
for p in list(param_grid.keys()):
    UniqueOptimalSets['mode_'+p]   = np.nan
    
UniqueOptimalSets['avg_int_c']   = np.nan
UniqueOptimalSets['avg_ext_c']   = np.nan
UniqueOptimalSets['min_ext_c']   = np.nan
UniqueOptimalSets['max_ext_c']   = np.nan
    
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
        
    UniqueOptimalSets['avg_int_c'].iloc[row1]    = subset['internal_c'].mean()
    UniqueOptimalSets['avg_ext_c'].iloc[row1]    = subset['external_c'].mean()
    UniqueOptimalSets['min_ext_c'].iloc[row1]    = subset['external_c'].min()
    UniqueOptimalSets['max_ext_c'].iloc[row1]    = subset['external_c'].max()


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













































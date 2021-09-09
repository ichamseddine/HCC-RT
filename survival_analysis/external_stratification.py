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
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.plotting import add_at_risk_counts
from lifelines import NelsonAalenFitter, WeibullFitter
from lifelines.utils import concordance_index

#
from sksurv.metrics import concordance_index_censored
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
endpoint = 'OS'  # OS, PFS
estname  = 'RSF' # Cox, RSF


#%%
if endpoint == 'OS':
    event = 'death'
    time  = 'followup'
    
elif endpoint == 'PFS':
    event = 'death'
    time  = 'NLF_months'
    
features = hcc.optimal_feature_set(endpoint, estname)
# cph = hcc.optimal_est(endpoint, estname)

#%% load model

filename = f'FinalModels/{endpoint}_{estname}.pkl'
est      = pickle.load(open(filename, 'rb'))
    
#%%

pathsave = 'stratification_external'
try:
    os.mkdir(pathsave)
except FileExistsError:
    pass
    
pathsave = f'stratification_external/{estname}'
try:
    os.mkdir(pathsave)
except FileExistsError:
    pass

#%% Read data
df = pd.read_excel("../mda_dataset.xlsx")

df = df[features+[time, event]]
df = df.dropna(subset = [event, time])
df = df.reset_index(drop=True)


contvars = []
for xx in features:
    if len(df[xx].unique())>10:
        contvars.append(xx)

for xx in features:
    if xx not in contvars:
        df[xx] = np.round(df[xx])

#%% Data scaling to internal dataset
        
dfint = pd.read_excel("../dataset.xlsx")
dfint[features] = ibr.impute_data(dfint[features])


# impute to internal
imp = IterativeImputer()
dfint[features] = imp.fit_transform(dfint[features])
df[features] = imp.transform(df[features])


# Scale continuous features
stdsc = StandardScaler()
stdsc.fit(dfint[contvars])
df[contvars]  = stdsc.transform(df[contvars])



#%% predict risk scores using the cox model
df1 = df.copy()
df1['risk'] = est.predict(df[features])


#%% useful function
def KM_startification(df1, time, event, fsave, c_index=-1):
    # df must include a group column with entries 0 (low risk) and 1 (high risk)
    # time, event, and fname are strings
    
    ix_lo = df1[df1.group==0].index.values
    ix_hi = df1[df1.group==1].index.values
    
    plt.figure()
    kmf_lo = KaplanMeierFitter()
    kmf_lo.fit(df1.iloc[ix_lo][time], df1.iloc[ix_lo][event], label = 'low risk')
    kmf_lo.plot_survival_function(color = 'gray', show_censors=True, censor_styles={'ms': 6})
    
    kmf_hi = KaplanMeierFitter()
    kmf_hi.fit(df1.iloc[ix_hi][time], df1.iloc[ix_hi][event], label = 'hi risk')
    kmf_hi.plot_survival_function(color = 'k',  show_censors=True, censor_styles={'ms': 6})
    plt.xlim([0,60])
    plt.legend().remove()
    add_at_risk_counts(kmf_lo, kmf_hi)
    
    # Hazard ratio based on hazard rate calculated using Weibull fitter
    timeline = np.linspace(1, 60, 100)
    wbf0 = WeibullFitter().fit(df1.iloc[ix_lo][time], df1.iloc[ix_lo][event])
    h0_t = wbf0.hazard_at_times(timeline)
    
    wbf1 = WeibullFitter().fit(df1.iloc[ix_hi][time], df1.iloc[ix_hi][event])
    h1_t = wbf1.hazard_at_times(timeline)
    
    HR_t = h1_t/h0_t
    
    HR_m = np.mean(HR_t.values)
    HR_lo = np.percentile(HR_t.values, 2.5)
    HR_hi = np.percentile(HR_t.values, 97.5)
    
    if c_index>0:
        ttl = f"c-index = {c_index:0.2f}    HR {HR_m:0.1f} 95%CI[{HR_lo:0.1f}, {HR_hi:0.1f}]    \nmedian survivals: {kmf_hi.median_survival_time_:0.1f} and {kmf_lo.median_survival_time_:0.1f} months"
    else:
         ttl = f"HR {HR_m:0.1f} 95%CI[{HR_lo:0.1f}, {HR_hi:0.1f}]    \nmedian survivals: {kmf_hi.median_survival_time_:0.1f} and {kmf_lo.median_survival_time_:0.1f} months"
    plt.title(ttl, fontsize = 12)
    plt.tight_layout()
    plt.savefig(f"{fsave}.png", format = 'png', dpi = 600)

#%% Straify by median risk
group = pd.qcut(df1['risk'], q=2, labels=False, duplicates = 'drop')
df1['group'] = group.values
fname = 'MedianRisk'
fsave = f"{pathsave}/{fname}"
cindex,_,_,_,_ = concordance_index_censored(df1[event].astype(bool), df1[time], df1['risk'])  
KM_startification(df1, time, event, fsave, cindex)

#%% Stratify by top-buttom quartiles
group = pd.qcut(df1['risk'], q=4, labels=False, duplicates = 'drop')
df2 = df1.copy()
df2['group'] = group.values
df2 = df2[(df2.group==0) | (df2.group==3)]
df2.group[df2.group==3]=1
df2 = df2.reset_index(drop=True)
fname = 'QuartileRisk'
fsave = f"{pathsave}/{fname}"

cindex,_,_,_,_ = concordance_index_censored(df2[event].astype(bool), df2[time], df2['risk']) 
KM_startification(df2, time, event, fsave, cindex)

    
#%% Stratift by CP0
var = 'CP0'

df3 = df[[event, time, var]]
df3 = df3.dropna()
df3 = df3.reset_index(drop=True)

df3['group'] = np.zeros(df3.shape[0])
df3.group[df3[var]>6] =1


fname = f'By{var}'
fsave = f"{pathsave}/{fname}"

KM_startification(df3, time, event, fsave, c_index = -1)

#%% Stratift by ALBI0
var = 'ALBI0'

df4 = pd.read_excel("../mda_dataset.xlsx")
df4 = df4[[event, time, var]]
df4 = df4.dropna()
df4 = df4.reset_index(drop=True)


df4['group'] = np.zeros(df4.shape[0])
df4.group[df4[var]>-2.6] =1


fname = f'By{var}'
fsave = f"{pathsave}/{fname}"

KM_startification(df4, time, event, fsave, c_index = -1)





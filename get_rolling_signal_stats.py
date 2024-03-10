# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 14:10:27 2023

@author: cdim
"""

# %% import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from matplotlib.backends.backend_pdf import PdfPages
import time, re, gc, os
from tqdm import tqdm
import wrds
from pathlib import Path
from scipy.stats.mstats import winsorize
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas_datareader.data as web
from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count
import warnings


# %% set up / user input

warnings.simplefilter(action='ignore', category=FutureWarning)
tqdm.pandas()
tqdm.monitor_interval = 0

if cpu_count()>30:
    CPUUsed = 20
else:
    CPUUsed = cpu_count()-1

MachineUsed = 1

if MachineUsed==1: # Chuks 
    path_input = Path('C:/users/cdim/Dropbox/ChenDim/Stop-Worrying/Data/')
    path_output = Path('C:/users/cdim/Dropbox/ChenDim/Stop-Worrying/Data/')
    path_figs = Path('C:/users/cdim/Dropbox/ChenDim/Stop-Worrying/Chuks/Figures/')
    path_tables = Path('C:/users/cdim/Dropbox/ChenDim/Stop-Worrying/Chuks/Tables/')
elif MachineUsed==2: # Andrew 

    # CPUUsed = 8

    # get working directory
    path_root = Path(os.getcwd() + '/../../')

    path_input = Path(path_root / 'Data/')
    path_output = Path(path_root / 'Data/')
    path_figs = Path(path_root / 'Andrew/Figures/')
    path_tables = Path(path_root / 'Andrew/Tables/')    



# Statistical choices
MIN_YEAR = 1963
IS_N_YEARS = 20 # number of years in in-sample period
OOS_BEGIN_YEAR_START = 1983 # start of oos period
OOS_N_YEARS = 1 # number of years in oos period
MAX_LAGS = 0 # for standard error correction
formula = 'ret ~ 1' # formula for performance measurement



# define minimum number of observations for each signal in the in-sample
if IS_N_YEARS > 3:
    IS_MIN_OBS = 36 
else:
    IS_MIN_OBS = IS_N_YEARS * 12

# define minimum number of observations for each signal in the out-of-sample
if OOS_N_YEARS > 3:
    OOS_MIN_OBS = 36 
else:
    OOS_MIN_OBS = OOS_N_YEARS * 12

# %% Define Functions

# function to compute alphas and t-stats for in sample and out of sample
def get_alpha_tstat(s_id, df_sig, formula, oos_begin_year, oos_end_year): 
    
    res_2, res_3 = [], []
        
    df_IS = df_sig.query('year < @oos_begin_year')
    df_OOS = df_sig.query('@oos_begin_year <= year < @oos_end_year')
    if len(df_IS)>=IS_MIN_OBS and len(df_OOS)>=OOS_MIN_OBS:     
        
        # In-sample (IS)
        mod = smf.ols(formula, data=df_IS).fit(
            cov_type='HAC', cov_kwds={'maxlags': MAX_LAGS})
        res_2 = [s_id, df_IS['ret'].mean(), mod.params['Intercept'], mod.tvalues['Intercept'], mod.nobs, 'IS']
        
        # Out-of-Sample (OOS)
        mod = smf.ols(formula, data=df_OOS).fit(
            cov_type='HAC', cov_kwds={'maxlags': MAX_LAGS})
        res_3 = [s_id, df_OOS['ret'].mean(), mod.params['Intercept'], mod.tvalues['Intercept'], mod.nobs, 'OOS']
    
    return [res_2, res_3]


#%% load data

# load family strategies data
signal_family_names = [
    # 'ticker_Harvey2017JF_ew.csv', 'ticker_Harvey2017JF_vw.csv', 
    'DataMinedLongShortReturnsVW.csv', 'DataMinedLongShortReturnsEW.csv',
    'PastReturnSignalsLongShort.csv.gzip',
    'TickerSignalsLongShort.csv.gzip',
    'RavenPackSignalsLongShort.csv.gzip'
                       ]

signal_dic = {}
for name in tqdm(signal_family_names):
    df = pd.read_csv(path_input / f'{name}')
    label = re.sub(r'\.csv(?:\.gzip)?', '', name) # get file name without file extension

    if label in ['DataMinedLongShortReturnsVW', 'DataMinedLongShortReturnsEW']:
        df['day'] = 1
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    
    # elif label in ['ticker_Harvey2017JF_ew', 'ticker_Harvey2017JF_vw']: 
    #     df['signalid'], _ = pd.factorize(df['signalname'])
    #     df['date'] = pd.to_datetime(df['date']) - pd.offsets.MonthBegin() # makes dates to be first of each month
    
    elif label in ['PastReturnSignalsLongShort', 'RavenPackSignalsLongShort',
                   'TickerSignalsLongShort']:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year

        if df['date'].max().day != 1:
            # makes dates to be first of each month
            df['date'] = df['date'] - pd.offsets.MonthBegin() 
        
        
        # retain signal-periods with at least 2 stocks in the long and short leg
        # only relevant for RavenPackSInalsLongShort where this scenario can arise
        df = df.query('nshort>=2 & nlong>=2')
        
        # seperate value-weighted and equal-weighted portolio returns as different datasets
        signal_dic[f'{label}_ew'] = df[['signalid', 'date', 'year', 'ret_ew']].rename(
            columns={'ret_ew': 'ret'}).sort_values(['signalid', 'date'])
        
        signal_dic[f'{label}_vw'] = df[['signalid', 'date', 'year', 'ret_vw']].rename(
            columns={'ret_vw': 'ret'}).sort_values(['signalid', 'date'])
        continue

    
    df['year'] = df['date'].dt.year
    signal_dic[label] = df[['signalid', 'date', 'year', 'ret']].sort_values(['signalid', 'date'])
    
    
     
#%% get mean return and tstats for rolling in-sample and out-of-sample period

signal_tstats = []

# find last year for each signal family
last_year_ok = min([df['year'].max() for name, df in signal_dic.items()]) + 1 - OOS_N_YEARS


# compute alphas and t-stats in parallel
with Parallel(n_jobs=CPUUsed, verbose=3) as parallel:
    for name, df in signal_dic.items():
        
        print(f'Process Started {name}')
        
        for oos_begin_yr in range(OOS_BEGIN_YEAR_START, last_year_ok+1):
        
            print(f'OOS begin year: {oos_begin_yr}')
            oos_end_yr = oos_begin_yr + OOS_N_YEARS
            is_begin_yr = oos_begin_yr - IS_N_YEARS
            
            # select that starting from in-sample begin year
            # and keep only signals with enough observations
            df_use = df.query('@is_begin_yr <= year < @oos_end_yr').dropna()
            if len(df_use)==0:
                continue

            # execute procedure in parallel
            df_alpha = parallel(delayed(get_alpha_tstat)(s_id, df_sig, formula, oos_begin_yr, oos_end_yr)
                                  for s_id, df_sig in df_use.groupby('signalid'))
            
            # unpack list of lists
            df_alpha = [ls for sub_ls in df_alpha for ls in sub_ls]
            
            # check if any of the results in non-empty
            if sum(map(len, df_alpha)) == 0:
                continue
            
            # convert to data frame and include identifiers
            col_names = ['signalid', 'mean_ret', 'alpha', 'tstat', 'nobs', 's_flag']
            df_alpha = pd.DataFrame(df_alpha, columns=col_names)
            df_alpha['is_begin_year'] = is_begin_yr
            df_alpha['oos_begin_year'] = oos_begin_yr
            df_alpha['signal_family'] = name
        
            signal_tstats.append(df_alpha)
            

# concatenate dataframes and perform final clean up
signal_tstats = pd.concat(signal_tstats)
signal_tstats = signal_tstats.dropna(subset=['signalid'])
signal_tstats['signalid'] = signal_tstats['signalid'].astype(int)
signal_tstats = signal_tstats.set_index(['signal_family', 'oos_begin_year', 'signalid'])
signal_tstats = signal_tstats.drop(columns=['alpha'])

#%%  save data

signal_tstats.to_csv(path_output / f'OOS_signal_tstat_OosNyears{OOS_N_YEARS}.csv.gzip')

print('saved csv to ' + str(path_output))

#%%
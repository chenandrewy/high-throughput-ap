# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:47:38 2024

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
import pyreadr
from copy import copy
from itertools import combinations
from functools import reduce


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

    CPUUsed = 8

    # get working directory
    path_root = Path(os.getcwd() + '/../../')

    path_input = Path(path_root / 'Data/')
    path_output = Path(path_root / 'Data/')
    path_figs = Path(path_root / 'Andrew/Figures/')
    path_tables = Path(path_root / 'Andrew/Tables/')    





#%% define functions for constructing long short portfolios



def get_signal_long_short_ret(signals_t, value_weight_t, ret_t):
    
    # This function constructs long short portfolio for each signal for a specific date
    
    # Require at least 20 stocks -- equivalent to 2 stocks per decile portfolio
    if len(signals_t)<20:
        return {}
    
    nm = signals_t.columns[0]
    
    # take top and bottom 10 percent without explicitly creating all portfolios
    top_threshold = signals_t[nm].quantile(0.9)
    bottom_threshold = signals_t[nm].quantile(0.1)
    signals_t['decile'] = np.where(signals_t[nm]>=top_threshold, 10,
                                   np.where(signals_t[nm]<=bottom_threshold, 1,
                                   5))
    temp = signals_t[(signals_t['decile']==1) | (signals_t['decile']==10)]
    
    # merge lag market cap and day t return
    temp = temp.merge(value_weight_t, how='left', on='permno')
    temp = temp.merge(ret_t, how='left', on='permno')
    temp = temp.dropna(subset=['w', 'ret'])
    temp['nstock'] = 1
    
    # normalize weights to sum to 1
    temp['w'] = temp['w']/temp.groupby('decile')['w'].transform('sum')
    temp['ret_vw'] = temp.eval('ret * w')
    
    # compute value-weighted and equal-weighted portfolio returns
    temp = temp.groupby('decile').agg({'ret': 'mean', 'ret_vw': 'sum', 'nstock': 'sum'})
    
    # now get long short for value weighted and equal-weighted
    nstock = temp['nstock'].rename({1: 'nshort', 10:'nlong'}).to_dict()
    temp = temp[['ret', 'ret_vw']].T
    sig_long_short = (temp[10] - temp[1]).to_dict()
    sig_long_short = sig_long_short | nstock # concatenate the two dictionaries
    sig_long_short['signalname'] = nm
        
    return sig_long_short
    




def get_signals_and_returns(df_base_sig, cols, value_weight_t, ret_t):
    
    sfx = '_'.join(map(str, cols)) # sufix for signal names
    df = df_base_sig[[*cols]]
    dfs = np.exp(df) - 1 # simple returns for higher moments
    
    # compute signals based on cumulative return, std, skewness and kurtosis
    df_r = df.sum(axis=1, min_count=Ncomb).to_frame(f'ret_{sfx}').dropna() 
    permnos_use = df_r.index
    df_std = dfs.std(axis=1).loc[permnos_use].to_frame(f'std_{sfx}')
    df_skew = dfs.skew(axis=1).loc[permnos_use].to_frame(f'skew_{sfx}')
    df_kurt = dfs.kurtosis(axis=1).loc[permnos_use].to_frame(f'kurt_{sfx}')
 
    
    temp_sig = [df_r, df_std, df_skew, df_kurt]
    
    signal_ret = [get_signal_long_short_ret(signals_t, value_weight_t, ret_t) 
                  for signals_t in temp_sig]
    
    return signal_ret





#%% load and prepare data

# load crsp monthly returns from r-data format
df_crspm = pyreadr.read_r(path_input / 'CLZ/crspm.RData')
# take the dataframe containing the data. 
# it's first element of the dictionary values
df_crspm, = df_crspm.values() 

df_crspm['permno'] = df_crspm['permno'].astype(int)
df_crspm['date']  = pd.to_datetime(df_crspm['date'])
df_crspm['prc'] = df_crspm['prc'].abs()
df_crspm['w'] = df_crspm.eval('prc * shrout')

# apply filters for NYSE/NASDAQ/AMEX and share code 10/11
df_crspm = df_crspm.query('exchcd in [1, 2, 3] & shrcd in [10, 11]')


# take market value and price
cols_use = ['permno', 'date', 'w', 'prc']
df_mktval = df_crspm[cols_use].copy()


# take returns and unstack
cols_use = ['permno', 'date', 'ret']
df_crspm = df_crspm[cols_use].set_index(['permno', 'date']).sort_index(level=[0,1])
df_crspm['ret'] = df_crspm['ret'] / 100 # contert returns back to decimal instead of percent
df_crspm = df_crspm['ret'].unstack(0).sort_index()

# convert to log returns for easy aggregation
df_crspm = np.log(1+df_crspm)

# get integer indicators for dates
dates_id_map = pd.DataFrame(df_crspm.index)
dates_id_map.index = np.arange(1, len(dates_id_map)+1)
dates_id_map = dates_id_map['date'].to_dict()
reverse_dates_id_map = {v:k for k, v in dates_id_map.items()}

# map dates in the data to the integers above
df_crspm = df_crspm.rename(reverse_dates_id_map)
temp = pd.Series(dates_id_map).to_frame('date')
temp.index.name = 'date_id'
df_mktval = df_mktval.merge(temp.reset_index(), how='left', on='date'
                            ).drop(columns=['date']).rename(columns={'date_id': 'date'})
df_mktval = df_mktval.set_index(['date', 'permno']).sort_index(level=[0,1])


#%% compute signals' long-short return for each date in parallel



Nyrs = 5
nqtr = 4*Nyrs

# define quarter ids for each month over Nyrs years
qtr_ids = [[q]*3 for q in range(1, nqtr+1)]
qtr_ids = [v for sub_ls in qtr_ids for v in sub_ls]

# Generate all combinations of n items for a list of N>n items
Ncomb = 4
signal_names =  np.arange(1, nqtr+1)
sig_combinations = list(combinations(signal_names, Ncomb))

t_start = reverse_dates_id_map[pd.to_datetime('1962-01-31')]
t_ls = df_crspm.index[df_crspm.index>=t_start]


df_signal_long_short = []
with Parallel(n_jobs=CPUUsed, verbose=3) as parallel:
    
    for t in t_ls:
                
        # get market value weight on day t-1 for stocks with price above $1
        value_weight_t = df_mktval.loc[t-1].query('prc>=1').dropna()[['w']]
        permnos = value_weight_t.index
        
        # get the earliest possible date used in signals for day t 
        t_0 = t - Nyrs*12 
        
        # select data from t_0 to t-1 for stocks that have market cap on t-1 
        df_crspm_t = df_crspm.loc[t_0:t-1, permnos].dropna(how='all', axis=1)
        df_crspm_t_simp = np.exp(df_crspm_t) - 1
        permnos = df_crspm_t.columns
        
        print(f'\nDate: {dates_id_map[t]}; # stocks: {len(permnos)}\n')
        
        # get non-overlapping quarterly returns as base signals
        df_crspm_t['qtr'] = qtr_ids
        df_base_sig = df_crspm_t.groupby('qtr').sum(min_count=2).sort_index().T
        
        # get day t log return and convert back to simple return for long short returns 
        ret_t = np.exp(df_crspm.loc[t, permnos].to_frame('ret')) - 1
        
        
        # run procedure in parallel to compute expanded signals and their 
        # long-short returns        
        signal_ret_t = parallel(delayed(get_signals_and_returns)(
            df_base_sig, cols, value_weight_t, ret_t) for cols in sig_combinations)
        
        # unpack signals in list of lists
        signal_ret_t = [ls for sub_ls in signal_ret_t for ls in sub_ls]
        
        
        # compute additional signal returns based on original returns' quaterly moments
        # we don't compute kurtosis here because it requires at leat 4 values 
        mask = df_base_sig.isnull()
        df_crspm_t_simp['qtr'] = qtr_ids
        temp_std = df_crspm_t_simp.groupby('qtr').std().T
        temp_std[mask] = np.nan
        temp_skew = df_crspm_t_simp.groupby('qtr').skew().T
        temp_skew[mask] = np.nan

        df_base_sig.columns = [f'ret_{q}' for q in df_base_sig.columns]
        temp_std.columns = [f'std_{q}' for q in temp_std.columns]
        temp_skew.columns = [f'skew_{q}' for q in temp_skew.columns]
        
        addt_signals = ([df_base_sig[[c]].dropna() for c in df_base_sig.columns] +
                         [temp_std[[c]].dropna() for c in temp_std.columns] +
                         [temp_skew[[c]].dropna() for c in temp_skew.columns]
                         )
        
        addt_signal_ret = [get_signal_long_short_ret(signals_t, value_weight_t, ret_t) 
                           for signals_t in addt_signals]
        
        signal_ret_t = addt_signal_ret + signal_ret_t
        signal_ret_t = pd.DataFrame(signal_ret_t)
        signal_ret_t['date'] = dates_id_map[t]
        
        df_signal_long_short.append(signal_ret_t)
        
# final clean up
df_signal_long_short = pd.concat(df_signal_long_short)
df_signal_long_short = df_signal_long_short.reset_index()
df_signal_long_short = df_signal_long_short.rename(columns={'ret': 'ret_ew'})
df_signal_long_short[['ret_ew', 'ret_vw']] = df_signal_long_short[['ret_ew', 'ret_vw']] * 100

# get ids for signals
df_signal_long_short = df_signal_long_short.rename(columns={'signalid': 'signalname'})
df_signal_long_short['signalid'], _ = pd.factorize(df_signal_long_short['signalname'])
df_signal_names = df_signal_long_short[['signalid', 'signalname']].drop_duplicates()

cols_order = ['signalid', 'date', 'ret_ew', 'ret_vw', 'nshort', 'nlong',]
df_signal_long_short = df_signal_long_short.drop(columns=['signalname'])[cols_order]

#%% save data

df_signal_long_short.to_csv(path_output / 'PastReturnSignalsLongShort.csv.gzip', index=False)
df_signal_names.to_csv(path_output / 'PastReturnSignalNames.csv.gzip', index=False)





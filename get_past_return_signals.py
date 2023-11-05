# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 18:42:13 2023

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


# %% set up / user input

warnings.simplefilter(action='ignore', category=FutureWarning)
tqdm.pandas()
tqdm.monitor_interval = 0

if cpu_count()>30:
    CPUUsed = 20
else:
    CPUUsed = cpu_count()-2

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




# some global configurations
K_LS = np.array(list(np.arange(0,13)) + [24, 36, 48, 60]) # number of months to skip before portfolio formation date
J_LS = np.arange(1, 241) # number of months of returns used to construct signal
K_LS.sort()
J_LS.sort()




#%% define functions for constructing long short portfolios


def get_signals_stock(df, permno, t):
    
    # This function creates the signals for a specific stock and day t-1 base on Yan & Zheng (2017, RFS)
    
    df = df.reset_index().values
    stock_signals = []
    s_id = 0
    for k in K_LS: # number of months to skip before portfolio formation date
        for j in J_LS: # number of months of returns used to construct signal
            s_id += 1
            # get cumulative log return over window 
            t_arr = df[:,0]
            ret_obs = df[(t-k-j <= t_arr) & (t_arr <= t-k-1), 1]
            # ret_obs = df[t-k-j: t-k-1]
            # check if the stock has full observations to compute the signal
            if np.sum(~np.isnan(ret_obs)) < j:
                continue
            else:
                signal = ret_obs.sum()
                stock_signals.append([permno, s_id, signal])
    return stock_signals




def get_signal_long_short_ret(signals_t, value_weight_t, ret_t):
    
    # This function constructs long short portfolio for each signal for a specific date
    
    signals_t = pd.DataFrame(signals_t, columns=['permno', 's_id', 'signal']
                             ).dropna()
    
    # keep only signals with at least 20 stocks -- equivalent to 2 stocks per singal decile portfolio
    mask = signals_t.groupby(['s_id'])['signal'].transform('count') >= 20
    signals_t = signals_t[mask]
    
    
    # # Create decile portfolios and take portfolio 1 and 10 
    # # This approeahc fails if there are duplicate edges
    # signals_t['decile'] = signals_t.groupby(['s_id'])['signal'].transform(lambda x: pd.qcut(x, q=10, labels=False)) + 1
    # temp = signals_t.query('decile in [1, 10]')

    # take top and bottom 10 percent without explicitly creating all portfolios
    top_threshold = signals_t.groupby('s_id')['signal'].transform(lambda x: x.quantile(0.9))
    top = signals_t[signals_t['signal']>=top_threshold].reset_index(drop=True)
    top['decile'] = 10
    bottom_threshold = signals_t.groupby('s_id')['signal'].transform(lambda x: x.quantile(0.1))
    bottom = signals_t[signals_t['signal']<=bottom_threshold].reset_index(drop=True)
    bottom['decile'] = 1
    temp = pd.concat([bottom, top])
    
    
    # merge lag market cap and day t return
    temp = temp.merge(value_weight_t, how='left', on='permno')
    temp = temp.merge(ret_t, how='left', on='permno')
    temp = temp.dropna(subset=['w', 'ret'])
    temp['nstock'] = 1
    
    
    # normalize weights to sum to 1
    temp['w'] = temp['w']/temp.groupby(['s_id', 'decile'])['w'].transform('sum')
    temp['ret_vw'] = temp.eval('ret * w')
    
    # compute value-weighted and equal-weighted portfolio returns
    temp = temp.groupby(['s_id', 'decile']).agg({'ret': 'mean', 'ret_vw': 'sum', 'nstock': 'sum'})
    
    # now get long short for value weighted and equal-weighted
    temp = temp.unstack(-1)
    df_nstock = temp['nstock'].rename(columns={1: 'nshort', 10:'nlong'})
    signal_long_short = [df_nstock]
    for r in ['ret', 'ret_vw']:
        df = temp[r]
        signal_long_short.append((df[10] - df[1]).to_frame(r))
    signal_long_short = pd.concat(signal_long_short, axis=1)   
    
    return signal_long_short
    

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


t_start = reverse_dates_id_map[pd.to_datetime('1962-01-31')]
t_ls = df_crspm.index[t_start-1:]

df_signal_long_short = []

with Parallel(n_jobs=CPUUsed, verbose=5) as parallel:
    
    for t in t_ls:
                
        # get market value weight on day t-1 for stocks with price above $1
        value_weight_t = df_mktval.loc[t-1].query('prc>=1').dropna()[['w']]
        permnos = value_weight_t.index
        

        # get the earliest possible date used in signals for day t including a buffer of 12 months
        t_0 = t - (K_LS[-1] + J_LS[-1] + 12) 
        
        # select data from t_0 to t for stocks that are still active as of t-1 
        df_crspm_t = df_crspm.loc[t_0:t, permnos].dropna(how='all', axis=1)
        permnos = df_crspm_t.columns
        
        
        print(f'\nDate: {dates_id_map[t]}; # stocks: {len(permnos)}\n')

        # run procedure in parallel
        signals_t = parallel(delayed(get_signals_stock)(df_crspm_t[permno], permno, t) 
                             for permno in permnos)
    
        # unpack signals in list of lists
        signals_t = [ls for sub_ls in signals_t for ls in sub_ls]
        
        
        # get day t log return and convert back to simple return for long short returns 
        ret_t = np.exp(df_crspm.loc[t, permnos].to_frame('ret')) - 1

        
        # rank stocks and get long short portfolio return for each signal
        long_short = get_signal_long_short_ret(signals_t, value_weight_t, ret_t)
        long_short['date'] = dates_id_map[t]
        df_signal_long_short.append(long_short)
        
# final clean up
df_signal_long_short = pd.concat(df_signal_long_short)
df_signal_long_short = df_signal_long_short.reset_index()
df_signal_long_short = df_signal_long_short.rename(columns={'s_id': 'signalid', 'ret': 'ret_ew'})
cols_order = ['signalid', 'date', 'ret_ew', 'ret_vw', 'nshort', 'nlong',]
df_signal_long_short = df_signal_long_short[cols_order]
df_signal_long_short[['ret_ew', 'ret_vw']] = df_signal_long_short[['ret_ew', 'ret_vw']] * 100


#%% save data

df_signal_long_short.to_csv(path_output / 'PastReturnSignalsLongShort.csv.gzip', index=False)




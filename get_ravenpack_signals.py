# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 17:34:48 2023

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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from functools import reduce, partial




# %% set up / user input

warnings.simplefilter(action='ignore', category=FutureWarning)
tqdm.pandas()
tqdm.monitor_interval = 0
idx = pd.IndexSlice

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




# The minimum of average number of stocks a news group needs to cover per month to be used as base signal
MIN_AVE_STOCKS = 100 


#%% Some functions we use


def get_diff_based_features(df, v):
    
    '''
    Generates additional features from the original sentiment signals.
    Based on change of original signal with some bechmarks:
        cross-sectional median, past moving average
    '''
    
    # initialize data for save results with original signal in it
    df_sig_add = [df]

    
    # re-arrange data to have permnos as columns and dates as index
    df = df[v].unstack(0)
    
    # initialize a dummy dataframe for inputing results
    temp = df.copy()
    
    # get change relative to crossectional median
    temp[:] = df.values - np.median(df.values, axis=1).reshape(-1,1)
    df_sig_add.append(temp.stack(dropna=False).to_frame(f'{v}_smid'))

    
    # get change relative to some moving average
    for i in [1, 3, 6]:
        
        # get rolling average
        df1 = df.rolling(window=i).mean().shift()
        # df_sig_add.append(df1.stack(dropna=False).to_frame(f'{v}_m{i}'))
        
        # get deviation of current value from rolling average
        temp[:] = df.values - df1.values
        df_sig_add.append(temp.stack(dropna=False).to_frame(f'{v}_d{i}'))
        
        
        # get deviation of rolling average from its median
        temp[:] = df1.values - np.median(df1.values, axis=1).reshape(-1,1)
        df_sig_add.append(temp.stack(dropna=False).to_frame(f'{v}_m{i}_smid'))
        
    # merge all
    df_sig_add = reduce(lambda l, r: pd.merge(l, r, how='outer', on=['permno', 'datem']), 
                        df_sig_add).sort_index(level=[0,1])
    
    
    return df_sig_add




def get_interactions(df_features, c1, all_signals_names):
    
    '''
    Generates one-way interaction terms between one variable and all other variables 
    if the interaction terms does not already exist.
    Note: We could have generated all pair-wise interactions with one line of code
        But it results in a gigantic dataframe that may not fit the ram in some machines
    '''
    
    cols = df_features.columns.drop(c1)
    df = df_features[c1]
    
    # initialize list for saving interactions with the original signal as first element
    df_sig = [df.to_frame(c1)]
    
    # get interactions that don't already exist
    for c2 in cols:
        if f'{c1}__{c2}' not in all_signals_names and f'{c2}__{c1}' not in all_signals_names:
            df1 = df * df_features[c2]
            df_sig.append(df1.to_frame(f'{c1}__{c2}'))
    
    df_sig = pd.concat(df_sig, axis=1)
    return df_sig




def get_signal_long_short_ret(signals_t, ret_t, t):
    
    '''
    Get long short returns on day t. Long is top 10%. Short is bottom 10%.
    '''
    
    # get top 10%
    top_threshold = signals_t['signal'].quantile(0.9)
    top = signals_t[signals_t['signal'] > top_threshold].copy()
    top['decile'] = 10

    # get bottom 10%
    bottom_threshold = signals_t['signal'].quantile(0.1)
    bottom = signals_t[signals_t['signal'] < bottom_threshold].copy()
    bottom['decile'] = 1
    temp = pd.concat([bottom, top])

    # merge with next month return
    temp = temp.merge(ret_t, how='inner', on='permno')
    temp = temp.dropna(subset=['w', 'ret'])
    temp['nstock'] = 1
    
    
    # normalize weights to sum to 1
    temp['w'] = temp['w']/temp.groupby('decile')['w'].transform('sum')
    temp['ret_vw'] = temp.eval('ret * w')
    
    # compute value-weighted and equal-weighted portfolio returns
    temp = temp.groupby('decile').agg({'ret': 'mean', 'ret_vw': 'sum', 'nstock': 'sum'})
    
    # now get long short for value weighted and equal-weighted
    nshort, nlong = temp['nstock'].loc[[1, 10]].values
    signal_long_short = [temp.at[10, r] - temp.at[1, r] for r in ['ret', 'ret_vw']]
    signal_long_short = [*signal_long_short, nshort, nlong, t]

    return signal_long_short




def get_long_short_all_periods(df, v, dates, df_crspm, df_mktval):
    
    '''
    Get long-short returns for all periods for signal v.
    '''
    
    # unstack signal series so permnos are in columns and dates are the index
    df = df.unstack(0)
    
    
    df_signal_ret = []
    for t in dates:
        
        signals_t = df.loc[t].dropna().to_frame('signal')
        value_weight_t = df_mktval.loc[t].dropna()
        ret_t = df_crspm.loc[t].to_frame('ret')
        
        # drop stocks with price less than 1 and merge with signal 
        value_weight_t = value_weight_t.loc[value_weight_t['prc']>=1, ['w']]
        signals_t = signals_t.merge(value_weight_t, how='inner', on='permno')

    
        try:
            # fails if top and bottom 10% are not well-defined
            long_short_t = get_signal_long_short_ret(signals_t, ret_t, t)
        except:
            continue
        
        df_signal_ret.append(long_short_t)
    
    col_names = ['ret_ew', 'ret_vw', 'nshort', 'nlong', 'datem']
    df_signal_ret = pd.DataFrame(df_signal_ret, columns=col_names)
    df_signal_ret['signal_name'] = v
    
    return df_signal_ret


#%% load and prepare returns data

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

# take relevant period
df_crspm = df_crspm.query('date.dt.year>=1996')

# take market value and price and lag them
cols_use = ['permno', 'date', 'w', 'prc']
df_mktval = df_crspm[cols_use].copy()
df_mktval['date'] = df_mktval['date'] + pd.DateOffset(months=1)

# convert date to integer for faster querying
df_mktval['date'] = df_mktval['date'].dt.year * 100 + df_mktval['date'].dt.month
df_mktval = df_mktval.set_index(['date', 'permno']).sort_index(level=[0,1])



# take returns and unstack
cols_use = ['permno', 'date', 'ret']
df_crspm = df_crspm[cols_use].set_index(['permno', 'date']).sort_index(level=[0,1])
df_crspm['ret'] = df_crspm['ret'] / 100 # contert returns back to decimal instead of percent
df_crspm = df_crspm['ret'].unstack(0).sort_index()

# save original date then convert date to integer for faster querying
crspm_dates = df_crspm.index
df_crspm.index = df_crspm.index.year * 100 + df_crspm.index.month


#%% read ravenpack ess, rescale data to (0, 1) and aggregate ess to monthly 


df_rp_agg = pd.read_parquet(path_output / 'RavenPack_ess.parquet')
df_rp_agg = df_rp_agg.drop(columns=['entity_name'])
df_rp_agg['datem'] = df_rp_agg['timestamp_utc'].dt.year * 100 + df_rp_agg['timestamp_utc'].dt.month 
df_rp_agg['n_news'] = df_rp_agg['ess'].notnull()


# first rescale data
scaler = MinMaxScaler(feature_range=(0.0001, 1))
df_rp_agg[['ess']] = scaler.fit_transform(df_rp_agg[['ess']])

# aggregate to monthly
df_rp_agg = df_rp_agg.groupby(['datem', 'permno', 'group']).agg({'ess': 'mean', 'n_news': 'sum'})
n_stock_per_nws_grp = df_rp_agg.groupby(['datem', 'group'])['n_news'].apply(lambda x: (x != 0).sum())


# take news groups with enough observaions
n_stock_per_nws_grp_ave = n_stock_per_nws_grp.groupby('group').mean()
group_ls = n_stock_per_nws_grp_ave[n_stock_per_nws_grp_ave>=MIN_AVE_STOCKS].index
print(len(group_ls))

# take only  news groups with enough stock coverage
df_rp_agg = df_rp_agg.reset_index().query('group in @group_ls').set_index(['datem', 'permno', 'group'])


# get full dates for each permno
df_rp_agg = df_rp_agg['ess'].unstack(0).stack(dropna=False).to_frame('ess')
# then put the news variables on the columns
df_rp_agg = df_rp_agg.reset_index().pivot(index=['permno', 'datem'], columns=['group'], values='ess').sort_index(level=[0,1])


# map news group names to short names
name_map = df_rp_agg.columns
name_map = {v: f'v{i+1}' for i, v in enumerate(name_map)}
df_rp_agg = df_rp_agg.rename(columns=name_map)

# fill mising values with lag value
# df_rp_agg = df_rp_agg.fillna(0.5) 
df_rp_agg = df_rp_agg.groupby('permno').transform(lambda x: x.fillna(method='ffill', limit=12))

# fill the remaining missing values with median in the month
df_rp_agg = df_rp_agg.groupby('datem').transform(lambda x: x.fillna(x.median()))

# fill the remaining missing values with neutral value of 0.5
df_rp_agg = df_rp_agg.fillna(0.5) 



#%%  Generate initial set of additional features based on change relative to some threshold

with Parallel(n_jobs=10, verbose=3) as parallel:
    
    df_features = parallel(delayed(get_diff_based_features)(df_rp_agg[[v]], v)  for v in df_rp_agg.columns)

df_features = pd.concat(df_features, axis=1)

# take data starting from 06/2000
df_features = df_features.loc[idx[:,200006:], :]

del df_rp_agg


#%% expand features through interactions and get portfolios

dates = df_crspm.index[df_crspm.index>=200101]

func_long_short = partial(get_long_short_all_periods, dates=dates, df_crspm=df_crspm, 
                          df_mktval=df_mktval)



df_signal_ret = []
all_signals_names = []
with Parallel(n_jobs=14, verbose=3) as parallel:
    
    start = time.time()
    # for each variables in the features 
    # if the interaction does not already exist. 
    for n, c1 in enumerate(df_features.columns):
        print(f'\n{n+1} {c1}\n')
        
        # get interaction with all other variables if interactions does not already exist
        # returns original variable and interactions
        df_sig = get_interactions(df_features, c1, all_signals_names)
        all_signals_names +=  list(df_sig.columns)
    
        # lag signals by one month
        df_sig = df_sig.groupby('permno').shift()

        # get long short portfolios in parrallel            
        temp_ret = parallel(delayed(func_long_short)(df_sig[v], v)  for v in df_sig.columns)
        
        temp_ret = pd.concat(temp_ret)
        df_signal_ret.append(temp_ret)

    end = time.time()
    print(f'Time taken: {(end-start)/3600} hrs')

df_signal_ret = pd.concat(df_signal_ret)

#%% final clean up and save

# merge original dates
crspm_dates = pd.DataFrame(crspm_dates)
crspm_dates['datem'] = crspm_dates['date'].dt.year * 100 + crspm_dates['date'].dt.month
df_signal_ret = df_signal_ret.merge(crspm_dates, how='left', on='datem')
df_signal_ret[['ret_ew', 'ret_vw']] = df_signal_ret[['ret_ew', 'ret_vw']] * 100




# drop redundant signals: those with correlation with another signal above a certain threshold
thresh = 0.996 # the highest correlation among signals in past returns data is 0.996
df = df_signal_ret.pivot(index='date', columns='signal_name', values='ret_ew').sort_index()
# get correlation matrix and take its upper triangular matrix
corr_mat = df.corr()
mask = np.triu(np.ones(corr_mat.shape), k=1)
corr_mat = corr_mat.where(mask==1)
# get correlations above thresh
high_corr = corr_mat[np.abs(corr_mat).round(3) > thresh].stack().to_frame('corr')
high_corr.index.names = ['sig_1', 'sig_2']
high_corr = high_corr.reset_index()
# remove redundant signals from full list of signals
signals_use = df.columns
for v1, v2 in tqdm(high_corr[['sig_1', 'sig_2']].values):
    if v2 in signals_use:
        signals_use = signals_use.drop(v2)
print(f'\n# remaining signals:  {len(signals_use)}')
# take non-redundant signal portfolios
df_signal_ret_use = df_signal_ret.query('signal_name in @signals_use').copy()
df_signal_ret_use['signalid'], _ = pd.factorize(df_signal_ret_use['signal_name'])



# reorder columns
cols_order = ['signalid', 'signal_name', 'date', 'ret_ew', 'ret_vw', 'nshort', 'nlong',]
df_signal_ret_use = df_signal_ret_use[cols_order]


# base signal names
df_base_sig_name = pd.Series(name_map).reset_index()
df_base_sig_name.columns = ['news_group', 'signal_name']


df_signal_ret_use.to_csv(path_output / 'RavenPackSignalsLongShort.csv.gzip', index=False)
df_base_sig_name.to_csv(path_output / 'RavenPackBaseSignalNames.csv', index=False)


#%%




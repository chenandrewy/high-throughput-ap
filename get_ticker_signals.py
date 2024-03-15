# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:24:31 2024

@author: cdim
"""

# %% import libraries

import numpy as np
import pandas as pd
import time, re, gc, os
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count
import warnings
import pyreadr
from itertools import combinations
import inspect



warnings.simplefilter(action='ignore', category=FutureWarning)
tqdm.pandas()
tqdm.monitor_interval = 0

if cpu_count()>30:
    CPUUsed = 20
else:
    CPUUsed = cpu_count()-1

# Get the directory of the current file
current_filename = inspect.getframeinfo(inspect.currentframe()).filename
current_directory = os.path.dirname(current_filename)
path_root = Path(current_directory + "/../../")

# directory to reading and saving
path_input = path_root / 'Data/'
path_output = path_root / 'Data/'



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

# convert dates from month end to begining of month
df_crspm['date'] = df_crspm['date'] - pd.tseries.offsets.MonthBegin(1)


# take market value and price
cols_use = ['permno', 'date', 'w', 'prc', 'ret', 'ticker']
df_crspm = df_crspm[cols_use].reset_index(drop=True)



#%% construct ticker strategies

'''
Idea is to rank stocks based on the first letter of their ticker then split
the ranked stocks into 20 groups. Then take combination 20 choose 4.
For each set of four groups, Long the stocks in the first two groups and short
stocks in the other 2 groups. 
Repeat for the above process for the second, third and fourth letters of the 
tickers respectively
'''


def get_combo_portfolio(df_t, t, Ngroups, combos):
    
    def _worker(df, tup, lt, t):
        
        # create long & short portfolio based on the elements in tup      
        
        lng_ls = tup[:2] # the group ids for stocks we long
        shrt_ls = tup[2:] # the group ids for stocks we short
        long_short = {}
        for nm, ls in [('long', lng_ls), ('short', shrt_ls)]:
            df_cmb = df.loc[df['group'].isin(ls), ['ret', 'w']].values
            r, w = df_cmb[:,0], df_cmb[:,1]
            ret_ew = r.mean()
            ret_vw = r @ (w/w.sum())
            long_short[f'{nm}_ew'] = ret_ew
            long_short[f'{nm}_vw'] = ret_vw
            long_short[f'n{nm}'] = len(r)
            
        long_short['date'] = t
        # construct signal name and include in result
        lng_str = f'{lng_ls[0]}_{lng_ls[1]}'
        shrt_str = f'{shrt_ls[0]}_{shrt_ls[1]}'
        long_short['signalname'] = f'{lt}_lng_{lng_str}_sht_{shrt_str}'
        
        return long_short
        
    
    # Loop over ticker letters and use _worker func to get long & short port
    long_short_all = []
    for lt in ['L1', 'L2', 'L3', 'L4']:
        
        # sort data by the ticker letter
        df = df_t.dropna(subset=[lt]).sort_values(lt)
        
        # to avoid error we require at least as many stocks as number of groups
        # we want to create
        if len(df)>=Ngroups:
            
            # create a group indictor for the data; total groups = Ngroups
            group_indicator = np.array([i % Ngroups for i in range(len(df))]) + 1
            group_indicator.sort() # sort the group indicator
            df['group'] = group_indicator
            
            # use the worker function to get the combo portfolios 
            res = [_worker(df, tup, lt, t) for tup in combos]
            long_short_all.append(res)
    
    # unpack list of lists
    long_short_all = [ls for sub_ls in long_short_all for ls in sub_ls]
    return long_short_all

    
    
        

# split ticker into consecutive letters
df_tk = df_crspm.dropna(subset=['ticker']).drop(columns=['ret'])
df = pd.DataFrame(df_tk['ticker'].apply(lambda x: list(x)).values.tolist(),
                  index=df_tk.index)
df.columns = [f'L{i+1}' for i in range(len(df.columns))]
df = df.replace({None: np.nan})
df_tk = df_tk.join(df)

# lag tickers and market size and then merge with returns
df_tk['date'] = df_tk['date'] + pd.DateOffset(months=1)
df_tk = df_tk.merge(df_crspm[['permno', 'date', 'ret']], 
                    how='inner', on=['permno', 'date'])

# drop missing returns or market value and select relevant period
df_tk = df_tk.dropna(subset=['ret', 'w'])
# df_tk = df_tk.query('date.dt.year >= 1963 & prc>=1')
df_tk = df_tk.query('date.dt.year >= 1963')


# get portfolios in parrallel; it takes about 25 mins when run with 15 cpus
Ngroups = 20
combos = list(combinations(np.arange(1, Ngroups+1), 4))
with Parallel(n_jobs=CPUUsed, verbose=5) as parallel:
    df_long_short = parallel(delayed(get_combo_portfolio)(df_t, t, Ngroups, combos) 
                             for t, df_t in df_tk.groupby('date'))


# unpack list of lists and convert to dataframe
df_long_short = [ls for sub_ls in df_long_short for ls in sub_ls]
df_long_short = pd.DataFrame(df_long_short)

# get long-short
df_long_short['ret_vw'] = df_long_short.eval('long_vw - short_vw')
df_long_short['ret_ew'] = df_long_short.eval('long_ew - short_ew')
cols = ['long_ew', 'long_vw', 'short_ew', 'short_vw']
df_long_short = df_long_short.drop(columns=cols)

# final cleanup
df_long_short['signalid'], _ = pd.factorize(df_long_short['signalname'])
df_signal_names = df_long_short[['signalid', 'signalname']].drop_duplicates(
    subset=['signalid'])
cols_order = ['signalid', 'date', 'ret_ew', 'ret_vw', 'nlong', 'nshort']
df_long_short = df_long_short[cols_order]


#%% save data

df_long_short.to_csv(path_output / 'TickerSignalsLongShort.csv.gzip', index=False)
df_signal_names.to_csv(path_output / 'TickerSignalNames.csv.gzip', index=False)


#%%


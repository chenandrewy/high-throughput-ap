# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 17:34:48 2023

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
    path_input = Path('C:/users/cdim/Dropbox/ChenDim/Stop-Worrying/Andrew/Data/')
    path_output = Path('C:/users/cdim/Dropbox/ChenDim/Stop-Worrying/Chuks/Data/')
    path_figs = Path('C:/users/cdim/Dropbox/ChenDim/Stop-Worrying/Chuks/Figures/')
    path_tables = Path('C:/users/cdim/Dropbox/ChenDim/Stop-Worrying/Chuks/Tables/')
elif MachineUsed==2: # Andrew 

    CPUUsed = 8

    # get working directory
    path_root = Path(os.getcwd() + '/../../')

    path_input = Path(path_root / 'Andrew/Data/')
    path_output = Path(path_root / 'Andrew/Data/')
    path_figs = Path(path_root / 'Andrew/Figures/')
    path_tables = Path(path_root / 'Andrew/Tables/')    



#%%


# load crsp monthly returns from r-data format
df_crspm = pyreadr.read_r(path_input / 'CLZ/crspm.RData')
# take the dataframe containing the data. 
# it's first element of the dictionary values
df_crspm, = df_crspm.values() 

df_crspm['permno'] = df_crspm['permno'].astype(int)
df_crspm['date']  = pd.to_datetime(df_crspm['date'])
df_crspm['prc'] = df_crspm['prc'].abs()
df_crspm['mktval'] = df_crspm.eval('prc * shrout')

# apply filters for NYSE/NASDAQ/AMEX and stocks prices above 1
df_crspm = df_crspm.query('exchcd in [1, 2, 3] & shrcd in [10, 11]')
permos_use = df_crspm['permno'].unique()
del df_crspm




#%% down ravenpack news data


db = wrds.Connection(wrds_username='ccdim')



# Get RavenPack ID to Permno Mapping
query =  '''SELECT DISTINCT a.permno, b.rp_entity_id 
    from crsp.dse as a, 
    rpna.wrds_company_names as b
    where a.ncusip=substr(b.isin, 3, 8)
'''
df_rp_id_permno = db.raw_sql(query).sort_values(['permno'])
df_rp_id_permno = df_rp_id_permno.drop_duplicates()
df_rp_id_permno['permno'] = df_rp_id_permno['permno'].astype(int)
entity_ls = tuple(df_rp_id_permno.query('permno.isin(@permos_use)')['rp_entity_id'].unique())



# download news sentiment by year
df_rp = []
# 2000 is the start date of ravenpack
for year in tqdm(range(2000, 2023)):

    query = '''
        SELECT DISTINCT a.rp_entity_id, a.timestamp_utc, a.entity_name, a.event_sentiment_score, a.relevance,
        a.group, a.type, a.category
        
        from rpna.{product}_equities_{year} as a
        
        WHERE rp_entity_id IN {entity_ls} 
        AND relevance >= 70
        /* AND event_similarity_days >= 90 */
        /* AND country_code IN ('US') */
        order by rp_entity_id, timestamp_utc
    '''
        
    df = db.raw_sql(query.format(product='rpa_djpr', year=year, entity_ls=entity_ls)) # down jones and PR
    
    df_rp.append(df)

df_rp = pd.concat(df_rp)


df_rp = df_rp.merge(df_rp_id_permno, how='left', on='rp_entity_id').dropna(subset=['permno'])
df_rp['permno'] = df_rp['permno'].astype(int)
df_rp = df_rp.rename(columns={'event_sentiment_score': 'ess'})
df_rp['obs_id'] = np.arange(1, len(df_rp)+1)
df_rp = df_rp.set_index('obs_id')
df_rp = df_rp.drop(columns=['rp_entity_id'])


df_rp.to_parquet(path_output / 'RavenPack_ess.parquet')

#%%


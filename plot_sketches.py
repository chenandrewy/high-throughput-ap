# 2023 10 Andrew: testing ways of visualizing rolling OOS and shrinkage estimates
# to do: 
#   use abs(tstat), add sign information to oos returns
#   split sample at 2004


#%% environment ---------------------------------------------------------------

import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from pathlib import Path

MachineUsed = 2

if MachineUsed==1: # Chuks 
    path_input = Path('C:/users/cdim/Dropbox/ChenDim/Stop-Worrying/Andrew/Data/')
    path_output = Path('C:/users/cdim/Dropbox/ChenDim/Stop-Worrying/Andrew/Data/')
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

# Select rolling t-stat / oos file
rollsignal_file = 'OOS_signal_tstat_OosNyears1.csv.gzip'

#%% Load data ----------------------------------------------------------------

# load up signal level data 
#   not sure this is compatible with Chuks machine...
rollsignal0 = pd.read_csv(str(path_output) + '/' + rollsignal_file)
rollsignal0.rename(columns={'mean_ret': 'ret'}, inplace=True)
rollsignal0['s_flag'] = rollsignal0['s_flag'].str.lower()

# reshape 
rollsignal1 = rollsignal0.pivot(index=['signal_family', 'oos_begin_year', 'signalid'],\
                columns='s_flag', values=['ret','tstat'])
rollsignal1.columns = ['_'.join(col) for col in rollsignal1.columns.values]
rollsignal1.reset_index(inplace=True)

# clean family names
def clean_names(x):
    if x == 'DataMinedLongShortReturnsEW': return 'acct_ew'
    elif x == 'DataMinedLongShortReturnsVW': return 'acct_vw'
    elif x == 'ticker_Harvey2017JF_ew': return 'ticker_ew'
    elif x == 'ticker_Harvey2017JF_vw': return 'ticker_vw'
        
rollsignal1['signal_family'] = rollsignal1['signal_family'].apply(clean_names)

#%% Estimate shrinkage --------------------------------------------------------

# estimate shrinkage each year
def get_shrinkage(tstatlist):
    return 1/max(tstatlist.var(),1)

rollfamily = rollsignal1\
    .groupby(['signal_family', 'oos_begin_year'])\
    .agg({'tstat_is': get_shrinkage})\
    .rename(columns={'tstat_is': 'shrinkage'}).reset_index()

# merge onto rollsignal and find predicted return
rollsignal2 = rollsignal1.merge(rollfamily, on=['signal_family', 'oos_begin_year'])\
    .assign(ret_pred = lambda x: x['ret_is']*(1-x['shrinkage']))

#%% Sort each year, take mean returns, then measure decay (plot) -------------
# replicates Chen Lopez Zim Table 4

# predictions work nicely before 2004 or so
# After 2004, there is too much decay, especially for acct vw
# this should work as a main figure if we make it look nice 
# right now it's understated compared to the box plots


# user input
n_groups = 20
mean_begin = 1983 # min 1993
mean_end = 2017 # max 2017
n_se_plot = 2 # for errorbars
y_lim_all = [-1,1]

# sort
rollsignal2['is_group'] = rollsignal2.groupby(['signal_family', 'oos_begin_year'])\
    ['ret_is'].transform(lambda x: pd.qcut(x, n_groups, labels=np.arange(1,n_groups+1)\
                                 , duplicates='drop'))

# aggregate to bin, year
roll_is_group = rollsignal2.groupby(['signal_family', 'oos_begin_year', 'is_group'])\
    .agg({'ret_is': 'mean', 'ret_pred': 'mean', 'ret_oos': 'mean'}).reset_index()

# agg to bin
sum_is_group = roll_is_group[(roll_is_group['oos_begin_year'] >= mean_begin) \
                            & (roll_is_group['oos_begin_year'] <= mean_end)]\
    .groupby(['signal_family','is_group'])\
    .agg({'ret_is': 'mean', 'ret_pred': 'mean', 'ret_oos': 'mean'}).reset_index()

# find standard error for ret_oos 
#   we should move to non-overlapping observations
temp = roll_is_group[(roll_is_group['oos_begin_year'] >= mean_begin) \
                            & (roll_is_group['oos_begin_year'] <= mean_end)]\
    .groupby(['signal_family','is_group'])\
    .agg({'ret_oos': ['std','count']}).reset_index()

sum_is_group['ret_oos_se'] = n_se_plot*temp['ret_oos']['std'] / np.sqrt(temp['ret_oos']['count'])

# find decay by family [is there a cleaner way to do this?]
sum_is_group['decay_oos'] = 1-sum_is_group['ret_oos']/sum_is_group['ret_is']
sum_is_group['decay_pred'] = 1-sum_is_group['ret_pred']/sum_is_group['ret_is']
family_decay = sum_is_group[abs(sum_is_group['ret_is']) > 0.10]\
    .groupby(['signal_family'])\
    .agg({'decay_oos': 'mean', 'decay_pred': 'mean'})

# plot
plt.rcParams.update({'font.size': 16})
fig = plt.figure(figsize=(16,16))
family_list = sum_is_group['signal_family'].unique()
for i in range(len(family_list)):
    ax = fig.add_subplot(2,2,i+1)
    family_cur = family_list[i]
    # line at zero
    ax.axhline(y=0, color='black', alpha=0.7, linestyle='-', linewidth=1) 
    temp = sum_is_group.query("signal_family == @family_cur").copy()
    ax.plot(temp['is_group'], temp['ret_is'], color='grey', alpha=0.7)
    # ax.scatter(temp['is_group'], temp['ret_oos'], color='blue', alpha=0.7)
    ax.errorbar(temp['is_group'], temp['ret_oos'], yerr=temp['ret_oos_se'], fmt='o', color='blue', alpha=0.7)
    ax.plot(temp['is_group'], temp['ret_pred'], color='red', alpha=0.7)
    # label
    ax.set_title(family_cur + ' ' + str(mean_begin) + '-' + str(mean_end))
    ax.set_xlabel('is group')
    ax.set_ylabel('mean return')
    ax.legend(['zero','is', 'pred', 'oos'], loc='upper left')
    # add decay by family text
    decay_oos = family_decay.loc[family_cur, 'decay_oos']
    decay_pred = family_decay.loc[family_cur, 'decay_pred']
    ax.text(0.6, 0.1, 'decay oos = ' + str(round(decay_oos,2)) + '\n' + \
            'decay pred = ' + str(round(decay_pred,2)), transform=ax.transAxes
            , color = 'blue')
    # fix y limits
    ax.set_ylim(y_lim_all)
    ax.set_xlabel('in-sample return group')
    ax.set_ylabel('out-of-sample return')     

# save as pdf
plt.savefig(str(path_figs) + '/shrinkage_vs_oos.pdf', bbox_inches="tight")

#%% Measure decay by ols each year, take means, then plot
# follows Chen Velikov Figure 6

# estimate oos decay each year
def get_oos_decay(data):
    slope = smf.ols(formula = 'ret_oos ~ 0+ret_is', data=data).fit().params[0]
    decay = 1-slope
    return decay

tempsignal = rollsignal2.copy()

# winsorize ret_oos (doesn't make a difference)
winsor_p = 0
tempsignal['ret_oos'] = tempsignal.groupby(['signal_family', 'oos_begin_year'])\
    ['ret_oos'].transform(
        lambda x: x.clip(lower=x.quantile(winsor_p), upper=x.quantile(1-winsor_p)))

temp = tempsignal.groupby(['signal_family', 'oos_begin_year'])\
    .apply(get_oos_decay)\
    .reset_index(name='oos_decay')

# merge onto rollfamily
rollfamily2 = rollfamily.merge(temp, on=['signal_family', 'oos_begin_year'])

# plot
plt.rc('lines', linewidth=4.0)
fig = plt.figure(figsize=(16,16))
family_list = rollfamily2['signal_family'].unique()
for i in range(len(family_list)):
    ax = fig.add_subplot(2,2,i+1)
    family_cur = family_list[i]
    temp = rollfamily2.query("signal_family == @family_cur").copy()
    ax.plot(temp['oos_begin_year'], temp['shrinkage'], color='red', alpha=0.7)
    ax.plot(temp['oos_begin_year'], temp['oos_decay'], color='blue', alpha=0.7)
    # label
    ax.set_title(family_cur)
    ax.set_xlabel('oos begin year')
    ax.set_ylabel('shrinkage and oos decay')
    ax.legend(['shrinkage', 'oos decay'])
    # fix y limits
    ax.set_ylim([-1,2])
    # guide lines at 0 and 1
    ax.axhline(y=0, color='black', alpha=0.7, linestyle='--')
    ax.axhline(y=1, color='black', alpha=0.7, linestyle='--')


# save as pdf
plt.savefig(str(path_figs) + '/decay_by_year.pdf', bbox_inches="tight")

#%% Let the user know whats up

print('saved pdf figures to ' + str(path_figs))
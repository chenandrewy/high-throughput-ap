# 2023 10 Andrew: testing ways of visualizing rolling OOS and shrinkage estimates
# to do: 
#   nicer split sample at 2004 (use a function)
#   flip decay sign for intuitiveness?
#   cross sectional standard errors using montonicity
#   fdr bound estimates

#%% environment ---------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.formula.api as smf
from scipy.stats import norm
from pathlib import Path



plt.rcParams.update({'font.size': 12})


MachineUsed = 2

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


# Select rolling t-stat / oos file
rollsignal_file = 'OOS_signal_tstat_OosNyears1.csv.gzip'
rvpk_rollsignal_file = 'RvPk_OOS_signal_tstat_IsNyears5_OosNyears1.csv.gzip'

USE_SIGN_INFO = True
SIGNAL_FAMILY_NAME_MAP = {
    'DataMinedLongShortReturnsEW': 'acct_ew',
    'DataMinedLongShortReturnsVW': 'acct_vw',
    'ticker_Harvey2017JF_ew': 'ticker_ew',
    'ticker_Harvey2017JF_vw': 'ticker_vw',
    'PastReturnSignalsLongShort_ew': 'past_ret_ew',
    'PastReturnSignalsLongShort_vw': 'past_ret_vw',
    'RavenPackSignalsLongShort_ew': 'ravenpack_ew',
    'RavenPackSignalsLongShort_vw': 'ravenpack_vw'

    }
    


#%% Load data ----------------------------------------------------------------

# load up signal level data 
rollsignal0 = pd.read_csv(path_input / rollsignal_file)

# load ravenpack signal data: currently based on differently IS window
df_rvpk = pd.read_csv(path_output / rvpk_rollsignal_file)

rollsignal0 = pd.concat([rollsignal0, df_rvpk])


rollsignal0.rename(columns={'mean_ret': 'ret'}, inplace=True)
rollsignal0['s_flag'] = rollsignal0['s_flag'].str.lower()

# reshape 
rollsignal1 = rollsignal0.pivot(index=['signal_family', 'oos_begin_year', 'signalid'],
                columns='s_flag', values=['ret','tstat'])
rollsignal1.columns = ['_'.join(col) for col in rollsignal1.columns]
rollsignal1.reset_index(inplace=True)

# clean family names
rollsignal1['signal_family'] = rollsignal1['signal_family'].replace(SIGNAL_FAMILY_NAME_MAP)

#%% Fit earliest histogram of t-statistics -------------------------------------------------


# firstsignal has only the earliest year for each signal family
earliest_year = rollsignal1.groupby('signal_family')['oos_begin_year'].min()\
    .rename('earliest_year').reset_index()
firstsignal = rollsignal1.merge(earliest_year, on=['signal_family'])\
    .query('oos_begin_year == earliest_year')
firstsignal['sign_is'] = np.sign(firstsignal['ret_is'])
    
# fit these t-stats by family
def fit_tstat(tstatlist):
    return max( (tstatlist**2).mean(), 1)

firstsignal\
    .groupby(['signal_family'])\
    .agg({'tstat_is': fit_tstat})

firstsignal\
    .groupby(['signal_family','sign_is'])\
    .agg({'tstat_is': fit_tstat})


#%% 
# histograms
with PdfPages(path_figs / 'tstat_hist.pdf') as pdf: 
    for family_cur in firstsignal['signal_family'].unique():
        fig, ax = plt.subplots(figsize=(10, 7))
        temp = firstsignal.query("signal_family == @family_cur")
        ax.hist(temp['tstat_is'], bins=50, alpha=0.7, label='IS')
        ax.legend(loc='upper left')
        ax.set_title(f'{family_cur} {temp["earliest_year"].values[0]} \n')
        ax.set_xlabel('t-statistic')
        ax.set_ylabel('Frequency')
        pdf.savefig(bbox_inches="tight") 
        plt.show(); plt.close()




#%% Estimate shrinkage --------------------------------------------------------


# estimate shrinkage each year
def get_shrinkage(tstatlist):
    return 1/max(tstatlist.var(), 1)

rollfamily = rollsignal1\
    .groupby(['signal_family', 'oos_begin_year'])\
    .agg({'tstat_is': get_shrinkage})\
    .rename(columns={'tstat_is': 'shrinkage'}).reset_index()

# merge onto rollsignal and find predicted return
rollsignal2 = rollsignal1.merge(rollfamily, on=['signal_family', 'oos_begin_year'])
rollsignal2['ret_pred'] = rollsignal2.eval('ret_is * (1 - shrinkage)')


# apply in-sample sign information if requested
# clarify why we do this step
if USE_SIGN_INFO:
    rollsignal2['sign_is'] = np.sign(rollsignal2['ret_is'])
    cols = ['ret_is', 'ret_oos', 'tstat_is', 'tstat_oos', 'ret_pred']
    rollsignal2[cols] = rollsignal2[cols].values * rollsignal2[['sign_is']].values

#%% Sort each year, take mean returns, then measure decay (plot) -------------

# standard errors are currently overstated.  They don't account for
# the monotonicity of returns w.r.t. in-sample groups
# we should use some kind of regression line w/ confidence intervals

# user input
n_groups = 20
mean_begin = 1983 # really min year
mean_end = 2021 # max year
n_se_plot = 2 # for errorbars
y_lim_all = [-1,1]

# sort into is_group bins
rollsignaltemp = rollsignal2.copy()
rollsignaltemp['is_group'] = rollsignaltemp.groupby(['signal_family', 'oos_begin_year'])\
    ['ret_is'].transform(lambda x: pd.qcut(x, n_groups, labels=np.arange(1,n_groups+1)\
                                 , duplicates='drop'))

# aggregate to bin, year
roll_is_group = rollsignaltemp.groupby(['signal_family', 'oos_begin_year', 'is_group'])\
    .agg({'ret_is': 'mean', 'ret_pred': 'mean', 'ret_oos': 'mean'}).reset_index()

# agg to bin
sum_is_group = roll_is_group.query('@mean_begin <= oos_begin_year <= @mean_end')\
    .groupby(['signal_family', 'is_group'])\
    .agg({'ret_is': 'mean', 'ret_pred': 'mean', 'ret_oos': 'mean'})#.reset_index()

# find standard error for ret_oos 
#   we should move to non-overlapping observations
temp = roll_is_group.query('@mean_begin <= oos_begin_year <= @mean_end')\
    .groupby(['signal_family', 'is_group'])\
    .agg({'ret_oos': ['std', 'count']})#.reset_index()
    
temp = temp.loc[sum_is_group.index]
sum_is_group['ret_oos_se'] = n_se_plot*temp['ret_oos']['std'] / np.sqrt(temp['ret_oos']['count'])
sum_is_group = sum_is_group.reset_index()

# find decay by family [is there a cleaner way to do this?]
sum_is_group['decay_oos'] = 1 - sum_is_group['ret_oos']/sum_is_group['ret_is']
sum_is_group['decay_pred'] = 1 - sum_is_group['ret_pred']/sum_is_group['ret_is']
family_decay = sum_is_group[sum_is_group['ret_is'].abs() > 0.10]\
    .groupby(['signal_family'])\
    .agg({'decay_oos': 'mean', 'decay_pred': 'mean'})


# plot
family_list = sum_is_group['signal_family'].unique()
df = roll_is_group.dropna(subset=['ret_is']).query('@mean_begin <= oos_begin_year <= @mean_end')\
    .groupby('signal_family')[['oos_begin_year']].agg(
        min_yr = ('oos_begin_year', np.min),
        max_yr = ('oos_begin_year', np.max)
        )
with PdfPages(path_figs / 'shrinkage_vs_oos_cross.pdf') as pdf: 

    for family_cur in family_list:
        temp = sum_is_group.query("signal_family == @family_cur").set_index('is_group')
        min_yr, max_yr = df.loc[family_cur].values
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        ax.axhline(y=0, color='black', alpha=0.7, linestyle='-', linewidth=1) 
        ax.plot(temp.index, temp['ret_is'], color='grey', alpha=0.7, label='IS')
        ax.errorbar(temp.index, temp['ret_oos'], yerr=temp['ret_oos_se'], fmt='o', 
                    color='blue', alpha=0.7, label='OOS')
        ax.plot(temp.index, temp['ret_pred'], color='red', alpha=0.7, label='Pred.')
        ax.legend(loc='upper left')

        # add decay by family text
        decay_oos = family_decay.loc[family_cur, 'decay_oos']
        decay_pred = family_decay.loc[family_cur, 'decay_pred']            
        ax.text(0.5, 0.8, f'Decay OOS = {decay_oos:.2f} \nDecay pred. = {decay_pred:.2f}', 
                transform=ax.transAxes  , color='blue')

        # fix y limits and add labels
        ax.set_ylim(y_lim_all)
        ax.set_xticks(temp.index)
        
        # label
        ax.set_title(f'{family_cur} {min_yr} - {max_yr}\n')
        ax.set_xlabel('In-sample return group')
        ax.set_ylabel('Mean return (%)')  
        
        pdf.savefig(bbox_inches="tight") 
        plt.show(); plt.close()




#%% Measure decay by ols each year, take means, then plot
# follows Chen Velikov Figure 6

# estimate oos decay each year
def get_oos_decay(data):
    slope = smf.ols(formula = 'ret_oos ~ 0 + ret_is', data=data).fit().params[0]
    decay = 1-slope
    return decay

tempsignal = rollsignal2.copy()

# winsorize ret_oos (doesn't make a difference)
winsor_p = 0.01
tempsignal['ret_oos'] = tempsignal.groupby(['signal_family', 'oos_begin_year'])['ret_oos']\
    .transform(lambda x: x.clip(lower=x.quantile(winsor_p), upper=x.quantile(1-winsor_p)))
tempsignal['ret_is'] = tempsignal.groupby(['signal_family', 'oos_begin_year'])['ret_is']\
    .transform(lambda x: x.clip(lower=x.quantile(winsor_p), upper=x.quantile(1-winsor_p)))


temp = tempsignal.groupby(['signal_family', 'oos_begin_year'])\
    .apply(get_oos_decay)\
    .reset_index(name='oos_decay')

# merge onto rollfamily
rollfamily2 = rollfamily.merge(temp, on=['signal_family', 'oos_begin_year'])

# plot
family_list = rollfamily2['signal_family'].unique()
n_ave = 5
with PdfPages(path_figs / 'shrinkage_vs_decay_ts.pdf') as pdf: 
    for family_cur in family_list:
                
        temp = rollfamily2.query("signal_family == @family_cur")\
            .set_index('oos_begin_year').sort_index().drop(columns=['signal_family'])
         
        # take moving average
        temp_ma = temp.rolling(n_ave).mean()
            
        for df, sfx in [(temp, ''), (temp_ma, 'ma')]:
            if sfx == '':
                tt_sfx = sfx
            else:
                tt_sfx = f'{n_ave} year {sfx}'
            
            fig, ax = plt.subplots(figsize=(10, 7))
    
            ax.plot(df.index, df['shrinkage'], color='red', alpha=0.7, 
                    linewidth=4.0, label='Shrinkage')
            ax.plot(df.index, df['oos_decay'], color='blue', alpha=0.7, 
                    linewidth=4.0, label='OOS decay')
            ax.scatter(df.index, df['oos_decay'], color='blue', alpha=0.7)
            
            # label
            
            ax.set_title(f'{family_cur} {tt_sfx}\n')
            ax.set_xlabel('OOS begin year')
            ax.set_ylabel('Shrinkage and OOS decay')
            ax.legend(loc='upper left')
            
            # fix y limits
            ax.set_ylim([-1,2])
            
            # guide lines at 0 and 1
            ax.axhline(y=0, color='black', alpha=0.7, linestyle='--')
            ax.axhline(y=1, color='black', alpha=0.7, linestyle='--')    
            
            pdf.savefig(bbox_inches="tight") 
            plt.show(); plt.close()


#%% Alternative to Measure decay by ols for group then plot


# year_split = 2001 # for two sub-sample split

# follows Chen Velikov Figure 6

def get_oos_decay_v2(data):
    
    mod = smf.ols(formula = 'ret_oos_is ~ 0 + neg_ret_is', data=data).fit(
        cov_type='cluster', cov_kwds={'groups': data[['signalid', 'oos_begin_year']]})
    
    # note, the slope from the above regression is the same as doing
    # slope = smf.ols(formula = 'ret_oos ~ 0 + ret_is', data=data).fit().params[0]
    # decay = 1 - slope
    # but the above allows us to compute std error and confidence interval directly
    
    decay =[mod.params[0], *mod.conf_int()[0]]
    return decay


def get_ave_shrinkage(data):
    mod = smf.ols(formula = 'shrinkage ~ 1', data=data).fit(
        cov_type='HAC', cov_kwds={'maxlags': 2})
    return [mod.params[0], *mod.conf_int()[0]]



tempsignal = rollsignal2.copy()

# winsorize ret_oos (doesn't make a difference)
winsor_p = 0.01
tempsignal['ret_oos'] = tempsignal.groupby(['signal_family', 'oos_begin_year'])['ret_oos']\
    .transform(lambda x: x.clip(lower=x.quantile(winsor_p), upper=x.quantile(1-winsor_p)))
tempsignal['ret_is'] = tempsignal.groupby(['signal_family', 'oos_begin_year'])['ret_is']\
    .transform(lambda x: x.clip(lower=x.quantile(winsor_p), upper=x.quantile(1-winsor_p)))


# tempsignal['sample'] = np.where(tempsignal['oos_begin_year']<year_split, 
#                                 f'Pre-{year_split}', f'Post-{year_split}')

# compute decay and it's standard error as 
# 1 - slope from the regression ret_oos on ret_is
tempsignal['ret_oos_is'] = tempsignal.eval('ret_oos - ret_is')
tempsignal['neg_ret_is'] = tempsignal.eval('-1 * ret_is')
temp = tempsignal.groupby(['signal_family']).apply(get_oos_decay_v2)
temp = pd.DataFrame(temp.values.tolist(), columns=['oos_decay', 'decay_lci'], 
                    index=temp.index).reset_index()

# merge onto rollfamily
rollfamily2 = rollfamily.sort_values(['signal_family', 'oos_begin_year'])
# rollfamily2['sample'] = np.where(rollfamily2['oos_begin_year']<year_split, 
#                                 f'Pre-{year_split}', f'Post-{year_split}')
rollfamily2 = rollfamily2.groupby(['signal_family']).apply(get_ave_shrinkage)
rollfamily2 = pd.DataFrame(rollfamily2.values.tolist(), columns=['shrinkage', 'shrinkage_lci'], 
                    index=rollfamily2.index).reset_index()

# get confidence interval gap for ploting
rollfamily2 = rollfamily2.merge(temp, on=['signal_family']).set_index('signal_family')
rollfamily2['shrinkage_ci'] = rollfamily2.eval('shrinkage - shrinkage_lci')
rollfamily2['decay_ci'] = rollfamily2.eval('oos_decay - decay_lci')
rollfamily2 = rollfamily2.rename(columns={'shrinkage': 'Shrinkage', 'oos_decay': 'OOS Decay'})


family_long_name = {'acct': 'Fundamental',
                    'past_ret': 'Past return',
                    'ticker': 'Ticker',
                    'ravenpack': 'News media'
                    }


with PdfPages(path_figs / 'shrinkage_vs_decay_pooled.pdf') as pdf: 
    for sfx in ['ew', 'vw']:
            
        family_list = [v for v in rollfamily2.index if v.endswith(sfx)]
    
        df = rollfamily2.loc[family_list]
        df.index = df.index.str.rstrip(f'_{sfx}')
        df = df.rename(family_long_name)
        
        cols = ['Shrinkage', 'OOS Decay']
        vals = df[cols]
        ci_cols = ['shrinkage_ci', 'decay_ci']
        ci_vals = df[ci_cols].rename(
            columns={tup[0]: tup[1] for tup in zip(ci_cols, cols)})
        
        ax = vals.plot.bar(figsize=(10, 7), color=['b', 'r'], alpha=0.7, yerr=ci_vals,
                           capsize=3)
        plt.xticks(rotation=360)
        plt.xlabel('')
        plt.legend(loc='upper left')
        plt.title(f'portfolio type: {sfx} \n')
    
        pdf.savefig(bbox_inches="tight") 
        plt.show(); plt.close()



#%% FDR bound vs oos cross plots =======================================================
# there may be issues with time-series aggregation of probabilities here


# user input
n_groups = 20
mean_begin = 1983 # really min year
mean_end = 2021 # max year
n_se_plot = 2 # for errorbars
y_lim_all = [-1,1]
t_oos_h = 0 # threshold for oos t-stat test
leglab_oos = 'Pr(next year ret < 0)' # oos label (make sure to match t_oos_h + oos length)

rollsignaltemp = rollsignal1.copy()

# apply in-sample sign information if requested
if USE_SIGN_INFO:
    rollsignaltemp['sign_is'] = np.sign(rollsignaltemp['ret_is'])
    for col in ['ret_is', 'ret_oos', 'tstat_is', 'tstat_oos']:
        rollsignaltemp[col] = rollsignaltemp[col]*rollsignaltemp['sign_is']

# find order stats for each family, year
percentiles = np.linspace(0,1,n_groups+1)
percentiles = percentiles[1:-1]

rollfamsum = rollsignaltemp.groupby(['signal_family', 'oos_begin_year'])\
    .apply(lambda x: x['tstat_is'].quantile(percentiles))\
    .reset_index()

# call these order stats t_max
tempname = ['t_max_' + str(i) for i in range(1,n_groups)]
rollfamsum.columns = ['signal_family', 'oos_begin_year'] + tempname
rollfamsum['t_max_' + str(n_groups)] = 100

# reshape to long
rollfamsum = rollfamsum.melt(id_vars=['signal_family', 'oos_begin_year'], \
                             value_vars=tempname + ['t_max_' + str(n_groups)])\
    .rename(columns={'variable': 'is_group', 'value': 'tstat_max'})
rollfamsum.sort_values(['signal_family', 'oos_begin_year', 'tstat_max']\
                          , ascending=[True, True, True], inplace=True)

rollfamsum['is_group'] = rollfamsum['is_group'].str.lstrip('t_max_').astype(int)

# find tstat min
rollfamsum['tstat_min'] = rollfamsum.groupby(['signal_family', 'oos_begin_year'])\
    ['tstat_max'].shift(1)
if USE_SIGN_INFO:
    rollfamsum['tstat_min'].fillna(0, inplace=True)
else:   
    rollfamsum['tstat_min'].fillna(-100, inplace=True)

# reorder tstat_max and tstat_min columns
rollfamsum = rollfamsum[['signal_family', 'oos_begin_year', 'is_group', \
                         'tstat_min', 'tstat_max']]
    
# apply normal cdf to get prob under null
if USE_SIGN_INFO:
    def cdf_null(x):
        return 2*norm.cdf(x)
else:
    def cdf_null(x):
        return norm.cdf(x)

rollfamsum['Pr_null'] = rollfamsum['tstat_max'].apply(cdf_null)\
        - rollfamsum['tstat_min'].apply(cdf_null)

# FDR upper bound (assumes the quantiles are found correctly)
rollfamsum['FDR_UB'] = rollfamsum['Pr_null'] / (1/n_groups)
rollfamsum['FDR_UB'] = rollfamsum['FDR_UB'].apply(lambda x: min(x, 1))

# compare with OOS stats (this is preliminary)
rollsignaltemp = rollsignal2.copy()
rollsignaltemp['is_group'] = rollsignaltemp.groupby(['signal_family', 'oos_begin_year'])\
    ['tstat_is'].transform(lambda x: pd.qcut(x, n_groups, labels=np.arange(1,n_groups+1)\
                                    , duplicates='drop'))

# function for measuring oos performance
def tempfun(x):
    return sum(x < t_oos_h)/len(x)

rollfamoos = rollsignaltemp.groupby(['signal_family', 'oos_begin_year', 'is_group'])\
    .agg({'tstat_oos': tempfun}).reset_index()
rollfamoos = rollfamoos.rename(columns={'tstat_oos': 'Pr_oos_lt_h'})

# merge onto rollfamsum
rollfamsum = rollfamsum.merge(rollfamoos, on=['signal_family', 'oos_begin_year', 'is_group'])

# aggregate over years
sum_is_group = rollfamsum.query('@mean_begin <= oos_begin_year <= @mean_end')\
    .groupby(['signal_family','is_group'])\
    .agg(
        tstat_min = ('tstat_min', np.mean)
        , tstat_max = ('tstat_max', np.mean)
        , Pr_null = ('Pr_null', np.mean)
        , FDR_UB = ('FDR_UB', np.mean)
        , SD_FDR_UB = ('FDR_UB', np.std)
        , Pr_oos_lt_h = ('Pr_oos_lt_h', np.mean)
        , SD_Pr_oos_lt_h = ('Pr_oos_lt_h', np.std)
        , nobs = ('Pr_oos_lt_h', 'count')
    )\
    .reset_index()

# plot
family_list = sum_is_group['signal_family'].unique()
df = rollfamsum.dropna(subset=['tstat_max']).query('@mean_begin <= oos_begin_year <= @mean_end')\
    .groupby('signal_family')[['oos_begin_year']].agg(
        min_yr = ('oos_begin_year', np.min),
        max_yr = ('oos_begin_year', np.max)
        )

with PdfPages(path_figs / 'fdr_vs_oos_cross.pdf') as pdf: 
    for family_cur in family_list:
        fig, ax = plt.subplots(figsize=(10, 7))
    
        temp = sum_is_group.query("signal_family == @family_cur").copy()   
        min_yr, max_yr = df.loc[family_cur].values
    
        # choose standard error 
        # (I suppose we could use instead sqrt(pq/n))
        temp['SE_oos'] = n_se_plot*temp['SD_Pr_oos_lt_h'] / np.sqrt(temp['nobs'])
        
        # main plots
        ax.plot(temp['is_group'], temp['FDR_UB'], color='red', alpha=0.7, label='FDR upper bound')
        ax.errorbar(temp['is_group'], temp['Pr_oos_lt_h'], yerr=temp['SE_oos'],
            fmt='o', color='blue', alpha=0.7, label=leglab_oos)
        
        # label
        ax.set_title(f'{family_cur} {mean_begin}-{mean_end} \n')
        ax.set_xlabel('In-sample t-stat group')
        ax.set_ylabel('Probability')
        # fix y limits
        ax.set_ylim([0, 1])
        # legend
        ax.legend() 
        # guide line at 0.5
        ax.axhline(y=0.5, color='black', alpha=0.7, linestyle='--')
        # y ticks
        ax.set_yticks(np.arange(0,1.1,0.1))
        ax.set_xticks(temp['is_group'].values)

        pdf.savefig(bbox_inches="tight") 
        plt.show(); plt.close()



#%% Let the user know whats up

print('saved pdf figures to ' + str(path_figs))

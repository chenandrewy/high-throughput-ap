# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 09:47:28 2024

@author: cdim
"""


import os, re, platform
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import norm
from pathlib import Path
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, ClusterWarning
from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
import inspect


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ClusterWarning)

tqdm.monitor_interval = 0
tqdm.pandas()
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({"font.size": 12})
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']



# Get the directory of the current file
current_filename = inspect.getframeinfo(inspect.currentframe()).filename
current_directory = os.path.dirname(current_filename)
path_root = Path(current_directory + "/../../")


# set directories for user
MachineUsed = platform.node()
if MachineUsed == 'GWSB-DUQ456-L30':  # Chuks
    CPUUsed = cpu_count()-1
    path_input = path_root / "Data/"
    path_output = path_root / "Data/"
    path_figs = path_root / "Chuks/Figures/"
    path_tables = path_root / "Chuks/Tables/"
else:  # Andrew
    CPUUsed = 8
    path_input = path_root / "Data/"
    path_output = path_root / "Data/"
    path_figs = path_root / "Andrew/Figures/"
    path_tables = path_root / "Andrew/Tables/"


# Select rolling t-stat / oos file
estimates_file = "ChuksDebug_Predict_SignalYear.csv.gzip"
rollsignal_file = "OOS_signal_tstat_OosNyears1.csv.gzip"



# USE_SIGN_INFO = True

SIGNAL_FAMILY_NAME_MAP = {
    "DataMinedLongShortReturnsEW": "acct_ew",
    "DataMinedLongShortReturnsVW": "acct_vw",
    "TickerSignalsLongShort_ew": "ticker_ew",
    "TickerSignalsLongShort_vw": "ticker_vw",
    "PastReturnSignalsLongShort_ew": "past_ret_ew",
    "PastReturnSignalsLongShort_vw": "past_ret_vw",
    "RavenPackSignalsLongShort_ew": "ravenpack_ew",
    "RavenPackSignalsLongShort_vw": "ravenpack_vw",
}
FAMILY_LONG_NAME = {
    'acct_ew': 'Accounting EW',
    'acct_vw': 'Accounting VW',
    'past_ret_ew': 'Past Return EW',
    'past_ret_vw':'Past Return VW',
    'ticker_ew':'Ticker EW',
    'ticker_vw': 'Ticker VW',
    'ravenpack_ew':'News Sentiment EW',
    'ravenpack_vw': 'News Sentiment VW'
    }


ret_freq_adj = 12 # annualize
YEAR_MIN = 1963
YEAR_MAX = 2020
families_use = ['acct_ew', 'acct_vw', 'past_ret_ew', 'past_ret_vw', 
                'ticker_ew', 'ticker_vw', 
                # 'ravenpack_ew', 'ravenpack_vw'
                ]




#%% load family strategies monthly returns data


signal_family_names = [
    # 'ticker_Harvey2017JF_ew.csv', 'ticker_Harvey2017JF_vw.csv', 
    'DataMinedLongShortReturnsVW.csv', 'DataMinedLongShortReturnsEW.csv',
    'PastReturnSignalsLongShort.csv.gzip',
    'TickerSignalsLongShort.csv.gzip',
    'RavenPackSignalsLongShort.csv.gzip'
                       ]

df_signal_ret = []
for name in tqdm(signal_family_names):
    df = pd.read_csv(path_input / f'{name}')
    label = re.sub(r'\.csv(?:\.gzip)?', '', name) # get file name without file extension

    if label in ['DataMinedLongShortReturnsVW', 'DataMinedLongShortReturnsEW']:
        df['day'] = 1
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        df['year'] = df['date'].dt.year
        df = df[['signalid', 'date', 'year', 'ret']].sort_values(['signalid', 'date'])
        df['signal_family'] = label
        df_signal_ret.append(df)
    
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
        
        # seperate value-weighted and equal-weighted portolio returns 
        # as different datasets
        df_ew = df[['signalid', 'date', 'year', 'ret_ew']].rename(
            columns={'ret_ew': 'ret'}).sort_values(['signalid', 'date'])
        df_ew['signal_family'] = f'{label}_ew'
        df_signal_ret.append(df_ew)
        
        df_vw = df[['signalid', 'date', 'year', 'ret_vw']].rename(
            columns={'ret_vw': 'ret'}).sort_values(['signalid', 'date'])
        df_vw['signal_family'] = f'{label}_vw'
        df_signal_ret.append(df_vw)

    
df_signal_ret = pd.concat(df_signal_ret)    
df_signal_ret = df_signal_ret.query('year>=@YEAR_MIN')
df_signal_ret["signal_family"] = df_signal_ret["signal_family"].replace(
    SIGNAL_FAMILY_NAME_MAP)


df_signal_ret['s_fid'] = df_signal_ret.groupby(['signal_family', 'signalid']).ngroup()
df_sfid = df_signal_ret[['signal_family', 'signalid', 's_fid']
                    ].drop_duplicates('s_fid')




#%% read annual estimates data and merge with signal future monthly returns

df_oos = df_signal_ret.rename(columns={'year': 'oos_begin_year'})

# load estimates
df_est = pd.read_csv(path_input / estimates_file)

df_est = df_est.rename(columns={'mean_ret': 'ret'}).drop(columns=['s_flag'])
df_est["signal_family"] = df_est["signal_family"].replace(
    SIGNAL_FAMILY_NAME_MAP)


# merge estimates and future returns
cols = ['signal_family', 'signalid', 'date', 'oos_begin_year', 'ret']
df_est_oos = df_est.merge(df_oos[cols], how='inner', suffixes=('_is', '_oos'),
                          on=['signal_family', 'signalid', 'oos_begin_year'])

df_est_oos = df_est_oos.rename(columns={'tstat': 'tstat_is'})

del df_oos

#%% summarize strategies data

sig_sumr = []
for fam, df in df_signal_ret.groupby('signal_family'):
    if fam in ['ravenpack_ew', 'ravenpack_vw']:
        continue
    df = df.query('@YEAR_MIN <= year <= @YEAR_MAX')
    smr = df.groupby('signalid')['ret'].mean() * ret_freq_adj
    smr = smr.describe(percentiles=[0.05, 0.5, 0.95]).round(1).to_frame(fam).T
    sig_sumr.append(smr)
    
sig_sumr = pd.concat(sig_sumr).drop(columns=['mean', 'std', 'min', 'max'])
sig_sumr['count'] = sig_sumr['count'].apply(lambda x: f'{x:,.0f}')
sig_sumr = sig_sumr.rename(columns={'count': '# strategies','5%': '5 pctile',    
                                    '50%': '50 pctile','95%': '95 pctile'})
print(sig_sumr)


# %% plot distribution of in-sample t-stats and model-implied distribution

'''currently not sure if the content of the estimates data is enough to plot 
the distribution of fitted t-stats
'''







#%% Plot IS, OOS and Prediced returns


ngroup = 20
oos_freq_adj = 1
year_set = [(1983, 2004), 
            (2004, 2020), 
            # (2020, 2022)
            ]

for year_min, year_max in year_set:
    temp = df_est_oos.query('@year_min<=oos_begin_year<=@year_max').copy()
    # yr_min, yr_max =  temp['oos_begin_year'].min(), temp['oos_begin_year'].max()
    
    cols = ['ret_is', 'ret_oos', 'pred_ret']
    temp[cols] = temp[cols] * ret_freq_adj
    
    temp['bin'] = temp.groupby(['signal_family', 'date'])['ret_is'].transform(
        lambda x: pd.qcut(x, ngroup, labels=np.arange(1, ngroup+1)))
    
    # summarize by family-oos_year-bin
    temp = temp.groupby(['signal_family', 'date', 'bin'], observed=True).agg(
        {'ret_is': 'mean',
        'ret_oos': 'mean',
        'pred_ret': 'mean'}
        )
    
    # get family-bin average and SD
    df = temp.groupby(['signal_family', 'bin'], observed=True).agg(
        {'ret_is': 'mean',
        'ret_oos': 'mean',
        'pred_ret': 'mean'}
        )
    
    df_se = temp.groupby(['signal_family', 'bin'], observed=True).agg(
        {'ret_oos': ['std', 'count']})
    df_se.columns = [c[1] for c in df_se.columns]
    df_se = (df_se['std']/np.sqrt(df_se['count']*oos_freq_adj)
             ).to_frame('se_ret_oos')
    df = df.merge(df_se, how='left', on=['signal_family', 'bin'])
    
    
    fig = plt.figure(figsize=(8,9))
    for i, family in enumerate(families_use):
        
        df1 = df.loc[family]
        ax = fig.add_subplot(3,2,i+1)
        
        ax.axhline(y=0, color='grey', alpha=0.7, linewidth=1) 
        ax.plot(df1.index, df1['ret_is'], color='grey', linestyle='--', alpha=0.7, 
                label='In-Samp')
        ax.errorbar(df1.index, df1['ret_oos'], yerr=df1['se_ret_oos']*1.96, 
                    fmt='o', markersize=5, color=COLORS[0], alpha=0.8, label='OOS')
        ax.plot(df1.index, df1['pred_ret'], color='red', alpha=0.8, 
                label='Predicted')
        ax.set_ylim(-12, 12)
        plt.title(f'{FAMILY_LONG_NAME[family]}', fontsize=12)
        

        if i == 0:
            plt.legend(loc='upper left')
        if i in [4, 5]:
            ax.set_xlabel('In-sample return group', fontsize=12)
        if i == 2:
            ax.set_ylabel('Long-Short Return (% ann)', fontsize=12)
        
    plt.subplots_adjust(hspace=0.3)
    plt.suptitle(f'Sample: {year_min}-{year_max}', y=0.95)
    # fig.savefig(path_figs / f'pred-vs-oos-withIS-{year_min}-{year_max}.pdf', format='pdf', 
    #             bbox_inches="tight")
    plt.show()
    


#%% Get best strategies based on predicted return, then compute OOS return

families_ew = [f for f in families_use if f.endswith('_ew') ]
families_vw = [f for f in families_use if f.endswith('_vw') ]

family_set = {
    '1 all': families_use, 
    # '2 acct only': ['acct_ew', 'acct_vw'], 
    # '3 acct ew only': ['acct_ew'], 
    # '4 pastret only': ['past_ret_ew', 'past_ret_vw'],
    # '5 ew only': families_ew,
    # '6 ew only': families_vw,
    }


extreme_ranks = [1, 5, 10]

df_best_ret = []
cols = ['date', 'pred_ret', 'ret_oos']
for label, family_ls in tqdm(family_set.items()):
    qry = 'signal_family in @family_ls & oos_begin_year<=@YEAR_MAX'
    df = df_est_oos.query(qry)[cols].copy()
    df['rank_pct'] = df.groupby('date')['pred_ret'].transform(
        lambda x: x.rank(pct=True)) * 100
    
    for rank in extreme_ranks:
        
        df['port'] = np.where(df['rank_pct']<=rank, 'short',
                              np.where(df['rank_pct'] >= 100-rank, 'long', 
                                       'neutral')
                              )

        df1 = df.groupby(['port', 'date'])['ret_oos'].agg(
            ['mean', 'std', 'count'])
        df2 = (df1.loc['long', 'mean'] - df1.loc['short', 'mean']
               ).to_frame('ret_oos')
        df2['nstrat'] = df1.loc['long', 'count'] + df1.loc['short', 'count']
        df2['pctmin'] = rank
        df2['family_grp'] = label
        
        df_best_ret.append(df2)
    
df_best_ret = pd.concat(df_best_ret).reset_index()   


extreme_ranks_plot = [1, 5, 10]

# plot cumulative performance of top n% strategies
df = df_best_ret.query("family_grp == '1 all' & pctmin in @extreme_ranks_plot")
df = df.pivot(columns=['pctmin'], index='date', values='ret_oos').sort_index()
df.columns = [f'{int(c)}%' for c in df.columns]
# remove percent before cumulation of returns
df = (1 + df/100).cumprod() 
fig, ax = plt.subplots(figsize=(7,5))
df.plot(ax=ax, style=['-', '--', '-.'])    
ax.set_ylabel('Value of $1 Invvested in 1983') 
ax.set_xlabel('') 
plt.legend(title='Using Strats in Extreme', alignment='left')
# fig.savefig(path_figs / 'beststrats-cret.pdf', format='pdf', bbox_inches="tight")
plt.show()   




#%% summarize best strategy

#### prepare data for summarizing best performing strategies

samp_split = 2004
samp_start = df_best_ret['date'].min().year
samp_end = df_best_ret['date'].max().year

# for comparison, read in CZ returns
pubret0 = pd.read_csv(path_input / 'PredictorLSretWide.csv')
pubdoc = pd.read_excel(path_input / 'PredictorSummary.xlsx')
pubdoc = pubdoc[['signalname', 'Year', 'SampleEndYear']].rename(
    columns={'Year':'pubyear', 'SampleEndYear':'sampend'})

# process CZ returns 
pubret = pd.melt(pubret0, id_vars='date', var_name='signalname', value_name='ret')
pubret = pubret.query('ret.notnull()')
pubret['date'] = pd.to_datetime(pubret['date'])
pubret['year'] = pubret['date'].dt.year
pubret = pubret.query('@samp_start <= year <= @samp_end')
pubret = pubret.merge(pubdoc, how='left', on='signalname')

pubcomb = []
for nm, max_yr in [('Pub Anytime', samp_end), ('Pub Pre-2004', samp_split)]:
    df1 = pubret.query('pubyear <= @max_yr')
    # df1 = df1.groupby(['year', 'signalname'])[['ret']].mean()
    df1 = df1.groupby('date').agg({'ret': ['mean', 'count']})
    df1.columns = [f'{c[0]}_{c[1]}' for c in df1.columns]
    df1 = df1.rename(columns={'ret_mean': 'ret', 'ret_count': 'nstrat'})
    df1['name'] = nm
    pubcomb.append(df1.reset_index())
pubcomb = pd.concat(pubcomb)


# merge CZ with best performing strategies
best_sumr = df_best_ret.query("family_grp == '1 all'").drop(columns=['family_grp'])
best_sumr['pctmin'] = best_sumr['pctmin'].replace(
    {1: 'DM Extreme 1\%', 5: 'DM Extreme 5\%', 10: 'DM Extreme 10\%',})
best_sumr = best_sumr.rename(columns={'pctmin': 'name', 'ret_oos':'ret'})


best_sumr = pd.concat([best_sumr, pubcomb])
best_sumr['year'] = best_sumr['date'].dt.year


# prepare table with best strategies summary
tabdat = []
sample_ls = [(1983, 2020), (1983, 2004), (2005, 2020)]
for min_yr, max_yr in sample_ls:
    df = best_sumr.query("@min_yr<=year<=@max_yr")
    df = df.groupby(["name"]).agg({
        'ret': ['mean', 'std', 'count'],
        'nstrat': 'mean'}
        )
    df.columns = [f'{c0}_{c1}' for c0, c1 in df.columns]
    df['sr'] = df.eval('ret_mean/ret_std') 
    df['tstat'] = df.eval('ret_mean/ret_std*sqrt(ret_count)')
    df = df.drop(columns=['ret_std', 'ret_count'])
    df = df[['nstrat_mean', 'ret_mean', 'tstat', 'sr']]
    df = df.loc[['DM Extreme 1\%', 
                 'DM Extreme 5\%', 'DM Extreme 10\%', 
                 'Pub Anytime', 'Pub Pre-2004']]
    line = pd.DataFrame(index=[f'{min_yr}-{max_yr}'], columns=df.columns)
    df = pd.concat([line, df])
    df.loc[''] = np.nan
    tabdat.append(df)
tabdat = pd.concat(tabdat).iloc[:-1]

# annualize mean return
tabdat['ret_mean'] = tabdat['ret_mean'] * ret_freq_adj
# annualize sharpe ratio
tabdat['sr'] = tabdat['sr'] * np.sqrt(ret_freq_adj)

# some clean up
cols = ['ret_mean', 'tstat', 'sr']
tabdat[cols] = tabdat[cols].applymap(lambda x: f'{x:.2f}')
tabdat['nstrat_mean'] = tabdat['nstrat_mean'].apply(lambda x: f'{x:.0f}')
tabdat = tabdat.replace({'nan': ''})
print(tabdat)


tabdat = tabdat.rename(
    columns={'nstrat_mean': '\makecell{Num Strats \\\\ Combined}', 
              'ret_mean': '\makecell{Mean Return \\\\ (\% ann)}', 
              'tstat': '$t$-stat', 'sr': '\makecell{Sharpe Ratio \\\\ (ann)}'})

# covert table to latex
col_format = 'l' + 'c' * tabdat.shape[1]
latex_table = tabdat.style.to_latex(column_format = col_format, hrules=True)
print(latex_table)

# clean latex table for saving and export it
part = r'\\\\\nPub Anytime'
repl = r'\\\\\n\\hline\nPub Anytime'
latex_tableF = re.sub(part, repl, latex_table)
part = r'\\\\\n &  &  &  &'
repl = r'\\\\\n\\hline\n &  &  &  &'
latex_tableF = re.sub(part, repl, latex_tableF)
part = r'(\n\d{4}\-\d{4} &  &  &  &  \\\\\n)'
repl =  r'\1\\hline\n'
latex_tableF = re.sub(part, repl, latex_tableF)
print(latex_tableF)

# with open(path_tables / 'beststrats.tex', 'w') as fh:
#     fh.write(latex_table)



#%% HLZ's preferred Benji-Yeki Theorem 1.3 control 

tempsum = df_est_oos.groupby(['signal_family', 'oos_begin_year']
                              )['signalid'].count().to_frame('Nstrat')
tempsum['BY1.3_penalty'] = tempsum['Nstrat'].apply(lambda x: sum(1/np.arange(1, x+1)))


df_FDR = df_est_oos.merge(tempsum, how='left', 
                                  on=['signal_family', 'oos_begin_year'])


df_FDR['tabs_is'] = df_FDR['tstat_is'].abs()
df_FDR['Pr_emp'] = df_FDR.groupby(['signal_family', 'oos_begin_year'])[
    'tabs_is'].transform(lambda x: x.rank(pct=True, ascending=False))
df_FDR['Pr_null'] = 2*norm.cdf(-df_FDR['tabs_is'].values)
df_FDR['FDRmax_BH'] = df_FDR.eval('Pr_null/Pr_emp')
df_FDR['FDRmax_BY1.3'] = df_FDR['FDRmax_BH']*df_FDR['BY1.3_penalty']


# find t-stat hurdles
crit_ls = np.array([1, 5, 10])/100
rollfamFDR = []
for crit in crit_ls:
    df = df_FDR[df_FDR['FDRmax_BY1.3'] <= crit] 
    df = df.groupby(['signal_family', 'oos_begin_year'])[
        'tabs_is'].min().to_frame('tabs_hurdle')
    df['crit_level'] = crit
    rollfamFDR.append(df)
rollfamFDR = pd.concat(rollfamFDR).reset_index()


# define hurdle as max(tstat_is)+1 if no signals pass
df = df_FDR.groupby(['signal_family', 'oos_begin_year'])[
    'tabs_is'].max().to_frame('tabs_max')
temp = []
for crit in crit_ls:
    df1 = df.copy()
    df1['crit_level'] = crit
    temp.append(df1)
temp = pd.concat(temp).reset_index()  
rollfamFDR = temp.merge(rollfamFDR, how='left', 
                       on=['signal_family', 'oos_begin_year', 'crit_level'])
rollfamFDR['tabs_hurdle'] = np.where(rollfamFDR['tabs_hurdle'].isnull(), 
                                   rollfamFDR['tabs_max']+1, rollfamFDR['tabs_hurdle'])


# find oos returns by tstat bin
ngroup = 20
rollbin = df_est_oos.copy()

cols = ['ret_is', 'ret_oos']
rollbin[cols] = rollbin[cols] * ret_freq_adj

rollbin['group'] = rollbin.groupby(['signal_family', 'oos_begin_year'])[
    'tstat_is'].transform(lambda x: pd.qcut(x, ngroup, labels=np.arange(1,ngroup+1))
                          ).astype(int)
rollbin = rollbin.groupby(['signal_family', 'oos_begin_year', 'group']).agg(
    {'ret_oos': 'mean',
     'ret_is': 'mean',
     'tstat_is': 'mean',
     }).reset_index()



# prepare result and plot 

sample_ls = [(1983, 2020), (1983, 2004), (2005, 2020)]
for min_yr, max_yr in sample_ls:
    sumfamFDR = rollfamFDR.query('@min_yr <= oos_begin_year <= @max_yr')
    sumfamFDR = sumfamFDR.groupby(['signal_family', 'crit_level'])['tabs_hurdle'].mean()
    sumfamFDR = sumfamFDR.unstack()
    sumfamFDR.columns = [f'crit_{v}' for v in sumfamFDR.columns]

    temp = rollbin.query('@min_yr <= oos_begin_year <= @max_yr')
    sumbin = temp.groupby(['signal_family', 'group'], observed=True).agg(
        {'ret_is': 'mean',
        'ret_oos': 'mean',
        'tstat_is': 'mean'}
        ).reset_index()
    
    df_se = temp.groupby(['signal_family', 'group'], observed=True).agg(
        {'ret_oos': ['std', 'count']})
    df_se.columns = [c[1] for c in df_se.columns]
    df_se = (df_se['std']/np.sqrt(df_se['count']*oos_freq_adj)
             ).to_frame('se_ret_oos')
    sumbin = sumbin.merge(df_se, how='left', on=['signal_family', 'group'])

    sumbin = sumbin.merge(sumfamFDR, how='left', on='signal_family'
                          ).set_index(['signal_family', 'group'])

    # create plots    
    fig = plt.figure(figsize=(8,9))
    for i, family in enumerate(families_use):
        
        df = sumbin.loc[family]
        ax = fig.add_subplot(3,2,i+1)
        
        ax.axhline(y=0, color='grey', alpha=0.9, linewidth=1) 
        ax.errorbar(df['tstat_is'], df['ret_oos'], yerr=df['se_ret_oos']*1.96, 
                    fmt='o', markersize=5, color='k', alpha=0.8)
        
        ax.axvline(x=df['crit_0.01'].iloc[0], color='r', label='BY1.3: FDR<1%')
        ax.axvline(x=-df['crit_0.01'].iloc[0], color='r')
        
        ax.axvline(x=df['crit_0.05'].iloc[0], color='purple', linestyle='--',
                    label='BY1.3: FDR<5%')
        ax.axvline(x=-df['crit_0.05'].iloc[0], color='purple', linestyle='--')

        
        ax.set_ylim(-10, 15)
        ax.set_xlim(-6, 6)
        plt.title(f'{FAMILY_LONG_NAME[family]}', fontsize=12)
        

        if i == 0:
            plt.legend(loc='upper center', fontsize=10)
        if i in [4, 5]:
            ax.set_xlabel('In-sample $t$-statistic', fontsize=12)
        if i == 2:
            ax.set_ylabel('Out of Sample Long-Short Return (% ann)', fontsize=12)
        
    plt.subplots_adjust(hspace=0.3)
    # fig.savefig(path_figs / f'BY1.3-{min_yr}-{max_yr}.pdf', format='pdf', bbox_inches="tight")
    plt.show()

    
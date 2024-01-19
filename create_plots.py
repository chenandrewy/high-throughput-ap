# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:27:44 2023

@author: cdim
"""

# %% environment ---------------------------------------------------------------

import os, re
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
import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)
tqdm.pandas()
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({"font.size": 12})
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']



MachineUsed = 1

if MachineUsed == 1:  # Chuks
    path_input = Path("C:/users/cdim/Dropbox/ChenDim/Stop-Worrying/Data/")
    path_output = Path("C:/users/cdim/Dropbox/ChenDim/Stop-Worrying/Data/")
    path_figs = Path("C:/users/cdim/Dropbox/ChenDim/Stop-Worrying/Chuks/Figures/")
    path_tables = Path("C:/users/cdim/Dropbox/ChenDim/Stop-Worrying/Chuks/Tables/")
elif MachineUsed == 2:  # Andrew
    CPUUsed = 8

    # get working directory
    path_root = Path(os.getcwd() + "/../../")

    path_input = Path(path_root / "Data/")
    path_output = Path(path_root / "Data/")
    path_figs = Path(path_root / "Andrew/Figures/")
    path_tables = Path(path_root / "Andrew/Tables/")


# Select rolling t-stat / oos file
rollsignal_file = "OOS_signal_tstat_OosNyears1.csv.gzip"
# rvpk_rollsignal_file = "RvPk_OOS_signal_tstat_IsNyears5_OosNyears1.csv.gzip"



# USE_SIGN_INFO = True

SIGNAL_FAMILY_NAME_MAP = {
    "DataMinedLongShortReturnsEW": "acct_ew",
    "DataMinedLongShortReturnsVW": "acct_vw",
    "ticker_Harvey2017JF_ew": "ticker_ew",
    "ticker_Harvey2017JF_vw": "ticker_vw",
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

families_use = ['acct_ew', 'acct_vw', 'past_ret_ew', 'past_ret_vw', 
                'ticker_ew', 'ticker_vw', 
                # 'ravenpack_ew', 'ravenpack_vw'
                ]

# %% Define functions ----------------------------------------------------------


# function for fitting t-stats
def fit_fam(tstatvec, famselect):
    # select model based on simple rules
    if np.var(tstatvec) < 1:
        # set GuassianMixture model to standard normal
        mixnorm = GaussianMixture(
            n_components=1, random_state=0, covariance_type="diag", max_iter=500
        )
        mixnorm.means_ = np.array([[0.0]])
        mixnorm.covariances_ = np.array([[1.0]])
        mixnorm.weights_ = np.array([1.0])

    elif "past_ret" not in famselect:
        # estimate simple normal
        mixnorm = GaussianMixture(
            n_components=1, random_state=0, covariance_type="diag", max_iter=500
        )
        mixnorm.fit(tstatvec)
    else:
        # estimate 2 component mixture normal
        mixnorm = GaussianMixture(
            n_components=2, random_state=0, covariance_type="diag", max_iter=500
        )
        mixnorm.fit(tstatvec)

    return mixnorm

# function for shrinkage predictions
def predict_fam(mixnorm, tstatvec, retvec):
    shrink = 1 / mixnorm.covariances_
    shrink[shrink > 1] = 1  # shirnkage cannot be > 1

    # initialize pred
    pred_case = np.zeros((tstatvec.shape[0], 2))
    prob_case = pred_case.copy()
    for case_i in range(0, mixnorm.n_components):
        # predict each case as simple normal updating
        pred_case[:, case_i] = shrink[case_i] * mixnorm.means_[case_i] + (
            1 - shrink[case_i]
        ) * tstatvec.reshape(-1)

        # un-normalized prob of each case
        prob_case[:, case_i] = mixnorm.weights_[case_i] * norm.pdf(
            tstatvec,
            loc=mixnorm.means_[case_i],
            scale=np.sqrt(mixnorm.covariances_[case_i]),
        ).reshape(-1)
    
    # normalize probs
    denom = np.sum(prob_case, axis=1)
    prob_case = prob_case / denom.reshape(-1, 1)
    
    
    # predict t-stat and return
    pred_tstat = np.sum(prob_case * pred_case, axis=1).reshape(-1, 1)
    pred_ret = pred_tstat * retvec / tstatvec
    

    # combine and output
    pred = {"pred_tstat": pred_tstat.flatten(), "pred_ret": pred_ret.flatten()}
    return pred


# %% Load data ----------------------------------------------------------------

# load up signal level data
rollsignal0 = pd.read_csv(path_input / rollsignal_file)

# drop ravenpack for now
rollsignal0 = rollsignal0.query('~signal_family.str.startswith("RavenPack")')

rollsignal0['ret'] = rollsignal0['mean_ret'] * ret_freq_adj
rollsignal0["s_flag"] = rollsignal0["s_flag"].str.lower()

# reshape
rollsignal1 = rollsignal0.pivot(
    index=["signal_family", "oos_begin_year", "signalid"],
    columns="s_flag",
    values=["ret", "tstat"],
)
rollsignal1.columns = ["_".join(col) for col in rollsignal1.columns]
rollsignal1.reset_index(inplace=True)

# clean family names
rollsignal1["signal_family"] = rollsignal1["signal_family"].replace(
    SIGNAL_FAMILY_NAME_MAP
)



# %% plot distribution of in-sample t-stats and model-implied distribution


Nsim = 100_000
seed = 114

# get null t-stat for plotting
np.random.seed(seed)
tstatvec_snorm = pd.Series(np.random.randn(Nsim), name='Null')


for oos_begin_yr, ymax in [(1983, 0.6), (2004, 0.85), (2020, 0.85)]:
    temp = rollsignal1.query('oos_begin_year==@oos_begin_yr')
    
    # initialize figure
    fig = plt.figure(figsize=(7,8))
    for i, family in enumerate(families_use):
        df = temp.query('signal_family == @family')
        retvec, tstatvec = df[["ret_is"]].values,  df[["tstat_is"]].values

        # fit tstats
        mixnorm = fit_fam(tstatvec, family)
        
        # simulate t-stats
        np.random.seed(seed+1)
        tstatvec_sim = mixnorm.sample(Nsim)
        
        # plot for this year group
        ax = fig.add_subplot(3,2,i+1)
        ax.hist(tstatvec, bins=50, density=True, color=COLORS[0], alpha=0.6, 
                label='Data')
        ax.hist(tstatvec_sim[0], bins=50, density=True, color=COLORS[1], 
                alpha=0.6, label='Model')
        tstatvec_snorm.plot(ax=ax, kind='density', color='k', alpha=0.6,)
        ax.set_xlim(-6, 6)
        ax.set_ylim(0, ymax)
        plt.title(f'{FAMILY_LONG_NAME[family]}', fontsize=12)
        
        if i == 0:
            plt.legend(loc='upper left')
        if i in [4, 5]:
            ax.set_xlabel('$t$-statistic', fontsize=12)
        if i%2 != 0:
            ax.set_ylabel('')
        else:
            ax.set_ylabel('Density', fontsize=12)
        
    plt.subplots_adjust(hspace=0.3)
    fig.savefig(path_figs / f'tstat-hist-{oos_begin_yr}.pdf', format='pdf', 
                bbox_inches="tight")
    plt.show()


#%% Get predicted returns with the model for family-year


df_ret_predict = []
for g, df in tqdm(rollsignal1.groupby(['signal_family', 'oos_begin_year'])):
    family, oos_yr = g
    retvec, tstatvec = df[["ret_is"]].values,  df[["tstat_is"]].values
    mixnorm = fit_fam(tstatvec, family)
    pred = predict_fam(mixnorm, tstatvec, retvec)
    pred = pd.DataFrame(pred)
    cols = ['signalid', 'signal_family', 'oos_begin_year', "ret_is", "ret_oos"]
    pred[cols] = df[cols].values
    df_ret_predict.append(pred)

    
df_ret_predict = pd.concat(df_ret_predict)
cols = ['signalid', 'signal_family', 'oos_begin_year', "ret_is", "ret_oos",
        'pred_ret', 'pred_tstat']
df_ret_predict = df_ret_predict.reindex(cols, axis=1)
cols = ["ret_is", "ret_oos", 'pred_ret', 'pred_tstat']
df_ret_predict[cols] = df_ret_predict[cols].astype(float)



#%% Plot IS, OOS and Prediced returns

ngroup = 20
oos_freq_adj = 1
year_set = [(1983, 2004), 
            (2004, 2020), 
            #(1983, 2020), (2020, 2020)
            ]

for year_min, year_max in year_set:
    temp = df_ret_predict.query('@year_min<=oos_begin_year<=@year_max').copy()
    # yr_min, yr_max =  temp['oos_begin_year'].min(), temp['oos_begin_year'].max()
    
    temp['bin'] = temp.groupby(['signal_family', 'oos_begin_year'])['ret_is'].transform(
        lambda x: pd.qcut(x, ngroup, labels=np.arange(1, ngroup+1)))
    
    # summarize by family-oos_year-bin
    temp = temp.groupby(['signal_family', 'oos_begin_year', 'bin'], observed=True).agg(
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
    fig.savefig(path_figs / f'pred-vs-oos-withIS-{year_min}-{year_max}.pdf', format='pdf', 
                bbox_inches="tight")
    plt.show()
    

#%% Get best strategies based on predicted return, then compute OOS return

families_ew = [f for f in families_use if f.endswith('_ew') ]
family_set = {
    '1 all': families_use, 
    '2 acct only': ['acct_ew', 'acct_vw'], 
    '3 acct ew only': ['acct_ew'], 
    '4 pastret only': ['past_ret_ew', 'past_ret_vw'],
    '5 ew only': families_ew
    }

extreme_ranks = [1, 5, 10]


df_best_ret = []
cols = ['oos_begin_year', 'pred_ret', 'ret_oos']
for label, family_ls in family_set.items():
    
    df = df_ret_predict.query('signal_family in @family_ls')[cols].copy()
    df['rank_pct'] = df.groupby('oos_begin_year')['pred_ret'].transform(
        lambda x: x.rank(pct=True)) * 100
    
    for rank in extreme_ranks:
        
        df['port'] = np.where(df['rank_pct']<=rank, 'short',
                              np.where(df['rank_pct'] > 100-rank, 'long', 'neutral')
                              )

        df1 = df.groupby(['port', 'oos_begin_year'])['ret_oos'].agg(
            ['mean', 'std', 'count'])
        df2 = (df1.loc['long', 'mean'] - df1.loc['short', 'mean']).to_frame('ret_oos')
        df2['nstrat'] = df1.loc['long', 'count'] + df1.loc['short', 'count']
        df2['pctmin'] = rank
        df2['family_grp'] = label
        
        df_best_ret.append(df2)
    
df_best_ret = pd.concat(df_best_ret).reset_index()   
    


extreme_ranks_plot = [1, 5, 10]

# plot cumulative performance of top n% strategies
df = df_best_ret.query("family_grp == '1 all' & pctmin in @extreme_ranks_plot")
df = df.pivot(columns=['pctmin'], index='oos_begin_year', values='ret_oos').sort_index()
df.columns = [f'{int(c)}%' for c in df.columns]
df = (1 + df/100).cumprod()
ax = df.plot(figsize=(7,5), style=['-', '--', '-.'])    
ax.set_ylabel('Value of $1 Invvested in 1983') 
ax.set_xlabel('') 
plt.legend(title='Using Strats in Extreme', alignment='left')
fig.savefig(path_figs / 'beststrats-cret.pdf', format='pdf', bbox_inches="tight")
plt.show()   
    
    
    

#### prepare data for summarizing best performing strategies

samp_split = 2004
samp_start = df_best_ret['oos_begin_year'].min()
samp_end = df_best_ret['oos_begin_year'].max()

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
pubret['ret'] = pubret['ret'] * ret_freq_adj

pubcomb = []
for nm, max_yr in [('Pub Anytime', samp_end), ('Pub Pre-2004', samp_split)]:
    df1 = pubret.query('pubyear <= @max_yr')
    df1 = df1.groupby(['year', 'signalname'])[['ret']].mean()
    df1 = df1.groupby('year').agg({'ret': ['mean', 'count']})
    df1.columns = [f'{c[0]}_{c[1]}' for c in df1.columns]
    df1 = df1.rename(columns={'ret_mean': 'ret', 'ret_count': 'nstrat'})
    df1['name'] = nm
    pubcomb.append(df1.reset_index())
pubcomb = pd.concat(pubcomb)


# merge CZ with best performing strategies
best_sumr = df_best_ret.query("family_grp == '1 all'").drop(columns=['family_grp'])
best_sumr['pctmin'] = best_sumr['pctmin'].replace(
    {1: 'DM Extreme 1\%', 5: 'DM Extreme 5\%', 10: 'DM Extreme 10\%',})
best_sumr = best_sumr.rename(columns={'pctmin': 'name', 'ret_oos':'ret', 
                                      'oos_begin_year': 'year'})
best_sumr = pd.concat([best_sumr, pubcomb])


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
    df = df.loc[['DM Extreme 1\%', 'DM Extreme 5\%', 'DM Extreme 10\%', 'Pub Anytime',
                 'Pub Pre-2004']]
    line = pd.DataFrame(index=[f'{min_yr}-{max_yr}'], columns=df.columns)
    df = pd.concat([line, df])
    df.loc[''] = np.nan
    tabdat.append(df)
tabdat = pd.concat(tabdat).iloc[:-1]
cols = ['ret_mean', 'tstat', 'sr']
tabdat[cols] = tabdat[cols].applymap(lambda x: f'{x:.2f}')
tabdat['nstrat_mean'] = tabdat['nstrat_mean'].apply(lambda x: f'{x:.0f}')
tabdat = tabdat.replace({'nan': ''}).rename(
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

with open(path_tables / 'beststrats.tex', 'w') as fh:
    fh.write(latex_table)


#%% HLZ's preferred Benji-Yeki Theorem 1.3 control 

tempsum = rollsignal1.groupby(['signal_family', 'oos_begin_year']
                              )['signalid'].count().to_frame('Nstrat')
tempsum['BY1.3_penalty'] = tempsum['Nstrat'].apply(lambda x: sum(1/np.arange(1, x+1)))


df_FDR = rollsignal1.merge(tempsum, how='left', 
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
rollbin = rollsignal1.copy()
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
    fig.savefig(path_figs / f'BY1.3-{min_yr}-{max_yr}.pdf', format='pdf', bbox_inches="tight")
    plt.show()

    
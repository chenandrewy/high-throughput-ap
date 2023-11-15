# Environment -------------------------------

# v2.1 adds a proper mixture normal fitting to pastret

# sketch of new shrinkage model to deal with funky past ret t-stat distribution
# Andrew 2023 11

# Seems optimal thing is to just do N(\mu, \sigma) for most familes
# And do mixture of N(\mu_pos, \sigma_pos)  N(\mu_neg, \sigma_neg) for pastret

# using sign of predictions to flip oos sign does not work well if you do it at
# the signal level and apply it across all signal groups.
# intuitively, this is too bold.  This transformation will amplify any errosr
# in the modeling.  IT can probalby be done but we need to do it mroe carefully 
# and maybe deeper in the paper after we do the easy stuff.

library(data.table)
library(tidyverse)
library(mixtools)

rm(list = ls())

MATBLUE = rgb(0,0.4470,0.7410)
MATRED = rgb(0.8500, 0.3250, 0.0980)
MATYELLOW = rgb(0.9290, 0.6940, 0.1250)
MATPURPLE = rgb(0.4940, 0.1840, 0.5560)
MATGREEN = rgb(0.4660, 0.6740, 0.1880)

NICEBLUE = "#619CFF"
NICEGREEN = "#00BA38"
NICERED = "#F8766D"


# rollsignalfile = '../Data/OOS_signal_tstat_OosNyears1_IsNyears10.csv.gzip'
# plot_path = '../Figures/sketch IS 10 year/'

rollsignalfile = '../../Data/OOS_signal_tstat_OosNyears1.csv.gzip' # 20 years IS
plot_path = '../Figures/'
table_path = '../Tables/'

ret_freq_adj = 12 # annualize
  
# Load data ----------------------------------------

rollsignal0 <- fread(rollsignalfile) %>% 
    mutate(s_flag = tolower(s_flag)) %>% 
    mutate(ret = ret_freq_adj*mean_ret) %>% 
    mutate(
        signal_family = str_replace(signal_family, "PastReturnSignalsLongShort", "pastret")
        , signal_family = str_replace(signal_family, "DataMinedLongShortReturns", "acct_")
        , signal_family = str_replace(signal_family, "ticker_Harvey2017JF", "ticker")
    ) %>% 
    mutate(signal_family = tolower(signal_family))

rollsignal1 = dcast(rollsignal0, signal_family + signalid + oos_begin_year ~ s_flag,
    value.var = c("ret", "tstat")
)

# remove ravenpack for now
rollsignal1 = rollsignal1 %>% filter(!grepl('ravenpack', signal_family))

famlist = unique(rollsignal1$signal_family)
yearlist = unique(rollsignal1$oos_begin_year)

# Declare functions ----------------------------------------

# function for fitting t-stats
fit_fam = function(tstatvec, fam){

    # if variance overall < 1, assume null
    if (var(tstatvec) < 1){
        par = tibble(mu = 0, sigma = 1, lambda = 1)
        return(par)
    }

    if (fam %in% c('pastret_ew', 'pastret_vw')){
        # fit mixture normal
        mixfit = normalmixEM(tstatvec, k = 2, maxit = 1000, epsilon = 1e-3)
        par = tibble(mu = mixfit$mu, sigma = mixfit$sigma, lambda = mixfit$lambda)
        return(par)        
    } else {
        # fit normal
        par = tibble(mu = mean(tstatvec), sigma = sd(tstatvec), lambda = 1)         
        return(par)
    }
    

} # end fit_fam
# function for simulating t-stats 
sim_fam = function(par, N = 1e4){
    if (nrow(par) == 1){
        tstat_sim = rnorm(N, mean = par$mu, sd = par$sigma)
    } else {
        tstat_sim = rnormmix(N, lambda=par$lambda, mu=par$mu, sigma=par$sigma)
    }
    return(tstat_sim)
} # end sim_fam
# function for shrinkage predictions
predict_fam = function(par, signaldt){
    # signaldt has columns: signalid, tstat_is, ret_is

    # find shrinkage for each case
    par$shrink = 1 /par$sigma^2
    par$shrink = pmin(par$shrink, 1) # shrinkage cannot be > 1

    # predict depending on mix normal or not
    # could be "parallelized" if we filled par to have 2 rows always
    if (nrow(par) == 1){
        # simple normal
        signaldt = signaldt %>% 
            mutate(pred_tstat = par$shrink*par$mu + (1-par$shrink)*tstat_is
                , pred_ret = pred_tstat * ret_is / tstat_is)
    } else {
        # mix normal
        pred_case1 = par[1, ]$shrink*par[1, ]$mu + (1-par[1, ]$shrink)*signaldt$tstat_is
        pred_case2 = par[2, ]$shrink*par[2, ]$mu + (1-par[2, ]$shrink)*signaldt$tstat_is

        like_case1 = dnorm(signaldt$tstat_is, mean = par[1, ]$mu, sd = par[1, ]$sigma) 
        like_case2 = dnorm(signaldt$tstat_is, mean = par[2, ]$mu, sd = par[2, ]$sigma)
        prob_case1 = par[1, ]$lambda * like_case1 / (par[1, ]$lambda * like_case1 + par[2, ]$lambda * like_case2)        
        prob_case2 = 1 - prob_case1

        signaldt = signaldt %>% 
            mutate(pred_tstat = prob_case1*pred_case1 + prob_case2*pred_case2
                , pred_ret = pred_tstat * ret_is / tstat_is)        
    } # end if nrow(par) == 1

    return(signaldt)
} # end predict_fam
# name families
long_family_name = function(fam){
    longname = case_when(
        fam == 'acct_ew' ~ 'Accounting EW'
        , fam == 'acct_vw' ~ 'Accounting VW'
        , fam == 'pastret_ew' ~ 'Past Return EW'
        , fam == 'pastret_vw' ~ 'Past Return VW'
        , fam == 'ticker_ew' ~ 'Ticker EW'
        , fam == 'ticker_vw' ~ 'Ticker VW'
    )
}

# Model t-stats at selected years ----------------------------------------

tstat_dist_pdf = function(yearselect = 1983, ylimit = c(0,0.6)){

    # keep only first oos_begin_year for each signal_family and s_flag
    signalcur = rollsignal1 %>% 
        arrange(signal_family, oos_begin_year) %>%
        group_by(signal_family) %>%
        filter(oos_begin_year == yearselect) %>%
        setDT()

    # fit model for each family
    parlist = list()
    for (fam in famlist){
        parcur = fit_fam(signalcur[signal_family == fam, tstat_is], fam)
        parcur$signal_family = fam        
        parlist[[fam]] = parcur
    }
    parlist = bind_rows(parlist) %>% setDT()

    # simulate model for each family
    Nsim = 1e4
    simfit = data.table()
    for (fam in famlist){
        tstat_sim = sim_fam(parlist[signal_family == fam,], Nsim)
        simfit = rbind(simfit, 
            data.table(signal_family = fam, tstat_sim = tstat_sim
                , signalid = 1:Nsim)
        )
    }

    # simulate null
    simnull = data.table()
    for (fam in famlist){
        tstat_sim = rnorm(Nsim, mean = 0, sd = 1)
        simnull = rbind(simnull, 
            data.table(signal_family = fam, tstat_sim = tstat_sim
                , signalid = 1:Nsim)
        )
    }

    # for labeling variances
    varlab = parlist %>%
        mutate(var = sigma^2) %>% 
         select(signal_family, var) %>% 
        group_by(signal_family) %>% mutate(id = row_number()) %>%
                pivot_wider(names_from = id, names_prefix = 'var_'
                , values_from = 'var')      

    # histogram of tstat_is vs tstat simulated
    plotme = signalcur %>% 
        transmute(signal_family, signalid, tstat_is) %>% 
        mutate(type = 'emp') %>% 
        rbind(
            simfit %>% transmute(signal_family, signalid, tstat_is = tstat_sim) %>% 
            mutate(type = 'fit')
        ) %>% 
        rbind(
            simnull %>% transmute(signal_family, signalid, tstat_is = tstat_sim) %>% 
            mutate(type = 'null')
        ) %>% 
        mutate(
            signal_family = long_family_name(signal_family)
            , type = case_when(
                type == 'emp' ~ 'Data'
                , type == 'fit' ~ 'Model'
                , type == 'null' ~ 'Null'
            )
        )

    p = plotme %>%
        ggplot(aes(x = tstat_is, group = type)) +
        geom_density(data = plotme %>% filter(type == 'Null') 
                , aes(y = ..density.., fill = type, color = type), color = 'black', alpha = 0.6
                , adjust = 5) +
        geom_histogram(data = plotme %>% filter(type != 'Null')
            , aes(y = ..density.., fill = type)
            , position = 'identity', bins = 50, alpha = 0.6) +
        facet_wrap(~ signal_family, scales = "free_x", nrow = 3) +
        theme_bw() +
        theme(legend.position = c(1,9.3)/10
            , legend.title = element_blank()
            , text = element_text(size = 16)) +
        theme(legend.text = element_text(size = 10)) +
        scale_fill_manual(values = c(MATBLUE,MATRED, 'white')
            , labels = c('Data', 'Model', 'Null')) +
        scale_color_manual(values = c(MATBLUE,MATRED, 'black')
            , labels = c('Data', 'Model', 'Null'))  +
        xlab('t-statistic') +
        coord_cartesian(xlim = c(-1,1)*6, ylim = ylimit) +
        scale_x_continuous(breaks = seq(-10, 10, 2)) 
    ggsave(paste0(plot_path, 'tstat-hist-', yearselect, '.pdf')
            , p, scale = 1, height = 8, width = 6)

} # end tstat_dist_pdf

# run plotting function for select years
tstat_dist_pdf(1983, ylimit = c(0,0.6))
tstat_dist_pdf(2004, ylimit = c(0,0.8))
tstat_dist_pdf(2020, ylimit = c(0,0.8))

# Predict returns each year ---------------------------

# fit models each fam-year
rollfit = list()
rollpred = list()
for (yearcur in yearlist){
    paste0('fitting year ', yearcur) %>% print()
    for (fam in famlist){        
        signalcur = rollsignal1 %>% 
            filter(oos_begin_year == yearcur, signal_family == fam)
        fitcur = fit_fam(signalcur$tstat_is, fam) %>% 
            mutate(oos_begin_year = yearcur, signal_family = fam)
        signal_cur = predict_fam(fitcur, signalcur) 

        # save
        rollfit[[paste0(fam, yearcur)]] = fitcur
        rollpred[[paste0(fam, yearcur)]] = signal_cur
    }
}
rollfit = bind_rows(rollfit) %>% setDT()
rollpred = bind_rows(rollpred) %>% setDT()

# Plot by family / bin -------------------------------------------

pdf_plot_rollpred = function(ngroup = 20, oos_freq_adj = 1, yearmin = 1983, yearmax = 2020){
    temp = rollpred %>% filter(oos_begin_year >= yearmin & oos_begin_year <= yearmax)
    yearrange = c(min(temp$oos_begin_year), max(temp$oos_begin_year))

    p = temp %>% 
            group_by(signal_family, oos_begin_year) %>%
            mutate(bin = ntile(ret_is, ngroup)) %>%
            group_by(signal_family, oos_begin_year, bin) %>% 
            summarize(
                pred_ret = mean(pred_ret)
                , ret_is = mean(ret_is)
                , ret_oos = mean(ret_oos)
            ) %>% 
            group_by(signal_family, bin) %>% 
            summarize(
                pred_ret = mean(pred_ret)
                , se_ret_oos = sd(ret_oos)/sqrt(n()*oos_freq_adj)
                , ret_oos = mean(ret_oos)
            )  %>% 
            mutate(signal_family = long_family_name(signal_family)) %>% 
            ggplot(aes(x = bin)) +
            geom_hline(yintercept = 0, color = 'lightgray') +            
            # plot line / errorbar
            geom_line(aes(y = pred_ret, color = MATRED), size = 0.8) + 
            geom_point(aes(y = ret_oos, color = MATBLUE)) +
            geom_errorbar(aes(ymin = ret_oos - 1.96*se_ret_oos
                , ymax = ret_oos + 1.96*se_ret_oos
                , color = MATBLUE), width = 0.2) +
            facet_wrap(~ signal_family, scales = "free_x", nrow = 3) +
            coord_cartesian(ylim = c(-12, 12)) +
            xlab('in-sample return group') +
            ylab('Long-Short Return (% p.a.)') +
            theme_bw() +
            theme(legend.position = c(1.7,9.45)/10, legend.title = element_blank()
                , text = element_text(size = 18), legend.text = element_text(size = 12)
                , legend.margin = margin(t=0, unit = 'cm'))  +
            scale_color_manual(
                values = c(MATBLUE, MATRED)
                , labels = c('Out-of-sample', 'Predicted')
            )

    ggsave(paste0(plot_path, 'pred-vs-oos-', yearrange[1], '-', yearrange[2], '.pdf'), p, scale = 0.5)

}

# plot 
pdf_plot_rollpred(ngroup = 20, oos_freq_adj = 1, yearmin = 1983, yearmax = 2004)
pdf_plot_rollpred(ngroup = 20, oos_freq_adj = 1, yearmin = 2004, yearmax = 2020)
pdf_plot_rollpred(ngroup = 20, oos_freq_adj = 1, yearmin = 1983, yearmax = 2020)
pdf_plot_rollpred(ngroup = 20, oos_freq_adj = 1, yearmin = 2020, yearmax = 2020)

# Plot by family / bin + in-sample ret -------------------------------------------

pdf_plot_rollpred_is = function(ngroup = 20, oos_freq_adj = 1, yearmin = 1983, yearmax = 2020){
    temp = rollpred %>% filter(oos_begin_year >= yearmin & oos_begin_year <= yearmax)
    yearrange = c(min(temp$oos_begin_year), max(temp$oos_begin_year))

    p = temp %>% 
            group_by(signal_family, oos_begin_year) %>%
            mutate(bin = ntile(ret_is, ngroup)) %>%
            group_by(signal_family, oos_begin_year, bin) %>% 
            summarize(
                pred_ret = mean(pred_ret)
                , ret_is = mean(ret_is)
                , ret_oos = mean(ret_oos)
            ) %>% 
            group_by(signal_family, bin) %>% 
            summarize(
                pred_ret = mean(pred_ret)
                , se_ret_oos = sd(ret_oos)/sqrt(n()*oos_freq_adj)
                , ret_is = mean(ret_is)
                , ret_oos = mean(ret_oos)
            )  %>% 
            mutate(signal_family = long_family_name(signal_family)) %>% 
            ggplot(aes(x = bin)) +
            geom_hline(yintercept = 0, color = 'lightgray') +            
            # plot line / errorbar
            geom_line(aes(y = ret_is, color = 'gray')
                , linetype = 'dashed', size = 0.8) +            
            geom_point(aes(y = ret_oos, color = MATBLUE)) +
            geom_line(aes(y = pred_ret, color = MATRED), size = 0.8) +                             
            geom_errorbar(aes(ymin = ret_oos - 1.96*se_ret_oos
                , ymax = ret_oos + 1.96*se_ret_oos
                , color = MATBLUE), width = 0.2) +
            facet_wrap(~ signal_family, scales = "free_x", nrow = 3) +
            coord_cartesian(ylim = c(-12, 12)) +
            xlab('in-sample return group') +
            ylab('Long-Short Return (% ann)') +
            theme_bw() +
            theme(legend.position = c(1.3,9.37)/10
                , legend.title = element_blank()
                , text = element_text(size = 18)
                , legend.text = element_text(size = 8)
                , legend.margin = margin(t=-0.1, unit = 'cm')
                , legend.key = element_rect(fill = 'transparent')
                , legend.box.background = element_blank()) +
            scale_color_manual(
                values = c(MATBLUE, MATRED, 'gray40')
                , labels = c('OOS', 'Predicted', 'In-Samp')
            ) 
    ggsave(paste0(plot_path, 'pred-vs-oos-withIS-', yearrange[1], '-', yearrange[2], '.pdf'), p
        , scale = 1, width = 6, height = 8)
}

# plot 
pdf_plot_rollpred_is(ngroup = 20, oos_freq_adj = 1, yearmin = 1983, yearmax = 2004)
pdf_plot_rollpred_is(ngroup = 20, oos_freq_adj = 1, yearmin = 2004, yearmax = 2020)
pdf_plot_rollpred_is(ngroup = 20, oos_freq_adj = 1, yearmin = 1983, yearmax = 2020)
pdf_plot_rollpred_is(ngroup = 20, oos_freq_adj = 1, yearmin = 2020, yearmax = 2020)

# Find the best strats and plot -----------------------
# here we apply signs
top_pct_select = c(0.5, 1,5,10)

# flip signs
rollrank = rollpred %>% 
    select(signal_family, signalid, oos_begin_year, pred_ret, ret_oos) %>%
    mutate(pred_sign = sign(pred_ret)) %>% 
    mutate(
        pred_ret = pred_ret * pred_sign
        , ret_oos = ret_oos * pred_sign
    ) 

# define "best" strategies
Nstratall = rollrank %>% distinct(signal_family, signalid) %>% nrow()
rankdat = tibble(top_pct = top_pct_select) %>% 
    mutate(top_rank = round(top_pct/100*Nstratall)) 

# settings for best signals
famfilterlist = list(
    famlist, famlist[grepl('acct', famlist)]
    , famlist[grepl('acct_ew', famlist)], famlist[grepl('pastret', famlist)]
    , famlist[!grepl('_ew', famlist)]
)
famnamelist = c('1 all', '2 acct only'
        , '3 acct ew only', '4 pastret only'
        , '5 ew only')

# find cummulative returns for best signals
retbest = data.table()
for (famfilteri in 1:length(famfilterlist)){
    famfilter = famfilterlist[[famfilteri]]
    rollselect = rollrank %>% 
        filter(signal_family %in% famfilter) %>% 
        group_by(oos_begin_year) %>%
        arrange(desc(pred_ret)) %>% mutate(rank = row_number()) %>% 
        ungroup() %>% 
        mutate(famfilter = famnamelist[famfilteri]) %>% 
        setDT()

    for (rankx in rankdat$top_rank){
        
        temp = rollselect %>% 
            filter(rank <= rankx) %>%
            group_by(famfilter, oos_begin_year) %>%
            summarize(ret_oos = mean(ret_oos)) %>% 
            mutate(nstrat = as.factor(rankx)) %>% 
            mutate(pctmin = as.factor(rankdat[ rankdat$top_rank == rankx, ]$top_pct))

        retbest = rbind(retbest, temp)
    }
}

# plot cumulative return using various specs
legtitle = 'best pct strats, pct='
p = retbest %>% 
    group_by(famfilter, pctmin) %>% 
    arrange(famfilter, pctmin, oos_begin_year) %>%
    mutate(logcret = cumsum(log(1+ret_oos/100))) %>%
    mutate(pctmin = paste0(pctmin, '%')) %>% 
    ggplot(aes(x = oos_begin_year, y = logcret, group = pctmin)) +
    geom_hline(yintercept = 0, color = 'grey') +
    geom_line(aes(color = pctmin, linetype = pctmin))  +
    theme(legend.position = c(8,2)/10) +
    facet_wrap(~ famfilter, scales = "free_x", ncol = 2, nrow = 3) +
    scale_color_discrete(name = legtitle) +
    scale_linetype_discrete(name = legtitle) 

ggsave(paste0(plot_path, 'sketch-cret.pdf'), p, scale = 0.5)

# plot in levels
colorlist = c(MATBLUE, MATRED, MATYELLOW, MATPURPLE, MATGREEN)
legtitle = 'Using Strats in Top'
p =  retbest %>% 
    filter(famfilter == '1 all') %>%
    filter(pctmin %in% c(0.5,1,5)) %>%     
    group_by(pctmin) %>%
    arrange(pctmin, oos_begin_year) %>%
    mutate(pctmin = paste0(pctmin, '%')) %>%     
    mutate(pctmin = factor(pctmin, levels = c('0.5%', '1%', '5%'))) %>%
    mutate(cret = cumprod(1+ret_oos/100)) %>%
    ggplot(aes(x = oos_begin_year, y = cret, group = pctmin)) +
    geom_hline(yintercept = 0, color = 'grey') +
    geom_line(aes(color = pctmin, linetype = pctmin), size = 1)  +
    theme_bw() +
    theme(legend.position = c(2.5,8)/10
        , text = element_text(size = 20), legend.title = element_text(size = 14)
        , legend.key.size = unit(2,'line')) +
    scale_color_manual(values = colorlist, name = legtitle) +
    scale_linetype_manual(values = c(1,2,4), name = legtitle) +
    xlab(NULL) +
    ylab('Value of $1 Invested in 1983')  +
    coord_cartesian(xlim = c(1980, 2020)) +
    scale_y_continuous(trans='log10', breaks = 0:10)
ggsave(paste0(plot_path, 'beststrats-cret.pdf'), p, scale = 0.5)    
# summarize to console ----------------------------
retbest %>% 
    group_by(famfilter, pctmin, nstrat) %>%
    summarize(rbar = mean(ret_oos), vol = sd(ret_oos), nyear= n()) %>% 
    mutate(sharpe = rbar/vol) %>% 
    mutate(tstat = rbar/vol*sqrt(nyear))

retbest %>% 
    mutate(subsamp = if_else(oos_begin_year <= 2004, 'pre-2004','post-2004') )%>%
    group_by(famfilter, nstrat, pctmin, subsamp) %>%
    summarize(rbar = mean(ret_oos), vol = sd(ret_oos), nyear= n()) %>% 
    mutate(sharpe = rbar/vol) %>% 
    mutate(tstat = rbar/vol*sqrt(nyear)) %>% 
    arrange(nstrat, subsamp, famfilter) %>% 
    print(n=Inf) 
    
# Latex Table of best returns  ----------------------------

library(xtable)

samp_split = 2004
samp_start = retbest$oos_begin_year %>% min()
samp_end = retbest$oos_begin_year %>% max()

# for comparison, read in CZ returns
pubret0 = fread('../../Data/PredictorLSretWide.csv')
pubdoc = readxl::read_excel('../../Data/PredictorSummary.xlsx') %>% 
    transmute(signalname, pubyear = Year, sampend = SampleEndYear)

# process CZ returns 
pubret = melt(pubret0, id.vars = 'date'
    , variable.name = 'signalname'
    , value.name = 'ret') %>% 
    filter(!is.na(ret)) %>% 
    left_join(pubdoc, by = 'signalname') %>% 
    mutate(year = year(date)) %>%
    filter(year >= samp_start & year <= samp_end) 

# note we should annualize better later
pubcomb = pubret %>%     
    group_by(year, signalname) %>% summarize(ret = 12*mean(ret)) %>%
    group_by(year) %>% summarize(ret = mean(ret), nstrat = n()) %>% 
    mutate(name = 'pubcomb') %>% 
    rbind(
        pubret %>%     
        filter(pubyear <= samp_split) %>%
        group_by(year, signalname) %>% summarize(ret = 12*mean(ret)) %>%
        group_by(year) %>% summarize(ret = mean(ret), nstrat = n()) %>% 
        mutate(name = 'pub_pre2004') 
    )

# add to retbest
retbest2 = retbest %>% 
    filter(famfilter == '1 all') %>%  
    rbind(pubcomb %>% transmute(
            famfilter = '1 all', oos_begin_year = year, ret_oos = ret
            , nstrat, pctmin = name)) %>% 
    mutate(pctmin = pctmin %>% as.character()
        , nstrat = nstrat %>% as.character() %>% as.numeric()) %>% 
    mutate(pctmin = factor(pctmin, levels = c('0.5','1','5','10','pubcomb','pub_pre2004')
        , labels = c('DM Top 0.5\\%', 'DM Top 1\\%', 'DM Top 5\\%', 'DM Top 10\\%'
            , 'Pub Anytime', 'Pub Pre-2004'))) 

tabdat = retbest2 %>% 
    # subsample stats
    mutate(subsamp = if_else(oos_begin_year <= samp_split
        , paste0(samp_start, '-', samp_split), paste0(samp_split+1, '-', samp_end)) ) %>%
    group_by(pctmin, subsamp) %>%
    summarize( rbar = mean(ret_oos), vol = sd(ret_oos), nyear= n()
        , sharpe = rbar/vol, tstat = rbar/vol*sqrt(nyear), nstrat = mean(nstrat)) %>% 
    # add overall stats
    rbind(
        retbest2 %>% 
        mutate(subsamp = paste0(samp_start, '-', samp_end)) %>%
        group_by(pctmin, subsamp) %>%
        summarize( rbar = mean(ret_oos), vol = sd(ret_oos), nyear= n()
            , sharpe = rbar/vol, tstat = rbar/vol*sqrt(nyear), nstrat = mean(nstrat)) 
    ) %>% 
    # organize
    mutate(sampid = case_when(subsamp == '1983-2020' ~ 1
            , subsamp == '1983-2004' ~ 2
            , subsamp == '2005-2020' ~ 3)) %>%     
    arrange(sampid, pctmin) %>% 
    ungroup()

# add header rows
tabdat2 = tabdat %>% rbind(
    tabdat %>% distinct(subsamp, .keep_all = TRUE) %>% 
        mutate(across(-c(subsamp,sampid), function(x) {return(NA)})
            , pctmin = 'blank') 
    )  %>% 
    mutate(pctmin = factor(pctmin, 
        levels = c('blank',levels(tabdat$pctmin)))) %>% 
    arrange(sampid, pctmin)  %>% 
    # clean up
    filter(!grepl('DM Top 0.5', pctmin))

# output tex
xtable(tabdat2 %>% 
        select(pctmin, nstrat, rbar, tstat, sharpe) 
    , digits = c(0,0,0,2,2,2), align = 'llrccc') %>%
    print(include.rownames = FALSE, 
        , sanitize.text.function = function(x){x}) %>% 
        cat(file = paste0(table_path, 'beststrats.tex'))

# read in tex and edit
texline = readLines(paste0(table_path,'beststrats.tex'))

texline[7] = ' & Num Strats & Mean Return & t-stat & Sharpe Ratio \\\\'
texline[8] = ' & \\multicolumn{1}{c}{Combined} & \\multicolumn{1}{c}{(\\% ann)} & & (ann) \\\\ \\hline'

# find blanks
blankrow = which(grepl('blank', texline))
templeft = '\\multicolumn{4}{l}{'
tempright = '}\\\\ \\hline'
tempmid = tabdat2$subsamp %>% unique()
for (i in 1:length(blankrow)){
    texline[blankrow[[i]]] = paste0(templeft, tempmid[[i]], tempright)
}

# remove table environment
texline = texline[-c(3,4,length(texline))]

# == cosmetics ==
texline2 = texline

# add extra spaces 
rowlist = which(grepl('Pub Pre-2004', texline2))
for (i in 1:(length(rowlist)-1)){
    texline2[rowlist[[i]]] = paste0(texline2[rowlist[[i]]], '\\hline \\\\')
}

# add extra guide lines
rowlist = which(grepl('Pub Any', texline2)) - 1
for (i in rowlist){
    texline2[i] = paste0(texline2[i], ' \\hline')
}

writeLines(texline2, paste0(table_path,'beststrats.tex'))

# HLZ's preferred Benji-Yeki Theorem 1.3 control -------------------------------------------
options(tibble.width = Inf)

# Estimate FDR bounds
tempsum = rollsignal1 %>% 
    group_by(signal_family, oos_begin_year) %>%
    summarize(Nstrat = n()) %>% 
    group_by(signal_family, oos_begin_year) %>%
    mutate(BY1.3_penalty = sum(1/(1:Nstrat))) 

rollsignalFDR = rollsignal1 %>% 
    left_join(tempsum, by = c('signal_family', 'oos_begin_year')) %>%
    group_by(signal_family, oos_begin_year) %>%
    mutate(tabs_is = abs(tstat_is)) %>%
    arrange(signal_family, oos_begin_year, desc(tabs_is)) %>% 
    mutate(
         Pr_null = 2*pnorm(-tabs_is)
        , Pr_emp = row_number()/Nstrat
        , FDRmax_BH = Pr_null/Pr_emp
        , FDRmax_BY1.3 = FDRmax_BH*BY1.3_penalty
    ) 

# find t-stat hurdles
crit_list = c(1, 5, 10) / 100
rollfamFDR = list()
for (crit in crit_list){
    rollfamFDR[[as.character(crit)]] = rollsignalFDR %>% 
        group_by(signal_family, oos_begin_year) %>%
        filter(FDRmax_BY1.3 <= crit) %>%
        summarize(tabs_hurdle = min(tabs_is)) %>% 
        mutate(crit_level = crit)
}
rollfamFDR = bind_rows(rollfamFDR) %>% setDT()

# define hurdle as max(tstat_is)+1 if no signals pass
rollfamFDR = rollsignalFDR %>% 
    group_by(signal_family, oos_begin_year) %>%
    summarize(tabs_max = max(tabs_is)) %>%
    expand_grid(crit_level = crit_list) %>% 
    left_join(rollfamFDR, by = c('signal_family', 'oos_begin_year', 'crit_level')) %>% 
    mutate(tabs_hurdle = if_else(is.na(tabs_hurdle), tabs_max+1, tabs_hurdle)) 
    
# find oos returns by tstat bin
ngroup = 20
rollbin = rollsignal1 %>% 
    group_by(signal_family, oos_begin_year) %>%
    mutate(group = ntile(tstat_is, ngroup)) %>% 
    group_by(signal_family, oos_begin_year, group) %>% 
    summarize(
        ret_oos = mean(ret_oos)
        , ret_is = mean(ret_is)
        , tstat_is = mean(tstat_is)
    ) 

# function for plotting
pdf_plot_BY1_3 = function(samp_min = 1983, samp_max = 2020){

    # summarize for selected sample
    sumfamFDR = rollfamFDR %>% 
        filter(oos_begin_year >= samp_min & oos_begin_year <= samp_max) %>%
        group_by(signal_family, crit_level) %>%
        summarize(tabs_hurdle = mean(tabs_hurdle)) %>% 
        arrange(signal_family, crit_level) %>% 
        pivot_wider(names_from = crit_level, values_from = tabs_hurdle, names_prefix = 'crit_') 

    sumbin = rollbin %>% 
        filter(oos_begin_year >= samp_min & oos_begin_year <= samp_max) %>%
        group_by(signal_family, group) %>%
        summarize(
            se_ret_oos = sd(ret_oos)/sqrt(n()), ret_oos = mean(ret_oos)
            , ret_is = mean(ret_is), tstat_is = mean(tstat_is)
        ) 

    # plot
    leglab = c('BY1.3: FDR<1%', 'BY1.3: FDR<5%')
    p = sumbin %>% 
        left_join(sumfamFDR, by = 'signal_family') %>%
        mutate(signal_family = long_family_name(signal_family)) %>%
        ggplot(aes(x = tstat_is, y = ret_oos, group = signal_family)) +
        geom_hline(yintercept = 0, color = 'gray50') +
        # plot main stuff
        geom_point() +
        geom_errorbar(aes(ymin = ret_oos - 1.96*se_ret_oos
            , ymax = ret_oos + 1.96*se_ret_oos), width = 0.2) +
        # plot hurdle lines
        geom_vline(aes(xintercept = crit_0.01, color = '1%', linetype = '1%')) +
        geom_vline(aes(xintercept = -1*crit_0.01), color = MATRED, linetype = 'solid') +
        geom_vline(aes(xintercept = crit_0.05, color = '5%', linetype = '5%')) +
        geom_vline(aes(xintercept = -1*crit_0.05), color = MATPURPLE, linetype = 'dashed') +
        scale_color_manual(name = 'BY1.3 Hurdles'
            , values = c('1%' = MATRED, '5%' = MATPURPLE)
            , labels = leglab) +
        scale_linetype_manual(name = 'BY1.3 Hurdles'
            , values = c('1%' = 'solid', '5%' = 'dashed')
            , labels = leglab) +        
        facet_wrap(~ signal_family, scales = "free_x", nrow = 3) +
        theme_bw() +
        theme(legend.position = c(1.5,9.4)/10
            , legend.title = element_blank()
            , text = element_text(size = 18)
            , legend.text = element_text(size = 9)
            # , legend.margin = margin(t=0.1, unit = 'cm')
            , legend.key = element_rect(fill = 'transparent')
            , legend.box.background = element_rect(colour = 'black')) +
        coord_cartesian(xlim = c(-1,1)*6)  +
        ylab('Out-of-Sample Long-Short Return (% ann)') +
        xlab('in-sample t-statistic')
    ggsave(paste0(plot_path, paste0('BY1.3-', samp_min, '-', samp_max), '.pdf'), p
        , scale = 1, width = 6, height = 8)

} # end plotting function

# run plotting function
pdf_plot_BY1_3(samp_min = 1983, samp_max = 2004)
pdf_plot_BY1_3(samp_min = 2004, samp_max = 2020)
pdf_plot_BY1_3(samp_min = 1983, samp_max = 2020)

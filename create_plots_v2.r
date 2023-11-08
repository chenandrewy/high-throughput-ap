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

# Load data ----------------------------------------

rollsignal0 <- fread(rollsignalfile) %>% 
    mutate(s_flag = tolower(s_flag)) %>% 
    rename(ret = mean_ret) %>% mutate(
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

    # # debug
    # fit_fam = function(tstatvec, fam){
    #     # fit normal
    #     par = tibble(mu = mean(tstatvec), sigma = sd(tstatvec), lambda = 1)         
    #     par$mu = 0
    #     return(par)
    # } # end fit_fam

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

# Model t-stats at selected years ----------------------------------------

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

tstat_dist_pdf = function(yearselect = 1983){

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
        # left_join(varlab, by = 'signal_family') %>% 
        # mutate(
        #     signal_family = paste0(signal_family
        #     , ' var1=', round(var_1, 1)
        #     , ' var2=', round(var_2, 1)
        #     )
        # )
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
        theme(legend.position = c(1,9.2)/10, legend.title = element_blank())  +
        scale_fill_manual(values = c(MATBLUE,MATRED, 'white')
            , labels = c('Data', 'Model', 'Null')) +
        scale_color_manual(values = c(MATBLUE,MATRED, 'black')
            , labels = c('Data', 'Model', 'Null'))  +
        xlab('t-statistic') +
        coord_cartesian(xlim = c(-1,1)*6, ylim = c(0,0.9)) +
        scale_x_continuous(breaks = seq(-10, 10, 2)) 
    ggsave(paste0(plot_path, 'tstat-hist-', yearselect, '.pdf'), p, scale = 0.5)

} # end tstat_dist_pdf

# run plotting function for select years
tstat_dist_pdf(1983)
tstat_dist_pdf(2004)
tstat_dist_pdf(2020)

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
oos_freq_adj = 12
ngroup = 20

pdf_plot_rollpred = function(ngroup = 20, oos_freq_adj = 12, yearmin = 1983, yearmax = 2020){
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
            coord_cartesian(ylim = c(-1, 1)) +
            xlab('in-sample return group') +
            ylab('Long-Short Return (% p.m.)') +
            theme_bw() +
            theme(legend.position = c(1.3,9.3)/10, legend.title = element_blank())  +
            scale_color_manual(
                values = c(MATBLUE, MATRED)
                , labels = c('Out-of-sample', 'Predicted')
            )

    ggsave(paste0(plot_path, 'pred-vs-oos-', yearrange[1], '-', yearrange[2], '.pdf'), p, scale = 0.5)

}

# plot 
pdf_plot_rollpred(ngroup = 20, oos_freq_adj = 12, yearmin = 1983, yearmax = 2004)
pdf_plot_rollpred(ngroup = 20, oos_freq_adj = 12, yearmin = 2004, yearmax = 2020)
pdf_plot_rollpred(ngroup = 20, oos_freq_adj = 12, yearmin = 1983, yearmax = 2020)
pdf_plot_rollpred(ngroup = 20, oos_freq_adj = 12, yearmin = 2020, yearmax = 2020)


# find the best signals across families  -----------------------
# here we apply signs

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
rankdat = tibble(top_pct = c(0.1, 1, 5)) %>% 
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
            mutate(rankmin = as.factor(rankx)) %>% 
            mutate(pctmin = as.factor(rankdat[ rankdat$top_rank == rankx, ]$top_pct))

        retbest = rbind(retbest, temp)
    }
}

# plot cumulative return using various specs
legtitle = 'best pct strats, pct='
p = retbest %>% 
    group_by(famfilter, pctmin) %>% 
    arrange(famfilter, pctmin, oos_begin_year) %>%
    mutate(logcret = cumsum(log(1+ret_oos/100*12))) %>%
    mutate(pctmin = paste0(pctmin, '%')) %>% 
    ggplot(aes(x = oos_begin_year, y = logcret, group = pctmin)) +
    geom_hline(yintercept = 0, color = 'grey') +
    geom_line(aes(color = pctmin, linetype = pctmin))  +
    theme(legend.position = c(8,2)/10) +
    facet_wrap(~ famfilter, scales = "free_x", ncol = 2, nrow = 3) +
    scale_color_discrete(name = legtitle) +
    scale_linetype_discrete(name = legtitle) 

ggsave(paste0(plot_path, 'sketch-cret.pdf'), p, scale = 0.5)

# plot nicely for paper
colorlist = c(MATBLUE, MATRED, MATYELLOW, MATPURPLE, MATGREEN)
legtitle = 'Strats in Top'
p = retbest %>% 
    filter(famfilter == '1 all') %>%
    group_by(pctmin) %>%
    arrange(pctmin, oos_begin_year) %>%
    mutate(pctmin = paste0(pctmin, '%')) %>%     
    mutate(logcret = cumsum(log(1+ret_oos/100*12))) %>%
    ggplot(aes(x = oos_begin_year, y = logcret, group = pctmin)) +
    geom_hline(yintercept = 0, color = 'grey') +
    geom_line(aes(color = pctmin, linetype = pctmin), size = 1) +
    theme_bw() +
    theme(legend.position = c(2,8)/10
        , text = element_text(size = 20)
        , legend.key.size = unit(2,'line')) +
    scale_color_manual(values = colorlist, name = legtitle) +
    scale_linetype_manual(values = c(1,2,4), name = legtitle) +
    xlab(NULL) +
    ylab('Log Cumulative Return') 
ggsave(paste0(plot_path, 'beststrats-cret.pdf'), p, scale = 0.2)

    


# summarize to console ----------------------------
retbest %>% 
    group_by(famfilter, rankmin) %>%
    summarize(rbar = mean(ret_oos), vol = sd(ret_oos), nyear= n()) %>% 
    mutate(sharpe = rbar/vol) %>% 
    mutate(tstat = rbar/vol*sqrt(nyear*12))


retbest %>% 
    mutate(subsamp = if_else(oos_begin_year <= 2004, 'pre-2004','post-2004') )%>%
    group_by(famfilter, rankmin, subsamp) %>%
    summarize(rbar = mean(ret_oos), vol = sd(ret_oos), nyear= n()) %>% 
    mutate(sharpe = rbar/vol) %>% 
    mutate(tstat = rbar/vol*sqrt(nyear*12)) %>% 
    print(n=Inf)

# inspect ----------------------------

rollfit[signal_family == 'pastret_vw' & oos_begin_year == 2019]
rollfit[oos_begin_year == 2019]

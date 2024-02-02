# 2024 01 flexible distr EB estimation
# uses python style command line option parsing
# should be able to run with 
# Rscript estimate_EB_distr.r --data_path ../../Data/ --rollsignal_name OOS_signal_tstat_OosNyears1.csv.gzip --out_prefix Debug_
# or 
# shell('Rscript estimate_EB_distr.r --data_path ../../Data/ --rollsignal_name OOS_signal_tstat_OosNyears1.csv.gzip --out_prefix Debug_')
# see Option Parsing section for details

# takes about 10-20 min

# Environment ========================================================
rm(list = ls())

## Packages ====
library(tidyverse)
library(data.table)
library(distr)
library(nloptr)
library(gridExtra)
library(optparse)

## Functions ====
# logit and logistic functions for ease of notation
# (https://ro-che.info/articles/2018-08-11-logit-logistic-r)
logit = qlogis
logistic = plogis

# function for creating theta object --
make_theta_object = function(par){
  if (par$distfam == 'norm'){
    theta_o = Norm(par$mu,exp(par$log_sig))

  } else if (par$distfam == 'mixnorm'){
    pa = logistic(par$logit_pa) # map (-inf,inf) to [0,1]
    theta_o = UnivarMixingDistribution(
      Norm(par$mua,exp(par$log_siga)),
      Norm(par$mub,exp(par$log_sigb)),
      mixCoeff = c(pa, 1-pa)
    )
  } else if (par$distfam == 'ngamma'){
    theta_o = par$loc-1*Gammad(exp(par$log_shape), exp(par$log_scale)) 
  } else {
    stop('distfam not recognized')
  }
  return(theta_o)
}

# function for converting theta par to natural units --
# (not used)
transform_par_units = function(par){
  parnat = data.frame(distfamily = par$distfam)
  parnat = if (par$distfam == 'norm'){
    parnat = parnat %>% mutate(mu = par$mu, sig = exp(par$log_sig))
  } else if (par$distfam == 'mixnorm'){
    parnat = parnat %>% mutate(
      mua = par$mua, siga = exp(par$log_siga), 
      mub = par$mub, sigb = exp(par$log_sigb), 
      pa = logistic(par$logit_pa)
    )
  } else if (par$distfam == 'ngamma'){
    parnat = parnat %>% mutate(loc = par$loc, shape = exp(par$log_shape), scale = exp(par$log_scale))
  } else {
    stop('distfam not recognized')
  }
  return = parnat
}

# function for estimating theta object via qml --
estimate_qml = function(par_guess, est_names, opt_set, tstatvec){

  # define loss function
  minme = function(parvec){

    # translate parvec to theta object
    par = par_guess
    par[ , est_names] = parvec %>% c() %>% t() %>% as.numeric
    theta_o = make_theta_object(par)

    # create t-stat object
    tstat.o = theta_o + Norm(0,1)

    # individual likes
    singlelikes = d(tstat.o)(tstatvec)

    # clean up and take mean
    singlelikes[singlelikes<=0] = 1e-6
    
    return = -1*sum(log(singlelikes))
  }

  # translate par_guess to parvec_guess
  parvec_guess = par_guess[ , est_names] %>% t() %>% as.numeric

  # test loss function
  minme(parvec_guess) 

  # optimize 
  opt = nloptr(
      x0 = parvec_guess
      , eval_f = minme
      , opts = opt_set
  )

  # translate back to par
  parhat = par_guess
  parhat[ , est_names] = opt$solution %>% c() %>% t() %>% as.numeric

  return = list(parhat = parhat, opt = opt, loglike = -1*opt$objective)

} # end function estimate_qml 

# function for shrinkage prediction of theta --
predict_theta = function(theta_o, tstat){

  # define joint density of theta and tstat
  f_joint = function(theta_temp,tstat){
    dnorm(tstat,theta_temp,1) * d(theta_o)(theta_temp) 
  }

  # calculate E(theta|tstat)
  numer = integrate(
    function(theta_temp) theta_temp*f_joint(theta_temp,tstat),
    -Inf, Inf)$value
  denom = integrate(
    function(theta_temp) f_joint(theta_temp,tstat),
    -Inf, Inf)$value

  # output 
  Etheta_Ctstat = numer/denom
  return = Etheta_Ctstat

} # end function predict_theta

# function for creating interpolant for thetahat --
make_thetahat_interpolant = function(parhat, tstatgrid=seq(-10,10,0.05)){
  # instead of evaluating the integrals for all 30,000 signals, 
  # we map out a grid and then interpolate
  thetahat_dat = data.table(tstat = tstatgrid)
  theta_o = make_theta_object(parhat)
  for (i in 1:nrow(thetahat_dat)){
    thetahat_dat[i, thetahat := predict_theta(theta_o, thetahat_dat[i, tstat])]
  }
  thetahat_fun = approxfun(thetahat_dat$tstat, thetahat_dat$thetahat)
  return = thetahat_fun
} # end function thetahat_interpolant

## Globals ====
# templates for par settings 
par_base_norm = data.frame(
  distfam = 'norm', mu = 0, log_sig = log(1)
)
par_base_mixnorm = data.frame(
  distfam = 'mixnorm', 
  mua =  0.5, log_siga = log(1), logit_pa = logit(0.5),
  mub = -0.5, log_sigb = log(1)
)
par_base_ngamma = data.frame(
  distfam = 'ngamma', loc = 1, log_shape = log(2), log_scale = log(0.5)
)

# nlopt settings
opt_set = list(algorithm = 'NLOPT_LN_BOBYQA', # use quadratic approx of obj 
                  maxeval = 1000,
                  xtol_rel = 1e-3,
                  ftol_rel = 1e-6,
                  print_level = 0)

# Option Parsing ========================================================

# command line options
cmd_option_list = list(
  make_option(c("-d", "--data_path"), type="character", default="../../Data/",
    help="Path to data directory"),
  make_option(c("-r", "--rollsignal_name"), type="character", default="OOS_signal_tstat_OosNyears1.csv.gzip",
    help="Name of roll signal file"),
  make_option(c("-o", "--out_prefix"), type="character", default="Debug_",
    help="Output prefix for saved files")
)
cmd_opt <- OptionParser(option_list = cmd_option_list) %>% parse_args()

# select distfams for model comparison
par_g_list = list(par_base_mixnorm, par_base_norm)

# select parameters that you want to estimate
# (could choose a subset manually)
for (i in 1:length(par_g_list)){
  if (i==1){par_name_list = list()}
  par_name_list[[i]] = names(par_g_list[[i]])[names(par_g_list[[i]])!= 'distfam'] # chooses everything
}

# Compute shrinkage for all family-years =======================================
# 2 sec per family-year, about 10 minutes total

# setup
rollsignal = fread(paste0(cmd_opt$data_path, cmd_opt$rollsignal_name))
rollsignal_is = rollsignal[s_flag=='IS',]
fam_yr_list = rollsignal %>% distinct(signal_family, oos_begin_year) %>% 
  arrange(signal_family, oos_begin_year)

# loop over all family-years
for (i in 1:nrow(fam_yr_list)){
  if (i==1){list.qml = list(); list.pred_ret = list()}

  tic = Sys.time()
  signalfam = fam_yr_list[i, signal_family]
  yr = fam_yr_list[i, oos_begin_year]
  print(paste(signalfam, yr, '===================='))

  # select data
  tempdat = rollsignal_is[signal_family==signalfam & oos_begin_year==yr, ]

  # estimate theta properties for each par_g_list item
  tempqmllist = list()
  for (iguess in 1:length(par_g_list)){
    tempqmllist[[iguess]] = estimate_qml(par_g_list[[iguess]], par_name_list[[iguess]], 
        opt_set, tempdat$tstat)
    # evaluate BIC
    tempqmllist[[iguess]]$bic = 0*length(par_name_list[[iguess]])*log(length(tempdat$tstat)) - 
      2*tempqmllist[[iguess]]$loglike
    # evaluate AIC (not used)
    tempqmllist[[iguess]]$aic = 2*length(par_name_list[[iguess]]) - 
      2*tempqmllist[[iguess]]$loglike
  }
  
  # find estimate with lowest BIC
  bicvec = sapply(tempqmllist, function(x) x$bic) 
  tempqml = tempqmllist[[which.min(bicvec)]]

  # compute shrinkage function
  thetahat_fun = make_thetahat_interpolant(tempqml$parhat)

  # evaluate thetahat_fun for each tstatdat$tstat
  tempdat[ , thetahat := thetahat_fun(tstat)][
    , pred_ret := thetahat*mean_ret/tstat]

  # feedback
  tempfeed = tempdat %>% filter(abs(tstat)>2) %>% 
    mutate(sign_ = sign(tstat)) %>% group_by(sign_) %>%
    summarise(pred_ret = mean(pred_ret), mean_ret = mean(mean_ret)) %>% 
    mutate(shrink = 1-pred_ret/mean_ret) 
  print(tempqml$parhat)
  print(paste0('shrinkage for |t|>2 = ', round(mean(tempfeed$shrink),3)))
  print(paste0('time for family-year = ', round(Sys.time()-tic,1)))

  # save
  # qml estimates
  list.qml[[i]] = tempqml$parhat %>% 
    mutate(opt_status = tempqml$opt$status,opt_iter = tempqml$opt$iterations, 
      opt_obj = tempqml$opt$objective,
      signal_family = signalfam, oos_begin_year = yr) %>% 
    select(signal_family, oos_begin_year, everything())

 
  # shrinkage predictions
  list.pred_ret[[i]] = tempdat 
  
} # end loop over family-years

# Save all estimates to disk ========================================================

# clean up list.qml
for (i in 1:length(list.qml)){
  if (i==1){list.qml2 = list()}
  tempq = list.qml[[i]]

  # get organized
  first_cols = c('signal_family', 'oos_begin_year', 'opt_status', 'opt_iter', 'opt_obj',
    'distfam')
  par_cols = setdiff(names(tempq), first_cols)

  # turn (mu, log_sig) to (par1, par2)
  temppar = tempq[par_cols]
  colnames(temppar) = paste0('par', 1:ncol(temppar))

  # make new data.table row
  list.qml2[[i]] = tempq[first_cols] %>% 
    cbind(data.frame(par_names = paste(par_cols, collapse = '|'))) %>% 
    cbind(temppar) %>% setDT()
    
} # end loop over list.qml

# bind 
family_year_qml = bind_rows(list.qml2) 
signal_year_predict = rbindlist(list.pred_ret)

# save
fwrite(family_year_qml, file = paste0(cmd_opt$data_path, cmd_opt$out_prefix, '_QML_FamilyYear.csv.gzip'))
fwrite(signal_year_predict, file = paste0(cmd_opt$data_path, cmd_opt$out_prefix, '_Predict_SignalYear.csv.gzip'))

# Debug: Output some plots ========================================================
# need to run Environment and options parsing first

# load data, if desired
# family_year_qml = fread(paste0(cmd_opt$data_path, 'QML_FamilyYear.csv.gzip'))
# signal_year_predict = fread(paste0(cmd_opt$data_path, 'Predict_SignalYear.csv.gzip'))

## Select a family-year
# family_year_qml %>% distinct(signal_family) %>% print()
signalfam = 'PastReturnSignalsLongShort_ew'
yr = 1987

# get theta par estimates
qml = family_year_qml[signal_family==signalfam & oos_begin_year==yr, ]

# extract names, translate par1, par2, to mu, sig, etc
par_names = qml$par_names %>% strsplit('\\|') 
par_names = par_names[[1]]
npar = length(par_names)

parhat = qml %>% select(distfam, starts_with('par')) %>% select(-par_names) 
parhat = parhat[ , 1:(npar+1)]
colnames(parhat) = c('distfam', par_names)

# get shrinkage predictions
tstatdat = signal_year_predict[signal_family==signalfam & oos_begin_year==yr, ]

# alternative estimate
par_guess = par_base_mixnorm
est_names = names(par_guess)[names(par_guess)!= 'distfam']

qml_alt = estimate_qml(par_guess, est_names, opt_set, tstatdat$tstat)
thetahat_fun_alt = make_thetahat_interpolant(qml_alt$parhat)
tstatdat[ , thetahat_alt := thetahat_fun_alt(tstat)][
  , pred_ret_alt := thetahat_alt*mean_ret/tstat]

## Plot ====

# plot t-stat fit
tedge = seq(-10,10,0.2)
tstathat_o = make_theta_object(parhat) + Norm(0,1)

plotme = data.table(
  tedge = tedge
  , Femp = ecdf(tstatdat$tstat)(tedge)
  , Fhat = p(tstathat_o)(tedge)
) 
if (parhat$distfam =='mixnorm'){
  # if mixnorm, also plot the two components
  tstata_o = Norm(parhat$mua,exp(parhat$log_siga))+Norm(0,1)
  tstatb_o = Norm(parhat$mub,exp(parhat$log_sigb))+Norm(0,1)
  pa = logistic(parhat$logit_pa)  
  plotme$Fhat_comp_a = pa*p(tstata_o)(tedge)
  plotme$Fhat_comp_b = (1-pa)*p(tstatb_o)(tedge)
} 
plotme$Fhat_alt = p(make_theta_object(qml_alt$parhat)+Norm(0,1))(tedge)

plotme = plotme %>% 
  melt(id.vars='tedge', variable.name='type', value.name='F') %>% 
  group_by(type) %>% mutate(dF = F-lag(F)) %>% ungroup() %>% 
  filter(!is.na(dF))

p_fit = plotme %>% ggplot(aes(x=tedge, y=dF, color=type,
  linetype=type)) +
  geom_line() +
  theme_bw() +
  labs(title='', x='Tstat', y='dF') 

# plot predictions by group
plotme = tstatdat %>% 
  mutate(bin = ntile(mean_ret, 20)) %>%
  group_by(bin) %>%
  summarise(
    tstat = mean(tstat),
    thetahat = mean(thetahat),
    pred_ret = mean(pred_ret),
    mean_ret = mean(mean_ret),
    pred_ret_alt = mean(pred_ret_alt)
  ) %>% 
  pivot_longer(-bin, names_to='name', values_to='value') %>% 
  filter(grepl('_ret', name))
p_shrink = plotme %>% ggplot(aes(x=bin, y=value, color=name, linetype=name,
  shape=name)) +
  geom_hline(yintercept=0, linetype='dashed') +
  geom_point() +
  geom_line() +
  theme_bw() 

# save plots
pboth = arrangeGrob(p_fit, p_shrink, ncol=1)
ggsave('deleteme.png', pboth, width=6, height=6)

## Some numbers ====
parhat %>% print()
tstatdat %>% 
  mutate(bin = ntile(mean_ret, 20)) %>%
  group_by(bin) %>%
  summarise(
    tstat = mean(tstat),
    thetahat = mean(thetahat),
    pred_ret = mean(pred_ret),
    mean_ret = mean(mean_ret)
  ) %>% 
  mutate(1-pred_ret/mean_ret) %>% 
  filter(bin >= 18 | bin <= 3) 


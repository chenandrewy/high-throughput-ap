#%% Setup ====================================
rm(list = ls())
library(tidyverse)
library(data.table)

# Read the QML parameter estimates file
qmlall = fread("../../Data/QML_FamilyYear.csv.gzip")

#%% Extract parameters =======================================
famselect = 'DataMinedLongShortReturnsEW'
yearselect = 1983

qml = qmlall[
    signal_family == famselect &
    oos_begin_year == yearselect
]

# extract parameters 
par_cols = grep('^par[0-9]$', names(qml), value = TRUE)
par = qml[,..par_cols] 
par_names = qml$par_names %>% strsplit('\\|') %>% unlist()
names(par) = par_names

# convert parameters to easier units
parclean = tibble(
    mua = par$mua,
    siga = exp(par$log_siga),
    pa = plogis(par$logit_pa),
    mub = par$mub,
    sigb = exp(par$log_sigb),
)


#%% Simulate parameters =========================================

nstrat = 29000
set.seed(920)
edge = seq(0,20,0.05)
mid = (edge[-1] + edge[-length(edge)])/2
h = 2

# simulate 
stratdat = data.table(
    i = 1:nstrat
    , compa = rnorm(nstrat, parclean$mua, parclean$siga)
    , compb = rnorm(nstrat, parclean$mub, parclean$sigb)
    , noise = rnorm(nstrat, 0, 1)
) %>% 
  mutate(
    theta = ifelse(i <= nstrat * parclean$pa, compa, compb)
    , tstat = theta + noise
  )

histdat = stratdat[abs(tstat) > h] %>% 
  mutate(mid = cut(abs(theta), breaks = edge, labels = mid)) %>% 
  group_by(mid) %>% 
  summarise(
    n = n()
  ) %>% 
  ungroup() %>% 
  mutate(
    pct = n / sum(n) * 100
  )

# plot
ggplot(histdat, aes(x = mid, y = pct)) +
  geom_line()
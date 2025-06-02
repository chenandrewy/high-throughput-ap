# %% Setup ====================================
rm(list = ls())
library(tidyverse)
library(data.table)

# Read the QML parameter estimates file
qmlall <- fread("../../Data/QML_FamilyYear.csv.gzip")

# %% Extract parameters =======================================
famselect <- "DataMinedLongShortReturnsVW"
yearselect <- 1983

qml <- qmlall[
    signal_family == famselect &
        oos_begin_year == yearselect
]

# extract parameters
par_cols <- grep("^par[0-9]$", names(qml), value = TRUE)
par <- qml[, ..par_cols]
par_names <- qml$par_names %>%
    strsplit("\\|") %>%
    unlist()
names(par) <- par_names

# convert parameters to easier units
parclean <- tibble(
    mua = par$mua,
    siga = exp(par$log_siga),
    pa = plogis(par$logit_pa),
    mub = par$mub,
    sigb = exp(par$log_sigb),
)


# %% Simulate parameters =========================================
nsim <- 2000
nstrat <- 29000
set.seed(920)
edge <- seq(0, 20, 0.25)
mid <- (edge[-1] + edge[-length(edge)]) / 2
h <- 3

# simulate
manyhistdat <- lapply(1:nsim, function(sim_id) {
    if (sim_id %% 100 == 0) {
        print(paste0("sim ", sim_id, " of ", nsim))
    }
    stratdat <- data.table(
        i = 1:nstrat,
        compa = rnorm(nstrat, parclean$mua, parclean$siga),
        compb = rnorm(nstrat, parclean$mub, parclean$sigb),
        noise = rnorm(nstrat, 0, 1)
    ) %>%
        mutate(
            theta = ifelse(i <= nstrat * parclean$pa, compa, compb),
            tstat = theta + noise
        )

    histdat <- stratdat[abs(tstat) > h] %>%
        mutate(
            mid = cut(abs(theta), breaks = edge, labels = mid),
            mid = mid %>% as.character() %>% as.numeric()
        ) %>%
        group_by(mid) %>%
        summarise(n = n()) %>%
        ungroup() %>%
        mutate(
            pct = n / sum(n) * 100,
            sim_id = sim_id
        )

    return(histdat)
}) %>% bind_rows()


manyhistdat %>% arrange(-mid)

# %% Plot ====
library(foreach)

MATBLUE <- rgb(0, 0.4470, 0.7410)
MATRED <- rgb(0.8500, 0.3250, 0.0980)
MATYELLOW <- rgb(0.9290, 0.6940, 0.1250)
MATPURPLE <- rgb(0.4940, 0.1840, 0.5560)
MATGREEN <- rgb(0.4660, 0.6740, 0.1880)

qlist <- c(0.75, 0.95, 0.99)

# Create aesthetics list
line_aes <- list(
    colors = c(
        "ptile_75" = MATYELLOW,
        "ptile_95" = MATRED,
        "ptile_99" = MATPURPLE
    ),
    linetypes = c(
        "ptile_75" = "solid",
        "ptile_95" = "dotdash",
        "ptile_99" = "longdash"
    ),
    labels = c(
        "ptile_75" = "75th Percentile",
        "ptile_95" = "95th Percentile",
        "ptile_99" = "99th Percentile"
    )
)

# Bar aesthetics
bar_aes <- list(
    fill = 'gray50',
    label = "Mean"
)

# find means
manysum_mean <- manyhistdat %>%
    mutate(stat = "mean") %>%
    group_by(stat, mid) %>%
    summarize(pct = mean(pct), .groups = "drop")

# find order stats
manysum_order0 <- foreach(q_val = qlist, .combine = rbind) %do% {
    manyhistdat %>%
        mutate(stat = paste0("ptile_", q_val * 100)) %>%
        group_by(stat, mid) %>%
        summarize(pct = quantile(pct, q_val), .groups = "drop")
}

manysum <- bind_rows(manysum_mean, manysum_order0)

# finally plot
p = manysum %>%
    # first plot means as bar
    filter(stat == "mean") %>%
    ggplot(aes(x = mid, y = pct)) +
    geom_bar(stat = "identity", aes(fill = "mean")) +
    # then plot order stats as line
    geom_line(
        data = manysum %>% filter(stat != "mean"),
        aes(x = mid, y = pct, color = stat, linetype = stat), size = 1
    ) +
    geom_point(
        data = manysum %>% filter(stat != "mean"),
        aes(x = mid, y = pct, color = stat), size = 2
    ) +
    scale_fill_manual(
        values = bar_aes$fill,
        labels = bar_aes$label
    ) +
    scale_color_manual(
        values = line_aes$colors,
        labels = line_aes$labels
    ) +
    scale_linetype_manual(
        values = line_aes$linetypes,
        labels = line_aes$labels
    ) +
    # plot horizonatal line at critical value
    geom_hline(yintercept = 5, color = 'black') +
    theme_minimal() +
    theme(
        legend.position = c(80,70)/100
        , legend.title = element_blank()
    ) +
    guides(
        fill = guide_legend(order = 1),
        color = guide_legend(order = 2),
        linetype = guide_legend(order = 2)
    ) +
    xlab("Actual Performance (scaled by S.E.)") +
    ylab("Percent of Strategies with |t-stat| > 3.0") +
    coord_cartesian(
        xlim = c(0, 6)
    )

ggsave('../../Paper/Figures/fdp-risk-demo.pdf', p, width = 6, height = 4, device = cairo_pdf)

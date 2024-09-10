# 2024 09 estimation of Romano-Shaikh-Wolf 2008 Econometric Theory FDP's FDP-StepM
# Same algo found in Chordia-Goyal-Saretto (2020); Harvey-Liu-Saretto (2020)

# Environment ========================================================
rm(list = ls())

## Packages ====
library(tidyverse)
library(data.table)
library(optparse)
library(foreach)
library(doParallel)

## Functions ====

# function for bootstrap
bootstrap_panel <- function(nboot, rmat, crossdat) {
    N <- nrow(rmat)
    T <- ncol(rmat)

    emat <- rmat - matrix(crossdat$rbar, N, T, byrow = FALSE)

    bootdat <- foreach(simi = 1:nboot, .combine = "rbind", .packages = "tidyverse") %do% {
        yearmboot <- sample(1:T, T, replace = TRUE)
        emat_cur <- emat[, yearmboot]
        ebar_cur <- rowMeans(emat_cur, na.rm = TRUE)
        nmonth_cur <- rowSums(!is.na(emat_cur))

        dat_cur <- data.table(
            simi = simi,
            id = rownames(rmat),
            t_orig = crossdat$tstat,
            tstat = ebar_cur / sqrt(rowMeans(emat_cur^2, na.rm = TRUE)) * sqrt(nmonth_cur),
            ebar = ebar_cur
        ) %>%
            setDT()

        return(dat_cur)
    } # end foreach simi

    # arrange nicely
    bootdat[, tabs := abs(tstat)]
    setorder(bootdat, simi, -tabs)
    bootdat[, rank := 1:.N, by = simi]

    return(bootdat)
} # end function bootstrap_panel


# Option Parsing / User Entry ========================================================

# command line options
cmd_option_list <- list(
    make_option(c("-r", "--panel_name"),
        type = "character", default = "DataMinedLongShortReturnsVW.csv",
        help = "Filename of signal-month returns"
    ),
    make_option(c("-o", "--out_prefix"),
        type = "character", default = "DebugRSW_",
        help = "Output prefix for saved files"
    ),
    make_option(c("-d", "--data_path"),
        type = "character", default = "../../Data/",
        help = "Path to data directory"
    )
)
cmd_opt <- OptionParser(option_list = cmd_option_list) %>% parse_args()

# Load Data =================================================

pan0 <- fread(paste0(cmd_opt$data_path, cmd_opt$panel_name))
pan <- pan0 %>% transmute(signalid, yearm = year * 100 + month, ret)


# Bootstrap =================================================
sampstart <- 196301
sampend <- sampstart + 2000 + 11
min_nmonth <- 12 * 5
nboot <- 100

# create matrix for bootstrap
signalkeep <- pan[yearm >= sampstart & yearm <= sampend,
    .(nmonth = .N),
    by = signalid
] %>%
    mutate(keep = nmonth >= min_nmonth)

rmat <- pan[yearm >= sampstart & yearm <= sampend &
    signalid %in% signalkeep[keep == TRUE]$signalid, ] %>%
    spread(key = yearm, value = ret)
# rownames(rmat) <- paste0("signal", rmat$signalid)
rownames(rmat) <- rmat$signalid
rmat <- as.matrix(rmat[, -1])

# cross-sectional summary
crossdat <- data.table(
    id = rownames(rmat),
    nmonth = rowSums(!is.na(rmat)),
    rbar = rowMeans(rmat, na.rm = TRUE),
    vol = sqrt(rowMeans(rmat^2, na.rm = TRUE))
) %>%
    mutate(tstat = rbar / vol * sqrt(nmonth))

tic <- Sys.time()
bootdat <- bootstrap_panel(nboot, rmat, crossdat)
toc <- Sys.time()
print(paste0("min to bootstrap: ", round(difftime(toc, tic, units = "mins"), 2), " nboot = ", nboot))


# Do a loop! =========================

statspar <- tibble(
    kmax = NULL,
    iterstep = 1,
    gamma = 0.05,
    alpha = 0.10,
    kstepM_itermax = 100,
    subsetmax = 1e3
)

# tbc: this statspar$kmax can be replaced with a closed form
# if  the highest k that would be needed
if (is.null(statspar$kmax)) {
    statspar$kmax <- statspar$gamma * (nrow(crossdat) + 1)
}


# though perhaps there should be a warning if the closed form implies too many iterations

# repeatedly increase the k in k-FWER
for (k in seq(1, statspar$kmax, by = statspar$iterstep)) {

    # tbc: add stepwise loop

    # initialize k-StepM
    disc <- c() # start with empty set

    # repeatedly add more discoveries
    for (j in 1:statspar$kstepM_itermax) {
        print(paste0("k = ", k, " j = ", j))

        # define test set (signals not declared discoveries)
        testme <- setdiff(crossdat$id, disc)
        
        # break if infeasible
        num_subset <- choose(length(disc), k-1)
        if (num_subset > statspar$subsetmax) {
            print(paste0("Infeasible: num_subset = ", num_subset, " > ", statspar$subsetmax))

            discdat = list(
                disc = disc,
                hurdle = h,
                jstep = j,
                break_reason = "num_subset > subsetmax"
            )

            break
        }

        # define the set of discovered subsets to check
        if (is.null(disc) | k == 1) {
            # if no discoveries or k=1, use empty set
            disc_sub_list <- list(c())
        } else {
            disc_sub_list <- combn(disc, k - 1) %>% t()
            disc_sub_list <- split(disc_sub_list, row(disc_sub_list))
        }            

        # loop over subsets
        h_list <- array(NA, length(disc_sub_list))
        for (subi in 1:length(disc_sub_list)) {
            # find hurdle based on testme union a subset of discoveries
            testme_plus <- c(testme, disc_sub_list[[subi]])
            t_kmax_dat <- bootdat[id %in% testme_plus & rank == k]
            h_list[subi] <- quantile(t_kmax_dat$tabs, 1 - statspar$alpha)
        }

        disc_sub_list
        is.null(disc)
        
        # use the worst case from h_list
        h <- max(h_list)

        # find new discoveries
        disc_new <- crossdat[id %in% testme & abs(tstat) > h]$id

        # if no new discoveries, then break
        if (length(disc_new) == 0) {
            discdat = list(
                disc = disc,
                hurdle = h,
                jstep = j,
                break_reason = "no new discoveries"
            )
            break
        }

        # update disc
        disc <- c(disc, disc_new)
    
    }   # end j loop

    stop_cond <- (statspar$gamma < k / (length(disc) + 1))
    if (stop_cond) {
        print(paste0("Stopping at k = ", k))
        print(paste0("gammahat = ", k / (length(disc) + 1)))
        print(paste0("Num discoveries: ", length(discdat$disc)))
        print(paste0("hurdle = ", discdat$hurdle))
        print(paste0("j iter = ", discdat$j))
        print(paste0("Break condition: ", discdat$break_reason))
        break
    }
} # end k in k-FWER loop




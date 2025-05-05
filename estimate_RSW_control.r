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


library(matrixStats)
library(future.apply)
library(fst)
library(future.apply)

## Functions ========================================================

# for debugging convenience
headm = function(x) {
    ncolmax = min(ncol(x), 10)
    x[ , 1:ncolmax] %>% head() %>% print()
}

# lsos memory check
.ls.objects <- function (pos = 1, pattern, order.by,
                        decreasing=FALSE, head=FALSE, n=5) {
    napply <- function(names, fn) sapply(names, function(x)
                                         fn(get(x, pos = pos)))
    names <- ls(pos = pos, pattern = pattern)
    obj.class <- napply(names, function(x) as.character(class(x))[1])
    obj.mode <- napply(names, mode)
    obj.type <- ifelse(is.na(obj.class), obj.mode, obj.class)
    obj.size <- round(napply(names, object.size) / (1024^2), 2) # Convert to MB
    obj.dim <- t(napply(names, function(x)
                        as.numeric(dim(x))[1:2]))
    vec <- is.na(obj.dim)[, 1] & (obj.type != "function")
    obj.dim[vec, 1] <- napply(names, length)[vec]
    out <- data.frame(obj.type, obj.size, obj.dim)
    names(out) <- c("Type", "Size (MB)", "Rows", "Columns")
    if (!missing(order.by))
        out <- out[order(out[[order.by]], decreasing=decreasing), ]
    if (head)
        out <- head(out, n)
    out
}
# shorthand
lsos <- function(..., n=10) {
    .ls.objects(..., order.by="Size (MB)", decreasing=TRUE, head=TRUE, n=n)
}

# panel bootstrap (not used)
bootstrap_slow <- function(retdat, nboot) {

    # create matrix for bootstrap
    rmat <- retdat %>%
        spread(key = yearm, value = ret)
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
            
    N <- nrow(rmat)
    T <- ncol(rmat)

    emat <- rmat - matrix(crossdat$rbar, N, T, byrow = FALSE)

    bootdat <- foreach(simi = 1:nboot, .combine = "rbind", .packages = "tidyverse") %do% {

        # feedback
        if (simi %% 100 == 0) print(paste0("bootstrap panel: ", simi, " of ", nboot))

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

bootstrap_chunk = function(retdat, nbootchunk) {
    # fast bootstrap for a single chunk (courtesey of o3)
    # (but painfully hand-checked by andrew)

    # create matrix for bootstrap
    rmat <- retdat %>%
        spread(key = yearm, value = ret)
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

    # pre-computations
    N  <- nrow(rmat)
    TT <- ncol(rmat)

    emat   <- rmat - matrix(crossdat$rbar, N, TT, byrow = FALSE)

    imat   <- !is.na(emat)                # indicator of non-NA (logical)
    emat0  <- ifelse(imat, emat, 0)       # NA → 0  for fast BLAS sums
    emat02 <- emat0^2                     # needed for mean of squares

    id_vec <- rownames(emat0)
    t_orig <- crossdat$tstat

    # month_id is a TT x nbootchunk matrix of month indices
    month_id <- sample.int(TT, TT * nbootchunk, replace = TRUE)
    dim(month_id) <- c(TT, length(month_id) / TT)        

    # convert month_id to a matrix of counts
    # w[i,b] is the number of times month i appears in bootstrap draw b
    w        <- apply(month_id, 2L, tabulate, nbins = TT)   
    w        <- matrix(as.numeric(w), TT)

    # number of return observations
    n_obs <- (imat %*% w)                # N x nbootchunk
    n_obs_dbl <- pmax(n_obs, 1)          # for handling division by zero

    # find moments (with n_obs == 0 set to 0)
    ebar    <- (emat0 %*% w) / n_obs_dbl
    msqbar  <- (emat02 %*% w) / n_obs_dbl
    tstat <- ebar / sqrt(msqbar) * sqrt(n_obs)

    # set     
    ebar[n_obs == 0]   <- NA_real_
    msqbar[n_obs == 0] <- NA_real_
    tstat[n_obs == 0] <- NA_real_

    # if you're paranoid, you can check with the following:
    # moncheck = month_id[ , 1]
    # echeck = emat[ , moncheck]
    # ebarcheck = rowMeans(echeck, na.rm = TRUE)
    # vcheck = sqrt(rowMeans(echeck^2, na.rm = TRUE))
    # ncheck = rowSums(!is.na(echeck))
    # tcheck = ebarcheck / vcheck * sqrt(ncheck)    

    ## assemble output
    bootdat <- data.table(
        simi  = rep(seq_len(ncol(w)), each = N),
        id    = rep.int(id_vec, times = ncol(w)),
        t_orig= rep.int(t_orig, times = ncol(w)),
        tstat = as.vector(tstat),
        ebar  = as.vector(ebar)
    )

    # arrange nicely
    bootdat[, tabs := abs(tstat)]
    setorder(bootdat, simi, -tabs)
    bootdat[, rank := seq_len(.N), by = simi]    

    return(bootdat)
}

bootstrap_fast = function(retdat, nboot, nbootchunk = 1000){
    # runs bootstrap_chunk many times and combines the results
    
    # run bootstrap_chunk nchunk times
    nchunk = ceiling(nboot / nbootchunk)
    bootdat = foreach(i = 1:nchunk, .combine = "rbind", .packages = "tidyverse") %do% {
        print(paste0("bootstrap chunk: ", i, " of ", nchunk))
        bootstrap_chunk(retdat, nbootchunk) %>%
            mutate(chunk = i)
    }

    # renumber the bootstraps
    bootdat[ , booti := simi + (chunk - 1) * nbootchunk]
    bootdat[ , c('simi', 'chunk') := NULL]
    setcolorder(bootdat, c('booti', 'id', 't_orig', 'tstat', 'ebar', 'tabs', 'rank'))

    # keep only the first nboot bootstraps
    bootdat = bootdat[booti <= nboot]

    return(bootdat)
} # end bootstrap_fast

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

# Create retdat: df of returns used to find good signals ===========================
sampstart <- 196301
sampend <- sampstart + 2000 + 11
min_nmonth <- 12 * 5

retdat <- fread(paste0(cmd_opt$data_path, cmd_opt$panel_name))
retdat <- retdat %>% transmute(signalid, yearm = year * 100 + month, ret) %>% 
    filter(yearm >= sampstart & yearm <= sampend)
signalkeep <- retdat[ , .(nmonth = .N), by = signalid] %>%
    mutate(keep = nmonth >= min_nmonth)
retdat <- retdat[signalid %in% signalkeep[keep == TRUE]$signalid]

# Run Bootstrap =================================================
nboot = 10000
# 2 minutes for 10,000 bootstraps => 8 hours for 40 x 6 

tic <- Sys.time()
bootdat = bootstrap_fast(retdat, nboot = nboot, nbootchunk = 1000)
toc <- Sys.time()
print(paste0("min to bootstrap: ", round(difftime(toc, tic, units = "mins"), 2), " nboot = ", nboot))

# construct crossdat
crossdat = bootdat[booti == 1] %>% 
    transmute(id, tstat = t_orig)

# Do the RSW thing =========================

# if num_subset = 1700, this can take a very long time, > 10 minutes for just one j step

statspar <- tibble(
    kmax = NULL,
    iterstep = 1,
    gamma = 0.05,
    alpha = 0.10,
    kstepM_itermax = 100,
    subsetmax = 1e4,
    feedback = TRUE
)

# ensures Pr(FDP > gamma) <= alpha
# implicitly uses crossdat and bootdat

# kmax = null, find the highest k that would be needed
if (is.null(statspar$kmax)) {
    statspar$kmax <- floor(statspar$gamma * (nrow(crossdat) + 1)) + 1
}

# do FDP-stepM: repeatedly increase the k in k-FWER
tic = Sys.time()
for (k in seq(1, statspar$kmax, by = 1)) {
    # initialize k-StepM
    disc <- c() # start with empty set (aka R_j)

    # do k-stepM: repeatedly add more discoveries
    for (j in 1:statspar$kstepM_itermax) {


        if ((j > 1) && (k > 1)) browser()         # debug

        # define test set (signals not declared discoveries, aka A_j)
        testme <- setdiff(crossdat$id, disc)


        ## check feasibility
        #   for length(disc) = 60, k = 3, the number of subsets to check is 1700
        #   and is not very feasible.
        #   so to be feasible, if k >= 3, we will typically need to stop at j = 1
        num_subset <- choose(length(disc), k - 1)

        # feedback
        if (statspar$feedback) {
            print(paste0("k = ", k,
                ", j = ", j,
                ", num_subset = ", num_subset))
        }

        # break if infeasible
        if (num_subset > statspar$subsetmax) {
            if (statspar$feedback) {
                print(paste0("Infeasible: num_subset = ", num_subset, " > ", statspar$subsetmax))
            }

            discdat <- list(
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
        tic = Sys.time()
        for (subi in 1:length(disc_sub_list)) {
            # find hurdle based on testme union a subset of discoveries
            testme_plus <- c(testme, disc_sub_list[[subi]]) # K = A_j union I
            t_kmax_dat <- bootdat[id %in% testme_plus & rank == k]  # k-max(T_{n,i}: i \in K)
            h_list[subi] <- quantile(t_kmax_dat$tabs, 1 - statspar$alpha) # hat{c}_{n,K}(1-alpha,K)
        }
        toc = Sys.time()
        print(paste0("min to find h: ", round(difftime(toc, tic, units = "mins"), 2)))

        bootdat_kmax = bootdat[rank == k] # pre-compute the k-largest tstat in each bootstrap


        # use the worst case from h_list
        h <- max(h_list)

        # find new discoveries
        disc_new <- crossdat[id %in% testme & abs(tstat) > h]$id

        if (statspar$feedback) {
            print(paste0(
                "  worst case h = ", round(h, 2),
                ", j loop new discoveries = ", length(disc_new)
            ))
        }        

        # if no new discoveries, then break
        if (length(disc_new) == 0) {
            discdat <- list(
                disc = disc,
                hurdle = h,
                jstep = j,
                break_reason = "no new discoveries"
            )
            break
        }

        # update disc
        disc <- c(disc, disc_new)
    } # end j loop

    stop_cond <- k / (length(disc) + 1) > statspar$gamma
    if (stop_cond) {
        if (statspar$feedback) {
            print(paste0("Stopping at k = ", k))
            print(paste0("gammahat = ", k / (length(disc) + 1)))
            print(paste0("Num discoveries: ", length(discdat$disc)))
            print(paste0("hurdle = ", discdat$hurdle))
            print(paste0("j iter = ", discdat$j))
            print(paste0("j break condition: ", discdat$break_reason))
            break
        }
    }
} # end k in k-FWER loop

toc = Sys.time()
print(paste0("min to FDP-stepM: ", round(difftime(toc, tic, units = "mins"), 2)))

discdat
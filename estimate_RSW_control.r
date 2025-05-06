# 2024 09 estimation of Romano-Shaikh-Wolf 2008 Econometric Theory FDP's FDP-StepM
# (Mostly) same algo found in Chordia-Goyal-Saretto (2020); Harvey-Liu-Saretto (2020)

# time required depends a lot on the size of tstats in the data
# VW returns => small tstats => 

# Environment ========================================================
rm(list = ls())
tic0 = Sys.time()

## Packages ====
library(tidyverse)
library(data.table)
library(optparse)
library(foreach)

## Option Parsing  ========================================================

# command line options
cmd_option_list <- list(
    make_option(c("--data_path"),
        type = "character", default = "../../Data/"
    ),    
    make_option(c("--signal_data"),
        type = "character", default = "pastret",
        help = c("acct, pastret, or ticker")
    ),
    make_option(c("--stock_weight"),
        type = "character", default = "vw",
        help = c("ew or vw")
    ),
    make_option(c("--out_path"),
        type = "character", default = "../../Data/RSW_result/",
    ),
    make_option(c("--out_prefix"),
        type = "character", default = "DebugRSW_",
    ),
    make_option(c("--sampstart"),
        type = "integer", default = 196301
    ),
    make_option(c("--sampend"),
        type = "integer", default = 198312
    ),
    make_option(c("--min_nmonth"),
        type = "integer", default = 60,
        help = "Drop signals with less than min_nmonth months of data"
    ),
    make_option(c("--nboot"),
        type = "integer", default = 2000,
        help = "Number of bootstraps"
    ),
    make_option(c("--alpha"),
        type = "numeric", default = 0.10,
        help = "P(FDP>gamma)<alpha"
    ),
    make_option(c("--gamma"),
        type = "numeric", default = 0.05,
        help = "P(FDP>gamma)<alpha"
    ),
    make_option(c("--kstepM_itermax"),
        type = "integer", default = 100,
        help = "Max number of iterations for k-stepM"
    ),
    make_option(c("--subsetmax"),
        type = "integer", default = 100, # honestly setting this above 10 doesn't change anything
        help = "Max number of subsets for k-stepM"
    )
)
opt <- OptionParser(option_list = cmd_option_list) %>% parse_args()

# hard coded
nbootchunk = 1000


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

bootstrap_chunk = function(retdat, nbootchunk) {
    # fast bootstrap for a single chunk (courtesey of o3)
    # (but painfully hand-checked by andrew)

    # create matrix for bootstrap
    rmat <- retdat[ ,.(id, yearm, ret)] %>% spread(key = yearm, value = ret)
    rownames(rmat) <- rmat$id
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
        chunkdat = bootstrap_chunk(retdat, nbootchunk)
        chunkdat[, chunk := i]
        chunkdat[, .(chunk, simi, id, tabs)]
    }

    # renumber the bootstraps
    bootdat[ , booti := simi + (chunk - 1) * nbootchunk]
    bootdat[ , c('simi', 'chunk') := NULL]
    setcolorder(bootdat, c('booti', 'id', 'tabs'))

    # keep only the first nboot bootstraps
    bootdat = bootdat[booti <= nboot]

    return(bootdat)
} # end bootstrap_fast


# Create retdat: df of returns used to find good signals ===========================

if (opt$signal_data == "acct") {
    # acct data has a different format
    retfile = paste0('DataMinedLongShortReturns', toupper(opt$stock_weight), '.csv')
    retdat <- fread(paste0(opt$data_path, retfile))
} else {
    
    # pastret and ticker data have the same format
    if (opt$signal_data == "pastret") {
        retfile = 'PastReturnSignalsLongShort.csv.gzip'
    } else if (opt$signal_data == "ticker") {
        retfile = 'TickerSignalsLongShort.csv.gzip'
    }
    retdat <- fread(paste0(opt$data_path, retfile))

    # select stock weight
    if (opt$stock_weight == "ew") {
        setnames(retdat, 'ret_ew', 'ret')
    } else if (opt$stock_weight == "vw") {
        setnames(retdat, 'ret_vw', 'ret')
    }

    # match acct data format
    retdat[ , ':='(year = year(date), month = month(date))]
} # end if signal_data

retdat <- retdat %>% transmute(signalid, yearm = year * 100 + month, ret) %>% 
    filter(yearm >= opt$sampstart & yearm <= opt$sampend)
signalkeep <- retdat[ , .(nmonth = .N), by = signalid] %>%
    mutate(keep = nmonth >= opt$min_nmonth)
retdat <- retdat[signalid %in% signalkeep[keep == TRUE]$signalid]

# construct crossdat and rename id to match ranking ================================
# id is a new signal id that is the rank of the signal based on tstat
crossdat = retdat[, .(
    nmonth = sum(!is.na(ret)), 
    rbar = mean(ret, na.rm = TRUE), 
    vol = sd(ret, na.rm = TRUE)
    ), by = signalid
] %>% 
mutate(tstat = rbar / vol * sqrt(nmonth),
    tabs = abs(tstat)) %>% 
    arrange(-tabs) %>% 
    transmute(signalid, id = 1:n(), tabs, tstat, nmonth, rbar, vol)

# merge on id
retdat[crossdat, on = .(signalid), id := i.id]
setcolorder(retdat, c('id','yearm','ret','signalid'))

# Run Bootstrap =================================================
tic <- Sys.time()   
bootdat = bootstrap_fast(retdat, nboot = opt$nboot, nbootchunk = nbootchunk)
toc <- Sys.time()
print(paste0("min to bootstrap: ", round(difftime(toc, tic, units = "mins"), 2), " nboot = ", opt$nboot))

# Do the RSW thing =========================

# maximum k needed
    # PFDP control comes from achieving kFWER control and 
    # finding k st k > gamma * num_discoveries
    # So the largest k needed is if every signal is a discovery
kmax = floor(opt$gamma * (nrow(crossdat) + 1)) + 1 

# do FDP-stepM: repeatedly increase the k in k-FWER
tic = Sys.time()
setorder(bootdat, booti, -tabs) # sort bootdat
for (k in seq(1, kmax, by = 1)) {

    # initialize k-StepM        
    disc_id = c() # no discoveries yet    

    # do k-stepM: repeatedly add more discoveries
    for (j in 1:opt$kstepM_itermax) {        

        # create test set bootstrap
        testboot = bootdat[!id %in% disc_id]

        if ((j == 1) | (k==1)) {
            # RW2007 Alg 2.1 Step j = 1

            # find critical value based on ids in test set
            kmaxboot = testboot[ , .SD[k], by = booti] 
            h = quantile(kmaxboot$tabs, 1 - opt$alpha)         
            
        } else {
            # RW2007 Alg 2.1 Step j > 1

            # check feasibility
            num_subsets = choose(length(disc_id), k-1)
            if (num_subsets > opt$subsetmax) {
                print(paste0('num_subsets = ', num_subsets, ' > subsetmax = ', opt$subsetmax, ' going to next k'))
                break
            }        

            # find new hurdle
            disc_id_subsets = combn(disc_id, k-1)
            hlist = numeric(num_subsets)*NA
            for (subi in 1:num_subsets) {
                # find hurdle from an augmented test set (\hat{c}_{n,K} for K = A_j \cup I)
                # this loop is infuriating because it's slow and essentially every single subi
                # produces the same hurdle (see below)
                
                disc_id_subset = disc_id_subsets[ , subi] # I
                temp = testboot %>% rbind(bootdat[id %in% disc_id_subset]) # K = A_j \cup I
                setorder(temp, booti, -tabs)
                kmaxboot = temp[ , .SD[k], by = booti]

                # find critical value based on ids in test set
                hlist[subi] = quantile(kmaxboot$tabs, 1 - opt$alpha) # \hat{c}_{n,K}

                print(paste0('subi = ', subi, ' h = ', round(hlist[subi], 2), ' of ', num_subsets))

                # compare with unaugmented (almost always the same)
                # kmaxboot_unaug = testboot[ , .SD[k], by = booti]
                # h_unaug = quantile(kmaxboot_unaug$tabs, 1 - opt$alpha)
                # print(paste0('unaug h = ', h_unaug))
            } # end for subi

            h = max(hlist) # \hat{d}_{n,A_j}         

        } # end if j > 1

        # declare new discoveries based on critical value 
        disc_id_new = setdiff(crossdat[tabs > h]$id, disc_id)             

        # if no new discoveries, go to next k
        if (length(disc_id_new) == 0) {
            break
        }        

        # update discoveries
        disc_id = c(disc_id, disc_id_new)

        print(paste0("k = ", k, ", j = ", j, ", h = ", round(h, 2), ", num_disc = ", length(disc_id)))

    } # end j loop

    stop_cond <- k / (length(disc_id) + 1) > opt$gamma
    print(paste0("FDPhat = ", round(k / (length(disc_id) + 1), 3)))
    if (stop_cond) {
        print(paste0("Stopping at k = ", k))
        print(paste0("gammahat = ", round(k / (length(disc_id) + 1), 2)))
        print(paste0("Num discoveries: ", length(disc_id)))
        print(paste0("hurdle = ", round(h, 2)))
        print(paste0("j iter = ", j))
        print(paste0("j break condition: ", stop_cond))

        break
    }
} # end k in k-FWER loop

toc = Sys.time()
print(paste0("min to FDP-stepM: ", round(difftime(toc, tic, units = "mins"), 2)))

# Assemble and save results ========================================================

RSW_result = tibble(
    panel_name = opt$panel_name,
    sampstart = opt$sampstart,
    sampend = opt$sampend,
    h = h,
    alpha = opt$alpha,
    gamma = opt$gamma,
    k_last = k,
    j_last = j,
    num_disc = length(disc_id)
)

# create outpath if it doesn't exist
if (!dir.exists(opt$out_path)) {
    dir.create(opt$out_path)
}

outname = paste0(opt$out_path, opt$out_prefix, 
    opt$signal_data, "_", 
    opt$stock_weight, "_", 
    opt$sampstart, "_", opt$sampend, "_", 
    ".csv") %>% 
    print()

fwrite(RSW_result, outname)

toc0 = Sys.time()
print(paste0("min for estimate_RSW_control: ", round(difftime(toc0, tic0, units = "mins"), 2)))
outname
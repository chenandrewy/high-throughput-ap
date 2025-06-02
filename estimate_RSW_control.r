# 2024 09 estimation of Romano-Shaikh-Wolf 2008 Econometric Theory FDP's FDP-StepM
# (Mostly) same algo found in Chordia-Goyal-Saretto (2020); Harvey-Liu-Saretto (2020)

# to run:
#   Rscript estimate_RSW_control.r --data_path "../../Data/" --signal_data "ticker" --stock_weight "ew" --sampstart 196301 --sampend 198312 --min_nmonth 60 --nboot 2000 --alpha 0.10 --gamma 0.05 --kstepM_itermax 200 --subsetmax 100 --bisect_itermax 200

# must runs take 60 seconds, but time required may have fat tails
#   entering the ridiculous "check all subsets" loop adds a ton of time (and little information)
#   so should consider --subsetmax 20 as a starting point


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
        type = "character", default = "ticker",
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
    ),
    make_option(c("--bisect_itermax"),
        type = "integer", default = 200,
        help = "Max number of iterations for bisection"
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

bootstrap_fast = function(retdat, nboot, nbootchunk = 1000, seed = 1206){
    # runs bootstrap_chunk many times and combines the results

    set.seed(seed)
    
    # run bootstrap_chunk nchunk times
    nchunk = ceiling(nboot / nbootchunk)
    bootdat = foreach(i = 1:nchunk, .combine = "rbind", .packages = "tidyverse") %do% {
        print(sprintf("bootstrap chunk: %d of %d", i, nchunk))
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

run_kstepM = function(k, alpha, subsetmax, kstepM_itermax) {
    # implicitly takes bootdat as input

    # initialize k-StepM        
    disc_id = c() # no discoveries yet    

    # do k-stepM: repeatedly add more discoveries
    for (j in 1:kstepM_itermax) {        

        # create test set bootstrap
        testboot = bootdat[!id %in% disc_id]

        if ((j == 1) | (k==1)) {
            # RW2007 Alg 2.1 Step j = 1

            # find hurdle
            kmaxboot = testboot[ , .SD[k], by = booti] 
            h = quantile(kmaxboot$tabs, 1 - alpha)         
            
        } else {
            # RW2007 Alg 2.1 Step j > 1 (and implicitly k > 1)

            # check feasibility
            num_subsets = choose(length(disc_id), k-1)
            if (num_subsets > subsetmax) {
                print(sprintf('num_subsets = %.2e > subsetmax = %d, stopping k-stepM', num_subsets, subsetmax))
                break
            } else if ((k-1) > length(disc_id)) {
                # strange case not mentioned in RW 2007 (?)
                # only occurs if j > 1, so we can still control kFWER
                print(sprintf('k-1 = %d > length(disc_id) = %d, stopping k-stepM', k-1, length(disc_id)))
                break
            }

            # find hurdle
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
                hlist[subi] = quantile(kmaxboot$tabs, 1 - alpha) # \hat{c}_{n,K}

                print(sprintf('subi = %d, h = %.2f of %d', subi, hlist[subi], num_subsets))

                # compare with unaugmented (almost always the same)
                # kmaxboot_unaug = testboot[ , .SD[k], by = booti]
                # h_unaug = quantile(kmaxboot_unaug$tabs, 1 - alpha)
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

        print(sprintf("k = %d, j = %d, h = %.2f, num_disc = %d, gammaplus = %.3f", k, j, h, length(disc_id), k / (length(disc_id) + 1)))

    } # end j loop

    kstepMout = tibble(
        h = h, 
        num_disc = length(disc_id),
        j_last = j,
        gammaplus = k / (num_disc + 1)
    )

    return(kstepMout)

} # end run_kstepM

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
print(sprintf("min to bootstrap: %.2f, nboot = %d", difftime(toc, tic, units = "mins"), opt$nboot))

# Do the RSW thing =========================

# maximum k needed
    # PFDP control comes from achieving kFWER control and 
    # finding k st k > gamma * num_discoveries
    # So the largest k needed is if every signal is a discovery
kmax = floor(opt$gamma * (nrow(crossdat) + 1)) + 1 

# do a bisection version of FDP-stepM: 
#   find the largest k such that k / (num_disc + 1) <= gamma
tic = Sys.time()
setorder(bootdat, booti, -tabs) # sort bootdat

# interval check
klo = 1
khi = floor(opt$gamma * (nrow(crossdat) + 1)) + 1 
outlo = run_kstepM(klo, opt$alpha, opt$subsetmax, opt$kstepM_itermax)
outhi = run_kstepM(khi, opt$alpha, opt$subsetmax, opt$kstepM_itermax)

if (outlo$gammaplus > opt$gamma) {

    # if outlo$gammaplus > opt$gamma, set h to the max tstat, and declare no discoveries
    result = tibble(
        h = max(crossdat$tabs) + 1,
        num_disc = 0,
        j_last = NA_integer_,
        gammaplus = NA_real_,
        k_last = NA_integer_
    )

} else if (outhi$gammaplus < opt$gamma) {
    # if outhi$gammaplus < opt$gamma, then no need to bisect, use the resulting h
    result = outhi %>% mutate(k_last = khi)
} else {
    # here we have a bracket, so we bisect
    for (iter in 1:opt$bisect_itermax) {

        # evaluate the midpoint
        # bias toward higher k => higher floor on FDP
        kmid = floor((klo + khi) / 2) + 1

        # stop if kmid == khi
        if (kmid == khi) {
            print(sprintf('optimal k found: k = %d', kmid))
            result = outhi %>% mutate(k_last = kmid)
            break
        }

        # check midpoint        
        outmid = run_kstepM(kmid, opt$alpha, opt$subsetmax, opt$kstepM_itermax)

        # update bracket
        if (outmid$gammaplus > opt$gamma) {
            # gammaplus too high => lower k
            khi = kmid
            outhi = outmid # bias toward higher k
        } else {
            # gammaplus too low => raise k
            klo = kmid
        }

        print(sprintf('bisecting: iter = %d, kmid = %d, gammaplus = %.3f', iter, kmid, outmid$gammaplus))
    } # end for iter

} # end if we have a bracket

toc = Sys.time()
print(sprintf("min to FDP-stepM: %.2f", difftime(toc, tic, units = "mins")))


# Assemble and save results ========================================================

RSW_result = tibble(
    panel_name = opt$panel_name,
    sampstart = opt$sampstart,
    sampend = opt$sampend,
    h = result$h,
    alpha = opt$alpha,
    gamma = opt$gamma,
    k_last = result$k_last,
    j_last = result$j_last,
    num_disc = result$num_disc,
    gammaplus = result$gammaplus
)

# create outpath if it doesn't exist
if (!dir.exists(opt$out_path)) {
    dir.create(opt$out_path)
}

outname = sprintf("%s%s%s_%s_%d_%d_.csv",
    opt$out_path, opt$out_prefix,
    opt$signal_data, opt$stock_weight,
    opt$sampstart, opt$sampend)
print(sprintf('writing to %s', outname))

fwrite(RSW_result, outname)

toc0 = Sys.time()
print(sprintf("min for estimate_RSW_control: %.2f", difftime(toc0, tic0, units = "mins")))
outname
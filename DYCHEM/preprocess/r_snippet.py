'''
This R code is adopted from https://github.com/bsouhaib/sparseHTS
'''


def sim():

    sim = """
    simulate_hts <- function(n_simul){
      n_warm <- 300
      p <- 2
      d <- 0
      q <- 1
      if(FALSE){
        phi_2 <- runif(1, min = 0.5, max = 0.7)
        phi_1 <- runif(1, min = phi_2 - 1, max = 1 - phi_2)
        theta_1 <- runif(1, min = 0.5, max = 0.7)
      }else{
        phi_2 <- runif(1, min = 0.5, max = 0.7)
        phi_1 <- runif(1, min = phi_2 - 0.71, max = 0.71 - phi_2)
        theta_1 <- runif(1, min = 0.5, max = 0.7)
      }

      PHI <- c(phi_1, phi_2)
      THETA <- theta_1

      mus <- rep(0, 4)

      if(FALSE){
        Sigma <- rbind(c(5, 3, 2, 1), c(3, 4, 2, 1), c(2, 2, 5, 3), c(1, 1, 3, 4))
      }else{
        #varVec <- rep(1, 4) 
        varVec <- rep(2, 4)
        corMatB <- rbind(c(1, 0.7, 0.2, 0.3), 
                       c(0.7, 1, 0.3, 0.2),
                       c(0.2, 0.3, 1, 0.6),
                       c(0.3, 0.2, 0.6, 1))
        Sigma <- as.matrix(Diagonal(x = sqrt(varVec)) %*% corMatB %*% Diagonal(x = sqrt(varVec)))
      }

      Ematrix_insample <- mvrnorm(n = n_simul, mus, Sigma = Sigma)
      Ematrix_start <- mvrnorm(n = n_warm, mus, Sigma = Sigma)

      bts <- matrix(NA, nrow = n_simul, ncol = 4)
      for(j in seq(4)){
        bts[, j] <- arima.sim(n = n_simul, list(order = c(p, d, q), ar = PHI, ma = THETA), 
        n.start = n_warm, start.innov = Ematrix_start[, j], innov = Ematrix_insample[, j])
      }

      #bts <- tail(bts, -n_warm)
      list(bts = bts, A = rbind(c(1, 1, 1, 1), c(1, 1, 0, 0), c(0, 0, 1, 1)))
    }
    """

    return sim


def sim_large():
    sim_large = """
    ordergenCor <- function(n.bot)
    {

      #order.diff <- sample(0:1, n.bot, replace = TRUE)
      order.diff <- sample(0, n.bot, replace = TRUE)
      order.ar <- sample(0:2, n.bot, replace = TRUE)
      order.ma <- sample(0:2, n.bot, replace = TRUE)
      order.d <- cbind(order.ar, order.diff, order.ma)


      ar.d <- matrix(, n.bot, 2)
      ma.d <- matrix(, n.bot, 2)

      for(j in 1:n.bot)
      {
        order.t <- order.d[j, ]

        # define AR coefficients
        ar.coeff <- c()
        if(order.t[1]==0)
        {
          ar.d[j, 1] <- NA
        }

        ar.coeff1 <- 0
        ar.coeff2 <- 0
        if(order.t[1]==1)
        {
          ar.coeff1 <- runif(1, 0.5, 0.7)
          ar.coeff <- ar.coeff1
          ar.d[j, 1] <- ar.coeff 
        }
        if(order.t[1]==2)
        {
          ar.coeff2 <- runif(1, 0.5, 0.7)
          lower.ar.b <- ar.coeff2 - 0.9
          upper.ar.b <- 0.9 - ar.coeff2
          ar.coeff1 <- runif(1, lower.ar.b, upper.ar.b)
          ar.coeff <- c(ar.coeff1, ar.coeff2)
          ar.d[j, 1:2] <- ar.coeff
        }

        # define MA coefficients
        ma.coeff <- c()
        if(order.t[3]==0)
        {
          ma.d[j, 1] <- NA
        }
        ma.coeff1 <- 0
        ma.coeff2 <- 0
        if(order.t[3]==1)
        {
          ma.coeff1 <- runif(1, 0.5, 0.7)
          ma.coeff <- ma.coeff1
          ma.d[j, 1] <- ma.coeff
        }
        if(order.t[3]==2)
        {
          ma.coeff2 <- runif(1, 0.5, 0.7)
          lower.ma.b <- -1 * (0.9 + ma.coeff2) / ((0.9+0.7)/0.5)
          upper.ma.b <- -1 * lower.ma.b
          ma.coeff1 <- runif(1, lower.ma.b, upper.ma.b)
          ma.coeff <- c(ma.coeff1, ma.coeff2)
          ma.d[j, 1:2] <- ma.coeff
        }
      }
      return(list(ar.d, order.d, ma.d))
    }

    # Simulating data from an ARIMA process - For correlated serie
    data.genCor <- function(n, n.bot, var.mat)
    { 
      order.gen <- ordergenCor(n.bot)
      ar.d <- order.gen[[1]]
      order.d <- order.gen[[2]]
      ma.d <- order.gen[[3]]
      data.bot <- matrix(, n, n.bot)
      error.arima <- rmvnorm(n, mean=rep(0, n.bot), sigma=var.mat)

      for(j in 1:n.bot)
      {
        # generating data from a ARIMA model
        data.bot[, j] <- arima.sim(list(order=order.d[j, ], ar=na.omit(ar.d[j, ]), ma=na.omit(ma.d[j, ])), n, innov=error.arima[, j])[(order.d[j, 2] + 1):(n+order.d[j, 2])]
      }
      return(list(data.bot, order.gen))
    }

    # Generating AR, MA terms with the correspoding coefficients - For the common pattern
    ordergenCom <- function(n.bot)
    {

      #order.diff <- rep(1, n.bot) # ?????????? why 1 ???
      order.diff <- rep(0, n.bot)
      order.ar <- sample(0:2, n.bot, replace = TRUE)
      order.ma <- sample(0:2, n.bot, replace = TRUE)
      order.d <- cbind(order.ar, order.diff, order.ma)

      ar.d <- matrix(, n.bot, 2)
      ma.d <- matrix(, n.bot, 2)

      for(j in 1:n.bot)
      {
        order.t <- order.d[j, ]

        # define AR coefficients
        ar.coeff <- c()
        if(order.t[1]==0)
        {
          ar.d[j, 1] <- NA
        }

        ar.coeff1 <- 0
        ar.coeff2 <- 0
        if(order.t[1]==1)
        {
          ar.coeff1 <- runif(1, 0.5, 0.7)
          ar.coeff <- ar.coeff1
          ar.d[j, 1] <- ar.coeff 
        }
        if(order.t[1]==2)
        {
          ar.coeff2 <- runif(1, 0.5, 0.7)
          lower.ar.b <- ar.coeff2 - 0.9
          upper.ar.b <- 0.9 - ar.coeff2
          ar.coeff1 <- runif(1, lower.ar.b, upper.ar.b)
          ar.coeff <- c(ar.coeff1, ar.coeff2)
          ar.d[j, 1:2] <- ar.coeff
        }

        # define MA coefficients
        ma.coeff <- c()
        if(order.t[3]==0)
        {
          ma.d[j, 1] <- NA
        }
        ma.coeff1 <- 0
        ma.coeff2 <- 0
        if(order.t[3]==1)
        {
          ma.coeff1 <- runif(1, 0.5, 0.7)
          ma.coeff <- ma.coeff1
          ma.d[j, 1] <- ma.coeff
        }
        if(order.t[3]==2)
        {
          ma.coeff2 <- runif(1, 0.5, 0.7)
          lower.ma.b <- -1 * (0.9 + ma.coeff2) / ((0.9+0.7)/0.5)
          upper.ma.b <- -1 * lower.ma.b
          ma.coeff1 <- runif(1, lower.ma.b, upper.ma.b)
          ma.coeff <- c(ma.coeff1, ma.coeff2)
          ma.d[j, 1:2] <- ma.coeff
        }
      }
      return(list(ar.d, order.d, ma.d))
    }

    # Simulating data from an ARIMA process - For the common pattern
    data.genCom <- function(n, n.bot, var)
    { 
      order.gen <- ordergenCom(n.bot)
      ar.d <- order.gen[[1]]
      order.d <- order.gen[[2]]
      ma.d <- order.gen[[3]]
      data.bot <- matrix(, n, n.bot)
      error.arima <- rnorm(n, mean = 0, sd = sqrt(var))

      # generating data from a ARIMA model
      data.bot[, ] <- arima.sim(list(order=order.d[1, ], ar=na.omit(ar.d[1, ]), ma=na.omit(ma.d[1, ])), n, innov=error.arima)[(order.d[1, 2] + 1):(n+order.d[1, 2])]

      return(list(data.bot, order.gen))
    }

    # Generating AR, MA terms with the correspoding coefficients - For Noise at the bottom level
    ordergenNoise <- function(n.bot)
    {

      order.diff <- rep(0, n.bot) # ?????
      order.ar <- sample(0:2, n.bot, replace = TRUE)
      order.ma <- sample(0:2, n.bot, replace = TRUE)
      order.d <- cbind(order.ar, order.diff, order.ma)


      ar.d <- matrix(, n.bot, 2)
      ma.d <- matrix(, n.bot, 2)

      for(j in 1:n.bot)
      {
        order.t <- order.d[j, ]

        # define AR coefficients
        ar.coeff <- c()
        if(order.t[1]==0)
        {
          ar.d[j, 1] <- NA
        }

        ar.coeff1 <- 0
        ar.coeff2 <- 0
        if(order.t[1]==1)
        {
          ar.coeff1 <- runif(1, 0.5, 0.7)
          ar.coeff <- ar.coeff1
          ar.d[j, 1] <- ar.coeff 
        }
        if(order.t[1]==2)
        {
          ar.coeff2 <- runif(1, 0.5, 0.7)
          lower.ar.b <- ar.coeff2 - 0.9
          upper.ar.b <- 0.9 - ar.coeff2
          ar.coeff1 <- runif(1, lower.ar.b, upper.ar.b)
          ar.coeff <- c(ar.coeff1, ar.coeff2)
          ar.d[j, 1:2] <- ar.coeff
        }

        # define MA coefficients
        ma.coeff <- c()
        if(order.t[3]==0)
        {
          ma.d[j, 1] <- NA
        }
        ma.coeff1 <- 0
        ma.coeff2 <- 0
        if(order.t[3]==1)
        {
          ma.coeff1 <- runif(1, 0.5, 0.7)
          ma.coeff <- ma.coeff1
          ma.d[j, 1] <- ma.coeff
        }
        if(order.t[3]==2)
        {
          ma.coeff2 <- runif(1, 0.5, 0.7)
          lower.ma.b <- -1 * (0.9 + ma.coeff2) / ((0.9+0.7)/0.5)
          upper.ma.b <- -1 * lower.ma.b
          ma.coeff1 <- runif(1, lower.ma.b, upper.ma.b)
          ma.coeff <- c(ma.coeff1, ma.coeff2)
          ma.d[j, 1:2] <- ma.coeff
        }
      }
      return(list(ar.d, order.d, ma.d))
    }


    # Simulating data from an ARMA process 
    data.genNoise <- function(n, n.bot, var.mat)
    { 
      order.gen <- ordergenNoise(n.bot)
      ar.d <- order.gen[[1]]
      order.d <- order.gen[[2]]
      ma.d <- order.gen[[3]]
      data.bot <- matrix(, n, n.bot)
      error.arima <- rmvnorm(n, mean=rep(0, n.bot), sigma=var.mat)

      for(j in 1:n.bot)
      {
        # generating data from a ARIMA model
        data.bot[, j] <- arima.sim(list(order=order.d[j, ], ar=na.omit(ar.d[j, ]), ma=na.omit(ma.d[j, ])), n, innov=error.arima[, j])[(order.d[j, 2] + 1):(n+order.d[j, 2])]
      }
      return(list(data.bot, order.gen))
    }


    simulate_large_hts <- function(n){

      #nodes <- list(6, rep(4, 6), rep(4, 24), rep(4, 96), rep(4, 384))
      # nodes <- list(6, rep(4, 6), rep(4, 24), rep(4, 96))
      #nodes <- list(6, rep(4, 6), rep(4, 24))
      nodes <- list(10, rep(4, 10), rep(4, 40))

      gmat <- hts:::GmatrixH(nodes)
      gmat <- apply(gmat, 1, table)
      n.tot <- sum(unlist(nodes)) + 1
      n.bot <- sum(nodes[[length(nodes)]])

      # Generating data for the common pattern
      varCom <- 0.005
      dataCom <- matrix(data.genCom(n, 1, var = varCom)[[1]], ncol = 1)
      allCom <- dataCom[, rep(1, times = n.bot)] 

      # Only 2 series out of 4 at the bottom level contains the common pattern
      idxCom <- c(seq(1, n.bot, 4), seq(2, n.bot, 4))
      allCom[, idxCom] <- 0

      # Generating data for the correlated pattern
      bCor <- runif(n.bot/4, 0.3, 0.8) # block correlations
      bCorMat <- lapply(bCor, function(x){
        y <- matrix(x, 4, 4)
        diag(y) = 1
        return(y)
      })

      corMatB <- bdiag(bCorMat) # correlation matrix at the bottom level
      varVec <- runif(n.bot, 0.05, 0.1)
      varMat <- as.matrix(Diagonal(x = sqrt(varVec)) %*% corMatB %*% Diagonal(x = sqrt(varVec))) # cov matrix
      allCor <- data.genCor(n, n.bot, varMat)[[1]] # generates data with correlated errors

      # Adding noise to the common pattern
      varRange <- list(0.4, 0.4, 0.4, 0.4, 0.4) # variance of the noise for each level (level 1 to level 5)

      # generates white noise errors for level 1 to level 4 (level 5: bottom level ignored)
      noiseL <- list()
      for(i in 1:(length(nodes)-1))
      {
        nodesLv <- sum(nodes[[i]])
        var.mat <- diag(nodesLv / 2)
        diag(var.mat) <- rep(varRange[[i]], nodesLv / 2)
        datL <- rmvnorm(n, rep(0, nodesLv / 2), var.mat)
        datL <- datL[, rep(1:ncol(datL), each = 2)] # replicating to get the data for -ve part
        sign.vec <- rep(c(1, -1), nodesLv / 2) # adding +/- into the data
        datL <- t(t(datL) * sign.vec / as.numeric(gmat[[i+1]])) # contribution that passes to the bottom level
        datL <- datL[, rep(1:ncol(datL), times = gmat[[i+1]])] # all noise series at the bottom level
        noiseL[[i]] <- datL
      }

      # generate ARMA series for the noise at the bottom level
      var.mat <- diag(n.bot/2)
      diag(var.mat) <- rep(varRange[[length(nodes)]], sum(nodes[[length(nodes)]])/2)
      noiseB <- data.genNoise(n, n.bot/2, var.mat)[[1]]
      noiseB <- noiseB[, rep(1:ncol(noiseB), each = 2)]
      sign.vec <- rep(c(1, -1), n.bot/2)
      noiseB <- t(t(noiseB) * sign.vec) 
      noiseB[, idxCom] <- 0L # adding noise only to common component

      # common + correlated + noise
      allB <- allCom + allCor + Reduce("+", noiseL) + noiseB
      dat.new <- ts(allB, frequency = 1L)
      sim.hts <- hts(dat.new, nodes = nodes)
      ally = allts(sim.hts)

      S <- smatrix(sim.hts)
      A <- head(S, nrow(S) - ncol(S))
      list(A = A, bts = allB, sim.hts = sim.hts)
    }
    """

    return sim_large


def bight():
    bight = """
    bights <- function(bts, A) {
    nbts <- ncol(bts)
    naggts <- nrow(A)
    nts <- naggts + nbts
    Tobs <- nrow(bts)
      
    A <- methods::as(A, "sparseMatrix")
    S <- rbind(A, Diagonal(nbts))
    S <- methods::as(S, "sparseMatrix")
      
    yts <- matrix(NA, nrow = nrow(bts), ncol = nts)
      
    if (nbts <= 1L) {
    stop("Argument bts must be a multivariate time series.", call. = FALSE)
    }
      
    yts[, seq(naggts)] <-  as.matrix(t(A %*% t(bts)))
    yts[, seq(naggts + 1, nts)] <- as.matrix(bts)
      
    if(is.ts(bts)){
    yts <- ts(yts, start(bts), freq = tsp(bts)[3])
    }
      
    output <- structure(
    list(yts = yts, A = A, S = as.matrix(S), nbts = nbts, naggts = naggts, nts = nts, Tobs = Tobs),
    class = c("bights")
    )
    return(output)
    }
    """

    return bight


def mint_shr():
    mint = """
    mint_recon <- function(allf, nodes, res) {
    allf <- ts(allf)
    y.f_cg <- MinT(allf, nodes, residual = res, covariance = "shr", keep = "all", algorithms = "cg")
    return(y.f_cg)
    }
    """
    return mint


def mint_sam():
    mint = """
    mint_recon <- function(allf, nodes, res) {
    allf <- ts(allf)
    y.f_cg <- MinT(allf, nodes, residual = res, covariance = "sam", keep = "all", algorithms = "cg")
    return(y.f_cg)
    }
    """
    return mint


def mint_ols():
    mint = """
    mint_recon <- function(allf, nodes, res) {
    allf <- ts(allf)
    y.f_cg <- combinef(allf, nodes, weights = NULL, keep = "all", algorithms = "cg")
    return(y.f_cg)
    }
    """
    return mint

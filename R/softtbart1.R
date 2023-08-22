

#' @title Type I Tobit Soft Bayesian Additive Regression Trees with sparsity inducing hyperprior implemented using MCMC
#'
#' @description Type I Tobit Soft Bayesian Additive Regression Trees with sparsity inducing hyperprior implemented using MCMC
#' @import dbarts
#' @import truncnorm
#' @import mvtnorm
#' @import censReg
#' @import fastncdf
#' @param x.train The training covariate data for all training observations. Number of rows equal to the number of observations. Number of columns equal to the number of covariates.
#' @param x.test The test covariate data for all test observations. Number of rows equal to the number of observations. Number of columns equal to the number of covariates.
#' @param y The training data vector of outcomes. A continuous, censored outcome variable.
#' @param n.iter Number of iterations excluding burnin.
#' @param n.burnin Number of burnin iterations.
#' @param below_cens Number at or below which observations are censored.
#' @param above_cens Number at or above which observations are censored.
#' @param n.trees A positive integer giving the number of trees used in the sum-of-trees formulation.
#' @param print.opt Print every print.opt number of Gibbs samples.
#' @param fast If equal to TRUE, then implements faster truncated normal draws and approximates normal pdf.
#' @export
#' @return The following objects are returned:
#' \item{Z.matcens}{Matrix of draws of latent (censored) outcomes for censored observations. Number of rows equals number of censored training observations. Number of columns equals n.iter . Rows are ordered in order of censored observations in the training data.}
#' \item{Z.matcensbelow}{Matrix of draws of latent (censored) outcomes for observations censored from below. Number of rows equals number of training observations censored from below. Number of columns equals n.iter . Rows are ordered in order of censored observations in the training data. }
#' \item{Z.matcensabove}{Matrix of draws of latent (censored) outcomes for observations censored from above. Number of rows equals number of training observations censored from above. Number of columns equals n.iter . Rows are ordered in order of censored observations in the training data. }
#' \item{mu}{Matrix of draws of the sum of terminal nodes, i.e. f(x_i), for all training observations. Number of rows equals number of training observations. Number of columns equals n.iter .}
#' \item{mucens}{Matrix of draws of the sum of terminal nodes, i.e. f(x_i), for all censored training observations. Number of rows equals number of censored training observations. Number of columns equals n.iter .}
#' \item{muuncens}{Matrix of draws of the sum of terminal nodes, i.e. f(x_i), for all uncensored training observations. Number of rows equals number of uncensored training observations. Number of columns equals n.iter .}
#' \item{mucensbelow}{Matrix of draws of the sum of terminal nodes, i.e. f(x_i), for all training observations censored from below. Number of rows equals number of training observations censored from below. Number of columns equals n.iter .}
#' \item{mucensabove}{Matrix of draws of the sum of terminal nodes, i.e. f(x_i), for all training observations censored from above Number of rows equals number of training observations censored from above Number of columns equals n.iter .}
#' \item{ystar}{Matrix of training sample draws of the outcome assuming uncensored (can take values below below_cens and above above_cens. Number of rows equals number of training observations. Number of columns equals n.iter .}
#' \item{ystarcens}{Matrix of censored training sample draws of the outcome assuming uncensored (can take values below below_cens and above above_cens. Number of rows equals number of censored training observations. Number of columns equals n.iter .}
#' \item{ystaruncens}{Matrix of uncensored training sample draws of the outcome assuming uncensored (can take values below below_cens and above above_cens. Number of rows equals number of uncensored training observations. Number of columns equals n.iter .}
#' \item{ystarcensbelow}{Matrix of censored from below training sample draws of the outcome assuming uncensored (can take values below below_cens and above above_cens. Number of rows equals number of training observations censored from below. Number of columns equals n.iter .}
#' \item{ystarcensabove}{Matrix of censored from above training sample draws of the outcome assuming uncensored (can take values below below_cens and above above_cens. Number of rows equals number of training observations censored from above. Number of columns equals n.iter .}
#' \item{test.mu}{Matrix of draws of the sum of terminal nodes, i.e. f(x_i), for all test observations. Number of rows equals number of test observations. Number of columns equals n.iter .}
#' \item{test.y_nocensoring}{Matrix of test sample draws of the outcome assuming uncensored. Can take values below below_cens and above above_cens. Number of rows equals number of test observations. Number of columns equals n.iter .}
#' \item{test.y_withcensoring}{Matrix of test sample draws of the outcome assuming censored. Cannot take values below below_cens and above above_cens. Number of rows equals number of test observations. Number of columns equals n.iter .}
#' \item{test.probcensbelow}{Matrix of draws of probabilities of test sample observations being censored from below. Number of rows equals number of test observations. Number of columns equals n.iter .}
#' \item{test.probcensabove}{Matrix of draws of probabilities of test sample observations being censored from above. Number of rows equals number of test observations. Number of columns equals n.iter .}
#' \item{sigma}{Vector of draws of the standard deviation of the error term. Number of elements equals n.iter .}
#' @examples
#'
#'#example taken from https://stats.idre.ucla.edu/r/dae/tobit-models/
#'
#'dat <- read.csv("https://stats.idre.ucla.edu/stat/data/tobit.csv")
#'
#'train_inds <- sample(1:200,190)
#'test_inds <- (1:200)[-train_inds]
#'
#'ytrain <- dat$apt[train_inds]
#'ytest <- dat$apt[test_inds]
#'
#'xtrain <- cbind(dat$read, dat$math)[train_inds,]
#'xtest <- cbind(dat$read, dat$math)[test_inds,]
#'
#'tobart_res <- tbart1(xtrain,xtest,ytrain,
#'                     below_cens = -Inf,
#'                     above_cens = 800,
#'                     n.iter = 400,
#'                     n.burnin = 100)
#'
#' @export

softtbart1 <- function(x.train,
                   x.test,
                   y,
                   n.iter=1000,
                   n.burnin=100,
                   below_cens = 0,
                   above_cens = Inf,
                   n.trees = 50L,
                   SB_group = NULL,
                   SB_alpha = 1,
                   SB_beta = 2,
                   SB_gamma = 0.95,
                   SB_k = 2,
                   SB_sigma_hat = NULL,
                   SB_shape = 1,
                   SB_width = 0.1,
                   # SB_num_tree = 20,
                   SB_alpha_scale = NULL,
                   SB_alpha_shape_1 = 0.5,
                   SB_alpha_shape_2 = 1,
                   SB_tau_rate = 10,
                   SB_num_tree_prob = NULL,
                   SB_temperature = 1,
                   SB_weights = NULL,
                   SB_normalize_Y = TRUE,
                   print.opt = 100,
                   fast=TRUE){




  # if(is.vector(x.train) | is.factor(x.train)| is.data.frame(x.train)) x.train = as.matrix(x.train)
  # if(is.vector(x.test) | is.factor(x.test)| is.data.frame(x.test)) x.test = as.matrix(x.test)

  # if((!is.matrix(x.train))) stop("argument x.train must be a double matrix")
  # if((!is.matrix(x.test)) ) stop("argument x.test must be a double matrix")

  if(nrow(x.train) != length(y)) stop("number of rows in x.train must equal length of y.train")
  if((ncol(x.test)!=ncol(x.train))) stop("input x.test must have the same number of columns as x.train")


  # mu_Y <- mean(y)
  # sd_Y <- sd(y)
  # y <- (y - mu_Y) / sd_Y
  # below_cens <- (below_cens - mu_Y) / sd_Y
  # above_cens <- (above_cens - mu_Y) / sd_Y

  make_01_norm <- function(x) {
    a <- min(x)
    b <- max(x)
    return(function(y0) (y0 - a) / (b - a))
  }

  ecdfs   <- list()
  for(i in 1:ncol(x.train)) {
    ecdfs[[i]] <- ecdf(x.train[,i])
    if(length(unique(x.train[,i])) == 1) ecdfs[[i]] <- identity
    if(length(unique(x.train[,i])) == 2) ecdfs[[i]] <- make_01_norm(x.train[,i])
  }
  for(i in 1:ncol(x.train)) {
    x.train[,i] <- ecdfs[[i]](x.train[,i])
    x.test[,i] <- ecdfs[[i]](x.test[,i])
  }

  rm(ecdfs)


  #indexes of censored observations
  cens_inds <- which(y <= below_cens | y >= above_cens)
  if(length(cens_inds)==0) stop("No censored observations")


  uncens_inds <- which(y > below_cens & y < above_cens)
  censbelow_inds <- which(y <= below_cens )
  censabove_inds <- which(y >= above_cens )

  #create z vector
  z <- rep(NA, length(y))
  z[uncens_inds] <- y[uncens_inds]

  #this line is perhaps unnecessary
  z[which(y <= below_cens )] <- below_cens
  z[which(y >= above_cens )] <- above_cens


  n <- length(y)
  n0 <- length(cens_inds)
  n1 <- length(uncens_inds)
  n_censbelow <- length(which(y <= below_cens))
  n_censabove <- length(which(y >= above_cens))

  ntest = nrow(x.test)


  draw = list(
    Z.mat = array(NA, dim = c(n, n.iter)),
    Z.matcens = array(NA, dim = c(n0, n.iter)),
    #Z.matuncens = array(NA, dim = c(n1, n.iter)),
    Z.matcensbelow = array(NA, dim = c(n_censbelow, n.iter)),
    Z.matcensabove = array(NA, dim = c(n_censabove, n.iter)),
    mu = array(NA, dim = c(n, n.iter)),#,
    mucens = array(NA, dim = c(n0, n.iter)),#,
    muuncens = array(NA, dim = c(n1, n.iter)),#,
    mucensbelow = array(NA, dim = c(n_censbelow, n.iter)),#,
    mucensabove = array(NA, dim = c(n_censabove, n.iter)),#,
    ystar = array(NA, dim = c(n, n.iter)),#,
    ystarcens = array(NA, dim = c(n0, n.iter)),#,
    ystaruncens = array(NA, dim = c(n1, n.iter)),#,
    ystarcensbelow = array(NA, dim = c(n_censbelow, n.iter)),#,
    ystarcensabove = array(NA, dim = c(n_censabove, n.iter)),#,
    test.mu =  array(NA, dim = c(ntest, n.iter)),#,
    test.y_nocensoring =  array(NA, dim = c(ntest, n.iter)),#,
    test.y_withcensoring =  array(NA, dim = c(ntest, n.iter)),#,
    test.probcensbelow =  array(NA, dim = c(ntest, n.iter)),#,
    test.probcensabove =  array(NA, dim = c(ntest, n.iter)),
    sigma = rep(NA, n.iter),
    cond_exp_train =  array(NA, dim = c(n, n.iter)),
    cond_exp_test =  array(NA, dim = c(ntest, n.iter))
  )




  hypers <- Hypers(x.train, y,
                   num_tree = n.trees, #sigma_hat = 1,
                   group = SB_group,
                   alpha = SB_alpha,
                   beta = SB_beta,
                   gamma = SB_gamma,
                   k = SB_k,
                   # sigma_hat = NULL,
                   shape = SB_shape,
                   width = SB_width,
                   # num_tree = 20,
                   alpha_scale = SB_alpha_scale,
                   alpha_shape_1 = SB_alpha_shape_1,
                   alpha_shape_2 = SB_alpha_shape_2,
                   tau_rate = SB_tau_rate,
                   num_tree_prob = SB_num_tree_prob,
                   temperature = SB_temperature,
                   weights = SB_weights,
                   normalize_Y = SB_normalize_Y
                   )


  opts <- Opts(update_sigma = TRUE, num_print = print.opt)


  sampler_forest <- MakeForest(hypers, opts)





  # control <- dbartsControl(updateState = updateState, verbose = FALSE,  keepTrainingFits = TRUE,
  #                          keepTrees = TRUE,
  #                          n.trees = n.trees,
  #                          n.burn = n.burn,
  #                          n.samples = n.samples,
  #                          n.thin = n.thin,
  #                          n.chains = n.chains,
  #                          n.threads = n.threads,
  #                          printEvery = printEvery,
  #                          printCutoffs = printCutoffs,
  #                          rngKind = rngKind,
  #                          rngNormalKind = rngNormalKind,
  #                          rngSeed = rngSeed)


  # print(colnames(Xmat.train))

  # print("begin dbarts")


  # if(nrow(x.test )==0){
  #   sampler <- dbarts(y ~ .,
  #                     data = as.data.frame(x.train),
  #                     #test = x.test,
  #                     control = control,
  #                     tree.prior = tree.prior,
  #                     node.prior = node.prior,
  #                     resid.prior = resid.prior,
  #                     proposal.probs = proposal.probs,
  #                     sigma = sigmadbarts
  #   )
  #
  # }else{
  #   sampler <- dbarts(y ~ .,
  #                     data = as.data.frame(x.train),
  #                     test = as.data.frame(x.test),
  #                     control = control,
  #                     tree.prior = tree.prior,
  #                     node.prior = node.prior,
  #                     resid.prior = resid.prior,
  #                     proposal.probs = proposal.probs,
  #                     sigma = sigmadbarts
  #   )
  #
  # }




  # sampler$setResponse(y = z)
  # sampler$setSigma(sigma = 1)

  #sampler$setPredictor(x= Xmat.train$x, column = 1, forceUpdate = TRUE)

  #mu = as.vector( alpha + X.mat %*% beta )
  # sampler$sampleTreesFromPrior()
  # samplestemp <- sampler$run()

  #mutemp <- samplestemp$train[,1]
  #suppose there are a number of samples


  mu <- sampler_forest$do_gibbs(x.train, z, x.train, 1)
  mutest <- sampler_forest$do_predict(x.test)


  # print("sigma = ")
  sigma <- sampler_forest$get_sigma()

  # mu <- samplestemp$train[,1]
  # mutest <- samplestemp$test[,1]

  ystar <- rnorm(n,mean = mu, sd = sigma)
  ystartest <- rnorm(ntest,mean = mutest, sd = sigma)


  # if(fast){
  #   ystartestcens <-tnorm(ntest, a = below_cens, b = above_cens, mean = mutest, sd = sigma)
  #
  #
  # }else{
  ystartestcens <-rtruncnorm(ntest, a = below_cens, b = above_cens, mean = mutest, sd = sigma)

  # }




  if(fast){
    probcensbelow_train <- fastpnorm((below_cens - mu)/sigma)
    probcensabove_train <- 1 - fastpnorm((above_cens- mu)/sigma)
  }else{
    probcensbelow_train <- pnorm(below_cens, mean = mu, sd = sigma)
    probcensabove_train <- 1 - pnorm(above_cens, mean = mu, sd = sigma)
  }


  if(fast){
    probcensbelow <- fastpnorm((below_cens - mutest)/sigma)
    probcensabove <- 1 - fastpnorm((above_cens - mutest)/sigma)

  }else{
    probcensbelow <- pnorm(below_cens, mean = mutest, sd = sigma)
    probcensabove <- 1 - pnorm(above_cens, mean = mutest, sd = sigma)
  }



  # condexptrain <- below_cens*probcensbelow_train +
  #   (mu)*(1- probcensabove_train - probcensbelow_train) +
  #   sigma*( fastnormdens(below_cens, mean = mu, sd = sigma) -
  #             fastnormdens(above_cens, mean = mu, sd = sigma) ) +
  #   above_cens*probcensabove_train
  #
  # condexptest <- below_cens*probcensbelow +
  #   (mutest)*(1- probcensabove - probcensbelow) +
  #   sigma*( fastnormdens(below_cens, mean = mutest, sd = sigma) -
  #             fastnormdens(above_cens, mean = mutest, sd = sigma) ) +
  #   above_cens*probcensabove




  if(below_cens == - Inf){
    if(above_cens == Inf){
      condexptrain <- (mu )

      condexptest <- (mutest )
    }else{ # above_cens !=Inf
      condexptrain <-
        (mu )*(1- probcensabove_train ) +
        sigma*(  - fastnormdens(above_cens, mean = mu , sd = sigma) ) +
        above_cens*probcensabove_train

      condexptest <-
        (mutest )*(1- probcensabove ) +
        sigma*(  -fastnormdens(above_cens, mean = mutest, sd = sigma) ) +
        above_cens*probcensabove
    }
  }else{ # below_cens != - Inf
    if(above_cens == Inf){
      condexptrain <- below_cens*probcensbelow_train +
        (mu )*(1 - probcensbelow_train) +
        sigma*( fastnormdens(below_cens, mean = mu , sd = sigma)  )

      condexptest <- below_cens*probcensbelow +
        (mutest )*(1 - probcensbelow) +
        sigma*( fastnormdens(below_cens, mean = mutest, sd = sigma)  )


    }else{ # above_cens !=Inf
      condexptrain <- below_cens*probcensbelow_train +
        (mu )*(1- probcensabove_train - probcensbelow_train) +
        sigma*( fastnormdens(below_cens, mean = mu , sd = sigma) -
                  fastnormdens(above_cens, mean = mu , sd = sigma) ) +
        above_cens*probcensabove_train

      condexptest <- below_cens*probcensbelow +
        (mutest )*(1- probcensabove - probcensbelow) +
        sigma*( fastnormdens(below_cens, mean = mutest, sd = sigma) -
                  fastnormdens(above_cens, mean = mutest, sd = sigma) ) +
        above_cens*probcensabove
    }
  }




  #save the first round of values
  if(n.burnin == 0){
    draw$Z.mat[,1] = z # * sd_Y + mu_Y
    draw$Z.matcens[,1] = z[cens_inds] # * sd_Y + mu_Y
    # draw$Z.matuncens[,1] = z[uncens_inds]
    draw$Z.matcensbelow[,1] = z[censbelow_inds] # * sd_Y + mu_Y
    draw$Z.matcensabove[,1] = z[censabove_inds] # * sd_Y + mu_Y
    draw$mu[,1] = mu # * sd_Y + mu_Y
    draw$mucens[,1] = mu[cens_inds] # * sd_Y + mu_Y
    draw$muuncens[,1] = mu[uncens_inds] # * sd_Y + mu_Y
    draw$mucensbelow[,1] = mu[censbelow_inds] # * sd_Y + mu_Y
    draw$mucensabove[,1] = mu[censabove_inds] # * sd_Y + mu_Y
    draw$ystar[,1] = ystar # * sd_Y + mu_Y
    draw$ystarcens[,1] = ystar[cens_inds] # * sd_Y + mu_Y
    draw$ystaruncens[,1] = ystar[uncens_inds] # * sd_Y + mu_Y
    draw$ystarcensbelow[,1] = ystar[censbelow_inds] # * sd_Y + mu_Y
    draw$ystarcensabove[,1] = ystar[censabove_inds] # * sd_Y + mu_Y
    draw$test.mu[,1] = mutest # * sd_Y + mu_Y
    draw$test.y_nocensoring[,1] = ystartest # * sd_Y + mu_Y
    draw$test.y_withcensoring[,1] = ystartestcens # * sd_Y + mu_Y
    draw$test.probcensbelow[,1] = probcensbelow
    draw$test.probcensabove[,1] = probcensabove
    draw$sigma[1] <- sigma # * sd_Y

    draw$cond_exp_train[, 1] <- condexptrain # * sd_Y + mu_Y
    draw$cond_exp_test[, 1] <- condexptest # * sd_Y + mu_Y
  }

  ######## loop through the Gibbs sampler iterations ###############
  for(iter in 2:(n.iter+n.burnin)){

    #draw the latent outcome
    # z[cens_inds] <- rtruncnorm(n0, a= below_cens, b = above_cens, mean = mu[cens_inds], sd = sigma)
    if(length(censbelow_inds)>0){
      z[censbelow_inds] <- rtruncnorm(n_censbelow, a= -Inf, b = below_cens, mean = mu[censbelow_inds], sd = sigma)
    }
    if(length(censabove_inds)>0){
      z[censabove_inds] <- rtruncnorm(n_censabove, a= above_cens, b = Inf, mean = mu[censabove_inds], sd = sigma)
    }


    #set the response.
    #Check that 0 is a reasonable initial value
    #perhaps makes more sense to use initial values of Z
    # sampler$setResponse(y = z)
    # sampler$setSigma(sigma = 1)
    #sampler$setPredictor(x= Xmat.train$x, column = 1, forceUpdate = TRUE)

    #mu = as.vector( alpha + X.mat %*% beta )
    # samplestemp <- sampler$run()
    #
    # sigma <- samplestemp$sigma
    #
    # mu <- samplestemp$train[,1]
    # mutest <- samplestemp$test[,1]


    mu <- sampler_forest$do_gibbs(x.train, z, x.train, 1)
    mutest <- sampler_forest$do_predict(x.test)


    # print("sigma = ")
    sigma <- sampler_forest$get_sigma()



    #draw uncensored predictions of y
    ystar <- rnorm(n,mean = mu, sd = sigma)
    ystartest <- rnorm(ntest,mean = mutest, sd = sigma)

    # ystartestcens <- rtruncnorm(ntest, a= below_cens, b= above_cens, mean = mutest, sd = sigma)
    ystartestcens <- ystartest
    ystartestcens[ystartest < below_cens] <- below_cens
    ystartestcens[ystartest > above_cens] <- above_cens


    if(fast){
      probcensbelow_train <- fastpnorm((below_cens - mu)/sigma)
      probcensabove_train <- 1 - fastpnorm((above_cens - mu)/sigma)
    }else{
      probcensbelow_train <- pnorm(below_cens, mean = mu, sd = sigma)
      probcensabove_train <- 1 - pnorm(above_cens, mean = mu, sd = sigma)
    }

    if(fast){
      probcensbelow <- fastpnorm((below_cens - mutest)/sigma)
      probcensabove <- 1 - fastpnorm((above_cens - mutest)/sigma)
    }else{
      probcensbelow <- pnorm(below_cens, mean = mutest, sd = sigma)
      probcensabove <- 1 - pnorm(above_cens, mean = mutest, sd = sigma)
    }



    # condexptrain <- below_cens*probcensbelow_train +
    #   (mu)*(1- probcensabove_train - probcensbelow_train) +
    #   sigma*( fastnormdens(below_cens, mean = mu, sd = sigma) -
    #             fastnormdens(above_cens, mean = mu, sd = sigma) ) +
    #   above_cens*probcensabove_train
    #
    #
    # condexptest <- below_cens*probcensbelow +
    #   (mutest)*(1- probcensabove - probcensbelow) +
    #   sigma*( fastnormdens(below_cens, mean = mutest, sd = sigma) -
    #                       fastnormdens(above_cens, mean = mutest, sd = sigma) ) +
    #   above_cens*probcensabove
    #
    #



    if(below_cens == - Inf){
      if(above_cens == Inf){
        condexptrain <- (mu )

        condexptest <- (mutest )
      }else{ # above_cens !=Inf
        condexptrain <-
          (mu )*(1- probcensabove_train ) +
          sigma*(  - fastnormdens(above_cens, mean = mu , sd = sigma) ) +
          above_cens*probcensabove_train

        condexptest <-
          (mutest )*(1- probcensabove ) +
          sigma*(  -fastnormdens(above_cens, mean = mutest, sd = sigma) ) +
          above_cens*probcensabove
      }
    }else{ # below_cens != - Inf
      if(above_cens == Inf){
        condexptrain <- below_cens*probcensbelow_train +
          (mu )*(1 - probcensbelow_train) +
          sigma*( fastnormdens(below_cens, mean = mu , sd = sigma)  )

        condexptest <- below_cens*probcensbelow +
          (mutest )*(1 - probcensbelow) +
          sigma*( fastnormdens(below_cens, mean = mutest, sd = sigma)  )


      }else{ # above_cens !=Inf
        condexptrain <- below_cens*probcensbelow_train +
          (mu )*(1- probcensabove_train - probcensbelow_train) +
          sigma*( fastnormdens(below_cens, mean = mu , sd = sigma) -
                    fastnormdens(above_cens, mean = mu , sd = sigma) ) +
          above_cens*probcensabove_train

        condexptest <- below_cens*probcensbelow +
          (mutest )*(1- probcensabove - probcensbelow) +
          sigma*( fastnormdens(below_cens, mean = mutest, sd = sigma) -
                    fastnormdens(above_cens, mean = mutest, sd = sigma) ) +
          above_cens*probcensabove
      }
    }




  #### save iteration output ##############

    if(iter>n.burnin){
      iter_min_burnin <- iter-n.burnin
      draw$Z.mat[,iter_min_burnin] = z # * sd_Y + mu_Y
      draw$Z.matcens[,iter_min_burnin] = z[cens_inds] # * sd_Y + mu_Y
      # draw$Z.matuncens[,iter_min_burnin] = z[uncens_inds] # * sd_Y + mu_Y
      draw$Z.matcensbelow[,iter_min_burnin] = z[censbelow_inds] # * sd_Y + mu_Y
      draw$Z.matcensabove[,iter_min_burnin] = z[censabove_inds] # * sd_Y + mu_Y
      draw$mu[,iter_min_burnin] = mu # * sd_Y + mu_Y
      draw$mucens[,iter_min_burnin] = mu[cens_inds] # * sd_Y + mu_Y
      draw$muuncens[,iter_min_burnin] = mu[uncens_inds] # * sd_Y + mu_Y
      draw$mucensbelow[,iter_min_burnin] = mu[censbelow_inds] # * sd_Y + mu_Y
      draw$mucensabove[,iter_min_burnin] = mu[censabove_inds] # * sd_Y + mu_Y
      draw$ystar[,iter_min_burnin] = ystar # * sd_Y + mu_Y
      draw$ystarcens[,iter_min_burnin] = ystar[cens_inds] # * sd_Y + mu_Y
      draw$ystaruncens[,iter_min_burnin] = ystar[uncens_inds] # * sd_Y + mu_Y
      draw$ystarcensbelow[,iter_min_burnin] = ystar[censbelow_inds] # * sd_Y + mu_Y
      draw$ystarcensabove[,iter_min_burnin] = ystar[censabove_inds] # * sd_Y + mu_Y
      draw$test.mu[,iter_min_burnin] = mutest # * sd_Y + mu_Y
      draw$test.y_nocensoring[,iter_min_burnin] = ystartest # * sd_Y + mu_Y
      draw$test.y_withcensoring[,iter_min_burnin] = ystartestcens # * sd_Y + mu_Y
      draw$test.probcensbelow[,iter_min_burnin] = probcensbelow
      draw$test.probcensabove[,iter_min_burnin] = probcensabove
      draw$sigma[iter_min_burnin] <- sigma # * sd_Y

      draw$cond_exp_train[, iter_min_burnin] <- condexptrain # * sd_Y + mu_Y
      draw$cond_exp_test[, iter_min_burnin] <- condexptest # * sd_Y + mu_Y

    }

    if(iter %% print.opt == 0){
      print(paste("Gibbs Iteration", iter))
      # print(c(sigma2.alpha, sigma2.beta))
    }


  }#end iterations of Giibs sampler



  return(draw)



}

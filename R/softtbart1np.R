
#' @export
fastcountuniq <- function(x)
{
  data.table(x)[, .N, keyby = x]$N
}

#' @export
fastnormdens <- function(x, mean = 0, sd = 0){
  (1/(sd*sqrt(2*pi)))*exp(-0.5*((x-mean)/sd)^2)
}


# speed up this function if necessary by replacing dt() with a faster function
dt_ls <- function(x, df=1, mu=0, sigma=1) (1/sigma) * dt((x - mu)/sigma, df)


#' @title Nonparametric Type I Tobit Soft Bayesian Additive Regression Trees with sparsity-inducing splitting hyperprior and a Dirichlet Process Mixture of normal distributions for the error term
#'
#' @description Type I Tobit Bayesian Additive Regression Trees implemented using MCMC with sparsity-inducing splitting hyperprior and a Dirichlet Process Mixture of normal distributions for the error term
#' @import dbarts
#' @import truncnorm
#' @import mvtnorm
#' @import censReg
#' @import dqrng
#' @import data.table
#' @import accelerometry
#' @import wrswoR
#' @param x.train The training covariate data for all training observations. Number of rows equal to the number of observations. Number of columns equal to the number of covariates.
#' @param x.test The test covariate data for all test observations. Number of rows equal to the number of observations. Number of columns equal to the number of covariates.
#' @param y The training data vector of outcomes. A continuous, censored outcome variable.
#' @param n.iter Number of iterations excluding burnin.
#' @param n.burnin Number of burnin iterations.
#' @param below_cens Number at or below which observations are censored.
#' @param above_cens Number at or above which observations are censored.
#' @param n.trees (dbarts control option) A positive integer giving the number of trees used in the sum-of-trees formulation.
#' @param n.chains (dbarts control option) A positive integer detailing the number of independent chains for the dbarts sampler to use (more than one chain is unlikely to improve speed because only one sample for each call to dbarts).
#' @param n.threads  (dbarts control option) A positive integer controlling how many threads will be used for various internal calculations, as well as the number of chains. Internal calculations are highly optimized so that single-threaded performance tends to be superior unless the number of observations is very large (>10k), so that it is often not necessary to have the number of threads exceed the number of chains.
#' @param printEvery (dbarts control option)If verbose is TRUE, every printEvery potential samples (after thinning) will issue a verbal statement. Must be a positive integer.
#' @param printCutoffs (dbarts control option) A non-negative integer specifying how many of the decision rules for a variable are printed in verbose mode
#' @param rngKind (dbarts control option) Random number generator kind, as used in set.seed. For type "default", the built-in generator will be used if possible. Otherwise, will attempt to match the built-in generator’s type. Success depends on the number of threads.
#' @param rngNormalKind (dbarts control option) Random number generator normal kind, as used in set.seed. For type "default", the built-in generator will be used if possible. Otherwise, will attempt to match the built-in generator’s type. Success depends on the number of threads and the rngKind
#' @param rngSeed (dbarts control option) Random number generator seed, as used in set.seed. If the sampler is running single-threaded or has one chain, the behavior will be as any other sequential algorithm. If the sampler is multithreaded, the seed will be used to create an additional pRNG object, which in turn will be used sequentially seed the threadspecific pRNGs. If equal to NA, the clock will be used to seed pRNGs when applicable.
#' @param updateState (dbarts control option) Logical setting the default behavior for many sampler methods with regards to the immediate updating of the cached state of the object. A current, cached state is only useful when saving/loading the sampler.
#' @param tree.prior (dbarts option) An expression of the form dbarts:::cgm or dbarts:::cgm(power,base) setting the tree prior used in fitting.
#' @param node.prior (dbarts option) An expression of the form dbarts:::normal or dbarts:::normal(k) that sets the prior used on the averages within nodes.
#' @param resid.prior (dbarts option) An expression of the form dbarts:::chisq or dbarts:::chisq(df,quant) that sets the prior used on the residual/error variance
#' @param proposal.probs (dbarts option) Named numeric vector or NULL, optionally specifying the proposal rules and their probabilities. Elements should be "birth_death", "change", and "swap" to control tree change proposals, and "birth" to give the relative frequency of birth/death in the "birth_death" step.
#' @param sigmadbarts (dbarts option) A positive numeric estimate of the residual standard deviation. If NA, a linear model is used with all of the predictors to obtain one.
#' @param print.opt Print every print.opt number of Gibbs samples.
#' @param lambda0 Lambda parameter for the base distribution for the error term.
#' @param nu0 nu parameter for sigma prior in base distribution G_0
#' @param sigest Estiamted standard deviation of outcome or error (used for setting base distirbution parameters).
#' @param sigquant Parameter for setting lambda0 (if NA). lambda0 set such that the sigquant quantile of the base distribution of sigma is the standard deviation of the outcome (as estimated by Maximum Likelihood assuming censored normal outcome).
#' @param alpha_prior The prior for the alpha parameter of the Dirichlet Process mixture of normals. If "vh" then apply the Gamma(c1,c2) prior of van Hasselt (2011) and Escobar (1994). If "george", then apply the prior of George (2019), McCulloch (2021), Conley (2008), and Antoniak (1974).
#' @param c1 If alpha_prior == "vh", then c1 is the shape parameter of the Gamma distribution.
#' @param c2 If alpha_prior == "vh", then c2 is the rate parameter of the Gamma distribution.
#' @param alpha_gridsize If alpha_prior = "george", this is the size of the grid to use for the discretized samples of alpha
#' @param mixstep If TRUE, includes a mixing step to speed up convergence of the Dirichlet Process Mixture draws. Default is TRUE.
#' @param init.many.clust If TRUE, initialize the Dirichlet Process Mixture with many clusters instead of 1 cluster. Default is FALSE.
#' @param k0_resids If FALSE (default) the maximum absolute value of the outcome determines k0 (with lambda0). If TRUE, the maximum residual from a linear regression determines k0.
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

softtbart1np <- function(x.train,
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
                     lambda0 = NA,
                     sigest = NA,
                     nu0=10,
                     sigquant = 0.95,
                     alpha_prior = "vh",
                     c1 = 2,
                     c2 = 2,
                     alpha_gridsize = 100L,
                     mixstep=TRUE,
                     init.many.clust = FALSE,
                     k0_resids = TRUE
){




  # if(is.vector(x.train) | is.factor(x.train)| is.data.frame(x.train)) x.train = as.matrix(x.train)
  # if(is.vector(x.test) | is.factor(x.test)| is.data.frame(x.test)) x.test = as.matrix(x.test)

  # if((!is.matrix(x.train))) stop("argument x.train must be a double matrix")
  # if((!is.matrix(x.test)) ) stop("argument x.test must be a double matrix")

  if(nrow(x.train) != length(y)) stop("number of rows in x.train must equal length of y.train")
  if((ncol(x.test)!= ncol(x.train))) stop("input x.test must have the same number of columns as x.train")


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


  #indexes of censored observations
  cens_inds <- which( (y <= below_cens) | (y >= above_cens))
  if(length(cens_inds)==0) stop("No censored observations")


  uncens_inds <- which( (y > below_cens) & (y < above_cens))
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
    sigmavecs_train =  array(NA, dim = c(n, n.iter)),
    sigmavecs_test =  array(NA, dim = c(ntest, n.iter)),
    error_mu_train =  array(NA, dim = c(n, n.iter)),
    error_mu_test = array(NA, dim = c(ntest, n.iter)),
    cond_exp_train =  array(NA, dim = c(n, n.iter)),
    cond_exp_test =  array(NA, dim = c(ntest, n.iter)),
    alpha = rep(NA,n.iter)
  )









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

  # alpha <- rgamma(n = 1, shape = c1, rate = c2) #draw from prior

  alpha <- c1/c2 # initialize at prior mean

  df0 <- data.frame(y)

  estResult <- censReg(y ~ 1,left = below_cens, right = above_cens, data = df0)
  sum_est <- summary( estResult )
  # mu0 <-  sum_est$estimate["(Intercept)", "Estimate"]
  mu0 <-  0#sum_est$estimate["(Intercept)", "Estimate"]

  # print("Line 222")
  # intialize parameters

  # nu  parameter for sigma prior in base distribution G_0
  # nu0 <- 10 # set to 10 as in full nonparametric BART paper
  # nu0 <- 3 # standard ABRT default would be to set it to 3

  # lambda set as in BART but with a 0.9 quantile instead of 0.95


  # code below for setting lambda taken from BART package

  if(is.na(lambda0)) {
    if(is.na(sigest)) {
      if(ncol(x.train) < n) {
        # df = data.frame(t(x.train),y.train)
        # lmf = lm(y.train~.,df)
        # sigest = summary(lmf)$sigma

        df0 <- data.frame(x.train,y)

        # print("df0 = ")
        # print(df0)

        estResult <- censReg(y ~ .,left = below_cens, right = above_cens, data = df0)
        sum_est <- summary( estResult )

        # print("sum_est = ")
        # print(sum_est)

        if(is.null(coef(estResult))){
          # estResult <- censReg(y ~ 1,left = below_cens, right = above_cens, data = df0)
          # sum_est <- summary( estResult )

          templm <- lm(y ~. , data = df0)
          df0 <- data.frame(y = y,
                            df0[,names(which(!is.na(templm$coefficients[2:length(templm$coefficients)])))])

          estResult <- censReg(y ~ .,left = below_cens, right = above_cens, data = df0)
          sum_est <- summary( estResult )

        }
        sigest <- exp(sum_est$estimate["logSigma", "Estimate"])



      } else {
        df0 <- data.frame(y)


        estResult <- censReg(y ~ 1,left = below_cens, right = above_cens, data = df0)


        sum_est <- summary( estResult )

        # print("sum_est = ")
        # print(sum_est)
        sigest <- exp(sum_est$estimate["logSigma", "Estimate"])


        # sigest = sd(y.train)
      }
    }
    qchi = qchisq(1.0-sigquant,nu0)
    lambda0 = (sigest*sigest*qchi)/nu0 #lambda parameter for sigma prior
  } else {
    sigest=sqrt(lambda0)
  }





  k_s <- 10 # using default from fully nonparametric BART paper

  #not obvious how to set this
  #could set to maximum of linear model residuals
  # emax <- max(abs(y -mu0   ))
  emax <- max(abs(y - mean(y)   ))

  if(k0_resids == TRUE){
    df0 <- data.frame(x.train,y)
    lmf <- lm(y~.,df0)
    emax <- max(abs(lmf$residuals))
  }


  k0 <- ( ( k_s*sqrt(lambda0))/emax )^2



  # print("nu0/2 = ")
  # print(nu0/2)
  #
  # print("nu0*lambda0/2 = ")
  # print(nu0*lambda0/2)


  sigma_init <- sigest#sqrt(1/rgamma(n = 1, shape =  nu0/2, rate = nu0*lambda0/2))

  mu_init <- 0#rnorm(1, mean = mu0, sd = sigma_init/sqrt(k0))


  sigma1_vec_train <- rep(sigma_init, n)
  sigma1_vec_test <- rep(sigma_init, ntest)


  mu1_vec_train <- rep(mu_init,n)
  mu1_vec_test <- rep(mu_init,ntest)


  if(init.many.clust==TRUE){
    # testinitinds <- sample(1:n, ntest, replace = FALSE)
    # testinitinds <- sample.int(n, ntest, replace = FALSE)
    testinitinds <- dqsample.int(n, ntest, replace = FALSE)

    sigma1_vec_train <- sqrt(1/rgamma(n = n, shape =  nu0/2, rate = nu0*lambda0/2))
    sigma1_vec_test <- sigma1_vec_train[testinitinds]

    mu1_vec_train <- rnorm(n = n, mean = mu0, sd = sigma_init/sqrt(k0))
    mu1_vec_test <- mu1_vec_train[testinitinds]
  }

  # mu2_vec_train <- rep(0,n)
  # mu2_vec_test <- rep(0,ntest)

  # print("Line 301")

  varthetamat <- cbind(mu1_vec_train, sigma1_vec_train)
  varthetamat_test <- cbind(mu1_vec_test, sigma1_vec_test)

  # varthetamat <- cbind(mu1_vec_train, sigma1_vec_train)


  if(n_censbelow > 0){
    z[which(y <= below_cens )] <- rtruncnorm(n_censbelow, a = -Inf, b = below_cens,
                                             mean = mu1_vec_train[censbelow_inds],
                                             sd = sigma1_vec_train[censbelow_inds])
  }
  if(n_censabove > 0){
    z[which(y >= above_cens )] <- rtruncnorm(n_censabove, a = above_cens, b = Inf,
                                             mean = mu1_vec_train[censabove_inds],
                                             sd = sigma1_vec_train[censabove_inds])
  }


  weightstemp_y  <- 1/(sigma1_vec_train^2)


  # print("Line 324")
  #
  # print("length(z) = ")
  # print(length(z) )
  # print("length(weightstemp_y) = ")
  # print(length(weightstemp_y) )
  #
  # print("weightstemp_y = ")
  # print(weightstemp_y)
  #
  # print("nrow(x.train) = ")
  # print(nrow(x.train))
  #
  # print("ncol(x.train) = ")
  # print(ncol(x.train))
  #
  # print("ncol(x.test) = ")
  # print(ncol(x.test))
  #
  # print("z = ")
  # print(z)

  # dftrain <- data.frame(y00 = z - 0, x00 = x.train)
  #
  # # print("dftrain = ")
  # # print(dftrain)
  # #
  # # print("lm(y00~., data = dftrain, weights = weightstemp_y)")
  # # print(lm(y00 ~., data = dftrain, weights = weightstemp_y))
  #
  #
  #
  # if(nrow(x.test ) == 0){
  #   sampler <- dbarts(y00 ~ . ,
  #                     data = dftrain,
  #                     #test = x.test,
  #                     weights = weightstemp_y,
  #                     control = control,
  #                     tree.prior = tree.prior,
  #                     node.prior = node.prior,
  #                     resid.prior = resid.prior,
  #                     proposal.probs = proposal.probs,
  #                     sigma = 1
  #   )
  #
  # }else{
  #   dftest <- data.frame(y00 = NA, x00 = x.test)
  #
  #   # print("dftest = ")
  #   # print(dftest)
  #   #
  #   # print("length(weightstemp_y) = ")
  #   # print(length(weightstemp_y))
  #
  #   # weightstemp_testy  <- 1/(sigma1_vec_test^2)
  #
  #   sampler <- dbarts(y00 ~ . ,
  #                     data = dftrain,
  #                     test = dftest,
  #                     # weights = weightstemp_y,
  #                     control = control,
  #                     tree.prior = tree.prior,
  #                     node.prior = node.prior,
  #                     resid.prior = resid.prior,
  #                     proposal.probs = proposal.probs,
  #                     sigma = 1
  #   )
  #
  # }



  hypers <- Hypers(x.train, y,
                   num_tree = n.trees, sigma_hat = 1,
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


  opts <- Opts(update_sigma = FALSE, num_print = print.opt)


  sampler_forest <- MakeForest(hypers, opts, warn = FALSE)


  sampler_forest$set_sigma(1)

  mu <- sampler_forest$do_gibbs_weighted(x.train, z, weightstemp_y, x.train, 1)
  mutest <- sampler_forest$do_predict(x.test)



  # print("Line 350")

  # sampler$setResponse(y = z - mu1_vec_train)
  # sampler$setSigma(sigma = 1)
  # sampler$setWeights(weights = weightstemp_y)
  #
  # #sampler$setPredictor(x= Xmat.train$x, column = 1, forceUpdate = TRUE)
  #
  # #mu = as.vector( alpha + X.mat %*% beta )
  # sampler$sampleTreesFromPrior()
  # samplestemp <- sampler$run()
  #
  # #mutemp <- samplestemp$train[,1]
  # #suppose there are a number of samples
  #
  # # print("sigma = ")
  # sigma <- samplestemp$sigma
  #
  # mu <- samplestemp$train[,1]
  # mutest <- samplestemp$test[,1]

  # mu <- sampler$predict(dftrain)
  #
  #
  # if(nrow(x.test ) == 0){
  #
  #   mutest <- samplestemp$test[,1]
  # }else{
  #   mutest <- sampler$predict(dftest)
  # }

  ystar <- rnorm(n,mean = mu + mu1_vec_train, sd = sigma1_vec_train)


  # must draw test parameter values
  # test_clusts <-  sample(0:n, size = ntest, replace = TRUE, prob = c(alpha/(alpha+n), rep(1/(alpha+n) ,n) ))
  test_clusts <-  sample.int(n+1, size = ntest, replace = TRUE, prob = c(alpha/(alpha+n), rep(1/(alpha+n) ,n) ))-1
  # test_clusts <-  dqsample.int(n+1, size = ntest, replace = TRUE, prob = c(alpha/(alpha+n), rep(1/(alpha+n) ,n) ))-1


  if(sum(test_clusts >0) == 0){
    # skip
  }else{
    sigma1_vec_test[test_clusts > 0] <- sigma1_vec_train[test_clusts[test_clusts > 0]  ]
    mu1_vec_test[test_clusts > 0] <- mu1_vec_train[test_clusts[test_clusts > 0]  ]
  }

  if(sum(test_clusts ==0) == 0){
    # skip
  }else{
    # draw from prior
    numzeros <- sum(test_clusts == 0)

    sigma1_vec_test[test_clusts == 0] <- sqrt(1/rgamma(n = numzeros, shape =  nu0/2, rate = nu0*lambda0/2) )

    mu1_vec_test[test_clusts == 0] <- rnorm(n = numzeros, mean = mu0, sd = sigma1_vec_test[test_clusts == 0]/sqrt(k0))

  }


  # print("Line 398")

  ystartest <- rnorm(ntest,
                     mean = mutest + mu1_vec_test,
                     sd = sigma1_vec_test)

  # ystartestcens <- rtruncnorm(ntest,
  #                            a = below_cens, b = above_cens,
  #                            mean = mutest + mu1_vec_test,
  #                            sd = sigma1_vec_test)


  ystartestcens <- ystartest
  ystartestcens[ystartest < below_cens] <- below_cens
  ystartestcens[ystartest > above_cens] <- above_cens


  probcensbelow_train <- pnorm(below_cens, mean = mu + mu1_vec_train, sd = sigma1_vec_train)
  probcensabove_train <- 1 - pnorm(above_cens, mean = mu + mu1_vec_train, sd = sigma1_vec_train)


  probcensbelow <- pnorm(below_cens, mean = mutest + mu1_vec_test, sd = sigma1_vec_test)
  probcensabove <- 1 - pnorm(above_cens, mean = mutest + mu1_vec_test, sd = sigma1_vec_test)


  if(below_cens == - Inf){
    if(above_cens == Inf){
      condexptrain <- (mu + mu1_vec_train)

      condexptest <- (mutest + mu1_vec_test)
    }else{ # above_cens !=Inf
      condexptrain <-
        (mu + mu1_vec_train)*(1- probcensabove_train ) +
        sigma1_vec_train*(  - fastnormdens(above_cens, mean = mu + mu1_vec_train, sd = sigma1_vec_train) ) +
        above_cens*probcensabove_train

      condexptest <-
        (mutest + mu1_vec_test)*(1- probcensabove ) +
        sigma1_vec_test*(  -fastnormdens(above_cens, mean = mutest + mu1_vec_test, sd = sigma1_vec_test) ) +
        above_cens*probcensabove
    }
  }else{ # below_cens != - Inf
    if(above_cens == Inf){
      condexptrain <- below_cens*probcensbelow_train +
        (mu + mu1_vec_train)*(1 - probcensbelow_train) +
        sigma1_vec_train*( fastnormdens(below_cens, mean = mu + mu1_vec_train, sd = sigma1_vec_train)  )

      condexptest <- below_cens*probcensbelow +
        (mutest + mu1_vec_test)*(1 - probcensbelow) +
        sigma1_vec_test*( fastnormdens(below_cens, mean = mutest + mu1_vec_test, sd = sigma1_vec_test)  )


    }else{ # above_cens !=Inf
      condexptrain <- below_cens*probcensbelow_train +
        (mu + mu1_vec_train)*(1- probcensabove_train - probcensbelow_train) +
        sigma1_vec_train*( fastnormdens(below_cens, mean = mu + mu1_vec_train, sd = sigma1_vec_train) -
                             fastnormdens(above_cens, mean = mu + mu1_vec_train, sd = sigma1_vec_train) ) +
        above_cens*probcensabove_train

      condexptest <- below_cens*probcensbelow +
        (mutest + mu1_vec_test)*(1- probcensabove - probcensbelow) +
        sigma1_vec_test*( fastnormdens(below_cens, mean = mutest + mu1_vec_test, sd = sigma1_vec_test) -
                            fastnormdens(above_cens, mean = mutest + mu1_vec_test, sd = sigma1_vec_test) ) +
        above_cens*probcensabove
    }
  }



  #save the first round of values
  if(n.burnin == 0){
    draw$Z.mat[,1] = z
    draw$Z.matcens[,1] = z[cens_inds]
    # draw$Z.matuncens[,1] = z[uncens_inds]
    draw$Z.matcensbelow[,1] = z[censbelow_inds]
    draw$Z.matcensabove[,1] = z[censabove_inds]
    draw$mu[,1] = mu
    draw$mucens[,1] = mu[cens_inds]
    draw$muuncens[,1] = mu[uncens_inds]
    draw$mucensbelow[,1] = mu[censbelow_inds]
    draw$mucensabove[,1] = mu[censabove_inds]
    draw$ystar[,1] = ystar
    draw$ystarcens[,1] = ystar[cens_inds]
    draw$ystaruncens[,1] = ystar[uncens_inds]
    draw$ystarcensbelow[,1] = ystar[censbelow_inds]
    draw$ystarcensabove[,1] = ystar[censabove_inds]
    draw$test.mu[,1] = mutest
    draw$test.y_nocensoring[,1] = ystartest
    draw$test.y_withcensoring[,1] = ystartestcens
    draw$test.probcensbelow[,1] = probcensbelow
    draw$test.probcensabove[,1] = probcensabove
    # draw$sigma[1] = sigma
    draw$sigmavecs_train[, 1] = sigma1_vec_train
    draw$sigmavecs_test[, 1] = sigma1_vec_test
    draw$error_mu_train[, 1] = mu1_vec_train
    draw$error_mu_test[, 1] = mu1_vec_test
    draw$cond_exp_train[, 1] = condexptrain
    draw$cond_exp_test[, 1] = condexptest
    draw$alpha[1] = 1
  }


  # print("Begin For loop")
  ###### Begin for loop ##################
  #loop through the Gibbs sampler iterations
  for(iter in 2:(n.iter+n.burnin)){

    ########### 2 draw the latent outcome####################
    # z[cens_inds] <- rtruncnorm(n0, a= below_cens, b = above_cens, mean = mu[cens_inds], sd = sigma)
    if(length(censbelow_inds) > 0){
      z[censbelow_inds] <- rtruncnorm(n_censbelow, a= -Inf, b = below_cens,
                                      mean = mu[censbelow_inds] + mu1_vec_train[censbelow_inds],
                                      sd = sigma1_vec_train[censbelow_inds])
    }
    if(length(censabove_inds) > 0){
      z[censabove_inds] <- rtruncnorm(n_censabove, a= above_cens, b = Inf,
                                      mean = mu[censabove_inds] + mu1_vec_train[censabove_inds],
                                      sd = sigma1_vec_train[censabove_inds])
    }


    ######## 3 Draw trees ######################
    # print("3 Draw trees")

    #set the response.
    #Check that 0 is a reasonable initial value
    #perhaps makes more sense to use initial values of Z
    # sampler$setResponse(y = z)

    # weightstemp_y  <- 1/sigma1_vec_train
    weightstemp_y  <- 1/(sigma1_vec_train^2)


    sampler_forest$set_sigma(1)

    mu <- sampler_forest$do_gibbs_weighted(x.train, z- mu1_vec_train, weightstemp_y, x.train, 1)
    mutest <- sampler_forest$do_predict(x.test)


    # sampler$setResponse(y = z- mu1_vec_train)
    # sampler$setSigma(sigma = 1)
    # sampler$setWeights(weights = weightstemp_y)
    #
    #
    # # sampler$setSigma(sigma = 1)
    # #sampler$setPredictor(x= Xmat.train$x, column = 1, forceUpdate = TRUE)
    #
    # #mu = as.vector( alpha + X.mat %*% beta )
    # samplestemp <- sampler$run()
    #
    # # sigma <- samplestemp$sigma
    #
    # # mu <- samplestemp$train[,1]
    # # mutest <- samplestemp$test[,1]
    #
    #
    # mu <- samplestemp$train[,1]
    # mutest <- samplestemp$test[,1]

    # mu <- sampler$predict(dftrain)
    #
    #
    # if(nrow(x.test ) == 0){
    #
    #   mutest <- samplestemp$test[,1]
    # }else{
    #   mutest <- sampler$predict(dftest)
    # }



    ######### 4 draw components of mixture ####################
    # print(" 4 draw components of mixture")

    qi0vec <- alpha*dt_ls(z, df = nu0, mu = mu, sigma =  sqrt(lambda0*(1 + (1/k0))) )

    #loop over individuals for updates
    for(i in 1:n){

      # qi0 <- alpha*dt_ls(z[i], df = nu0, mu = mu[i], sigma =  lambda0*(1 + (1/k0)) )

      qi0 <- qi0vec[i]

      # check if should drop just i^th row or all in same cluster as i.

      varthetamattemp <- varthetamat[-i,, drop = FALSE]
      # vartheta_unique_mat <- unique(varthetamattemp)

      tempcol <- varthetamattemp[,1, drop = FALSE]

      # THERE IS PROBABLY A MUCH FASTER WAY OF DOING THIS JUST BY UPDATING IN ITERATION
      # BY ACCOUNTING FOR LAST INCLUDED NUMBER AND DROPPPED ROW i

      tempord <- order(tempcol, method = "radix")
      tempsort <- tempcol[tempord]
      tempuniinds <- tempord[!duplicated(tempsort)]
      # tempuniinds <- unique(match(tempsort,tempcol))
      counts_ord <- rle2(tempsort)[,2]
      # counts_ord <- rle(tempsort)$lengths

      # tempsort <- sort(tempcol, na.last = TRUE)
      # # tempuniinds <- order(tempcol)[!duplicated(tempsort)]
      # tempuniinds <- unique(match(tempsort,tempcol))
      # counts_ord <- rle(tempsort)$lengths

      # tempuniinds <- order(tempcol)[!duplicated(sort(tempcol))]
      # temprle <- rle(sort(tempcol))$lengths
      # counts_ord <- fastcountuniq(tempcol)

      # CHECK THESE LINES
      # ux = sort(unique(tempcol)) #vartheta_unique_mat[,1, drop = FALSE] #sort(unique(tempcol))
      # idx = match(tempcol, ux)
      # tempuniinds <- unique(idx)
      # counts_ord = tabulate(idx, nbins=length(ux))


      # tempord <- order(ux[tempuniinds])




      vartheta_unique_mat <- varthetamattemp[tempuniinds, , drop = FALSE]


      #also try split(seq_along(vec), vec)


      # tempord <- order(unique(varthetamattemp[,1 , drop = FALSE]))
      #         tempord <- order(vartheta_unique_mat[,1 , drop = FALSE])


      # print("vartheta_unique_mat = ")
      # print(vartheta_unique_mat)


      # print("tempord = ")
      # print(tempord)
      #          vartheta_unique_mat <- vartheta_unique_mat[tempord, , drop = FALSE]


      # print("vartheta_unique_mat = ")
      # print(vartheta_unique_mat)

      #       tempcol <- varthetamattemp[,1, drop = FALSE]

      # counts_ord <- table(varthetamattemp[,1, drop = FALSE])

      #        ux = vartheta_unique_mat[,1, drop = FALSE] #sort(unique(tempcol))
      # ux = sort(unique(tempcol))
      # print("ux = ")
      # print(ux)
      #        idx = match(tempcol, ux)

      # print("idx = ")
      # print(idx)
      #
      # print("length(ux)")
      # print(length(ux))
      # counts_ord = tabulate(idx, nbins=length(ux))




      num_unique <- nrow(vartheta_unique_mat)

      # print("counts_ord = ")
      # print(counts_ord)

      # print("num_unique = ")
      # print(num_unique)

      if( sum(counts_ord) != n-1 ){
        print("tempcol = ")
        print(tempcol)
        print("fastcountuniq(1:5) = ")
        print(fastcountuniq(1:5))
        print("fastcountuniq(tempcol) = ")
        print(fastcountuniq(tempcol))
        print("counts_ord = ")
        print(counts_ord)
        print("num_unique = ")
        print(num_unique)
        print("n-1 = ")
        print(n-1)
        print("sum(counts_ord) = ")
        print(sum(counts_ord))

        stop("Bug in count of unique values")
      }

      if( length(counts_ord) != num_unique ){
        print("counts_ord = ")
        print(counts_ord)
        print("num_unique = ")
        print(num_unique)

        stop("Bug in count of unique values")
      }


      # if(any(is.na(counts_ord))){
      #
      #   print("counts_ord = ")
      #   print(counts_ord)
      #   print("tempuniinds = ")
      #   print(tempuniinds)
      #   stop("counts_ord NA")
      # }


      q_rs <- rep(NA, nrow(vartheta_unique_mat))

      # CHECK THIS AND SPEED IT UP
      q_rs <- counts_ord*fastnormdens(z[i],
                                      mean = mu[i] + vartheta_unique_mat[,1],
                                      sd =  vartheta_unique_mat[,2])


      # for(j in 1:num_unique){
      #   q_rs[j] <- counts_ord[j]*fastnormdens(z[i],
      #                                  mean = vartheta_unique_mat[j,1],
      #                                  sd =  vartheta_unique_mat[j,2])
      # }
      # if(is.na(qi0)){
      #
      #   print("qi0 = ")
      #   print(qi0)
      #   print("q_rs = ")
      #   print(q_rs)
      #   print("counts_ord = ")
      #   print(counts_ord)
      #   print("tempuniinds = ")
      #   print(tempuniinds)
      #   print("vartheta_unique_mat = ")
      #   print(vartheta_unique_mat)
      #   stop("qi0 NA before normalize")
      # }

      tempdemon <- qi0 + sum(q_rs)

      qi0 <- qi0/tempdemon
      q_rs <- q_rs/tempdemon

      # if(is.na(qi0)){
      #
      #   print("qi0 = ")
      #   print(qi0)
      #   print("q_rs = ")
      #   print(q_rs)
      #   print("counts_ord = ")
      #   print(counts_ord)
      #   print("tempuniinds = ")
      #   print(tempuniinds)
      #   print("vartheta_unique_mat = ")
      #   print(vartheta_unique_mat)
      #   stop("qi0 NA")
      # }
      # if(any(is.na(q_rs))){
      #
      #   print("qi0 = ")
      #   print(qi0)
      #   print("q_rs = ")
      #   print(q_rs)
      #   stop("q_rs NA")
      # }

      # SPEED UP THIS STEP
      # rprime <- sample(0:num_unique, size = 1, replace = TRUE, prob = c(qi0, q_rs))
      # rprime <- sample.int(num_unique+1, size = 1, replace = TRUE, prob = c(qi0, q_rs))-1
      rprime <- sample_int_expj(num_unique+1, size = 1, prob = c(qi0, q_rs))-1
      # rprime <- dqsample.int(num_unique+1, size = 1, replace = TRUE, prob = c(qi0, q_rs))-1

      if(rprime>0){

        # if none equal to current varthetamat[i,] and there are more than one equal to i+1^th row
        # then can keep same vartheta_unique_mat in next iteration and it is straightforward to update the counts.
        # not implemented yet

        varthetamat[i,] <- vartheta_unique_mat[rprime,]
        mu1_vec_train[i] <- varthetamat[i,1]
        sigma1_vec_train[i] <- varthetamat[i,2]

      }else{

        # speed up these samples
        varthetamat[i,2] <- sqrt(1/rgamma(n = 1,
                                          shape =  (nu0+1)/2,
                                          rate = (nu0*lambda0/2) + (z[i] -  mu[i] )^2/( 2*(1 + 1/k0))  ) )

        # speed this up
        varthetamat[i,1] <- rnorm(n=1,
                                  mean =  (z[i] -  mu[i] )/(k0+1) ,
                                  sd =  varthetamat[i,2]/sqrt(k0+1) )

        mu1_vec_train[i] <- varthetamat[i,1]
        sigma1_vec_train[i] <- varthetamat[i,2]
      } # end else statement corresponding to rprime == 0
    } # end loop over i that updates components



    ######### 5 mixing step ##########################
    # print("5 mixing step")

    if(mixstep == TRUE){
      vartheta_unique_mat <- unique(varthetamat)

      for(j in 1:nrow(vartheta_unique_mat)){

        #speed up sampling if possible
        clust_inds <- which(varthetamat[,1, drop = FALSE] == vartheta_unique_mat[j,1])

        n_j <- length(clust_inds)

        clust_mean <- mean(z[clust_inds] -  mu[clust_inds])

        varthetamat[clust_inds,2] <- sqrt(1/rgamma(n = 1,#n_j,
                                                   shape =  (nu0+n_j)/2,
                                                   rate = (nu0*lambda0/2) +
                                                     sum((z[clust_inds] -  mu[clust_inds] - clust_mean )^2)/2 +
                                                     (k0*n_j/( k0 + n_j) )*( clust_mean^2 / 2) ) )

        varthetamat[clust_inds,1] <- rnorm(n=1,#n_j,
                                           mean =  (n_j * clust_mean )/(k0+n_j) ,
                                           sd =  varthetamat[clust_inds,2]/sqrt(k0+n_j) )

        mu1_vec_train[clust_inds] <- varthetamat[clust_inds,1]
        sigma1_vec_train[clust_inds] <- varthetamat[clust_inds,2]
      }
    }

    ######### 6 sample alpha ######################
    # print("6 sample alpha")

    # vartheta_unique_mat <- unique(varthetamat)


    # This step can be implemented using the prior of van Hasselt (2011)
    # or the prior of George et al. (2019), McCulloch et al. (2021) ans Conley et al. (2008)

    # count number of unique mixture components
    # there is probably a more efficient way of doing this

    # create a matrix in which each row contains all of an individual's mixture component parameters

    # varthetamat <- cbind(mu1_vec_train, mu2_vec_train, phi1_vec_train, gamma1_vec_train)
    # varthetamat_test <- cbind(mu1_vec_test, mu2_vec_test, phi1_vec_test, gamma1_vec_test)

    #obtain the unique rows (components)
    vartheta_unique_mat <- unique(varthetamat)

    #number of unique components
    k_uniq <- nrow(vartheta_unique_mat)


    if(alpha_prior == "vh"){

      #########  VH Step 6 a: Sample auxiliary variable kappa  ######################################################

      kappa_aux <- rbeta(n = 1, shape1 = alpha+1, shape2 = n)

      #########  VH Step 6 b: Sample alpha from a mixture of gamma distributions  ######################################################

      #obtain the mixing probability
      p_kappa <- (c1+k_uniq-1)/(n*(c2-log(kappa_aux))+c1+k_uniq-1)

      #sample a mixture component
      mix_draw <- rbinom(n = 1,size = 1,prob = p_kappa)

      # draw alpha from the drawn component
      if(mix_draw==1){
        alpha <- rgamma(1,shape = c1 + k_uniq, rate = c2 - log(kappa_aux))
      }else{
        alpha <- rgamma(1,shape = c1 + k_uniq - 1, rate = c2 - log(kappa_aux))

      }


    }else{
      if(alpha_prior == "george"){

        #Calculate alpha_min and alpha_max
        Imin <- 1
        Imax <- floor(0.1*n)

        #consider IVBART prior
        # Imin <- 2
        # Imax <- floor(0.1*n)+1


        if(floor(0.1*n) <= 1){
          stop("Not enough observations for Prior of George et al. (2019)")
        }

        psiprior <- 0.5

        # from ivbart package code
        # https://github.com/rsparapa/bnptools/blob/master/ivbart/R/amode.R
        egamm <- 0.5772156649
        tempmin <- digamma(Imin) - log(egamm+log(n))
        tempmax <- digamma(Imax) - log(egamm+log(n))

        alpha_min <- exp(tempmin)
        alpha_max <- exp(tempmax)
        alpha_values <-  seq(from = alpha_min,
                             to = alpha_max,
                             length.out = alpha_gridsize)
        temp_aprior <- 1 - (alpha_values-alpha_min)/(alpha_max-alpha_min)
        temp_aprior <- temp_aprior^psiprior
        # temp_aprior = temp_aprior/sum(temp_aprior)


        log_tempvals <- k_uniq*log(alpha_values) + lgamma(alpha_values) - lgamma(n+alpha_values)

        # print("log_tempvals = ")
        # print(log_tempvals)
        # print("temp_aprior = ")
        # print(temp_aprior)

        temp_kgivenalpha <- exp(log_tempvals)

        # temp_kgivenalpha <- ((alpha_values)^(k_uniq))*gamma(alpha_values)/gamma(n+alpha_values)
        temp_alpha_postprobs <- temp_kgivenalpha*temp_aprior


        # print("temp_kgivenalpha = ")
        # print(temp_kgivenalpha)


        # print("gamma(alpha_values) = ")
        # print(gamma(alpha_values))
        #
        #
        # print("gamma(n+alpha_values) = ")
        # print(gamma(n+alpha_values))
        #
        #
        # print("alpha_values = ")
        # print(alpha_values)
        #
        # print("temp_kgivenalpha = ")
        # print(temp_kgivenalpha)

        # print("temp_aprior = ")
        # print(temp_aprior)
        #
        # print("temp_alpha_postprobs = ")
        # print(temp_alpha_postprobs)

        post_probs_alphs <- temp_alpha_postprobs/sum(temp_alpha_postprobs)

        # print("post_probs_alphs = ")
        # print(post_probs_alphs)

        #sample from 1 to alpha_gridsize
        # index_alpha <- sample.int(n = alpha_gridsize, size = 1, prob = post_probs_alphs, replace = TRUE)
        index_alpha <- sample_int_expj(n = alpha_gridsize, size = 1, prob = post_probs_alphs)


        alpha <- alpha_values[index_alpha]


      }else{
        stop("Alpha prior must be vh or george")
      }

    } # end of George et al. alpha draw code




    ########### draw y value predictions ##############################
    # not sure where to make these draws (relative to other steps)
    # print("draw y value predictions ")

    # must draw test parameter values
    # test_clusts <-  sample(0:n, size = ntest, replace = TRUE, prob = c(alpha/(alpha+n), rep(1/(alpha+n) ,n) ))
    test_clusts <-  sample.int(n+1, size = ntest, replace = TRUE, prob = c(alpha/(alpha+n), rep(1/(alpha+n) ,n) ))-1
    # test_clusts <-  dqsample.int(n+1, size = ntest, replace = TRUE, prob = c(alpha/(alpha+n), rep(1/(alpha+n) ,n) ))-1

    if(sum(test_clusts >0) == 0){
      # skip
    }else{
      sigma1_vec_test[test_clusts > 0] <- sigma1_vec_train[test_clusts[test_clusts > 0]  ]
      mu1_vec_test[test_clusts > 0] <- mu1_vec_train[test_clusts[test_clusts > 0]  ]
    }

    if(sum(test_clusts == 0) == 0){
      # skip
    }else{
      # draw from prior
      numzeros <- sum(test_clusts == 0)

      sigma1_vec_test[test_clusts == 0] <- sqrt(1/rgamma(n = numzeros, shape =  nu0/2, rate = nu0*lambda0/2) )

      mu1_vec_test[test_clusts == 0] <- rnorm(n = numzeros, mean = mu0, sd = sigma1_vec_test[test_clusts == 0]/sqrt(k0))

    }


    #draw uncensored predictions of y
    ystar <- rnorm(n,mean = mu+ mu1_vec_train , sd = sigma1_vec_train)
    ystartest <- rnorm(ntest,mean = mutest + mu1_vec_test, sd = sigma1_vec_test)

    # ystartestcens <- rtruncnorm(ntest, a= below_cens, b= above_cens, mean = mutest, sd = sigma)

    ystartestcens <- ystartest
    ystartestcens[ystartest < below_cens] <- below_cens
    ystartestcens[ystartest > above_cens] <- above_cens


    probcensbelow_train <- pnorm(below_cens, mean = mu + mu1_vec_train, sd = sigma1_vec_train)
    probcensabove_train <- 1 - pnorm(above_cens, mean = mu + mu1_vec_train, sd = sigma1_vec_train)


    probcensbelow <- pnorm(below_cens, mean = mutest + mu1_vec_test, sd = sigma1_vec_test)
    probcensabove <- 1 - pnorm(above_cens, mean = mutest + mu1_vec_test, sd = sigma1_vec_test)


    if(below_cens == - Inf){
      if(above_cens == Inf){
        condexptrain <- (mu + mu1_vec_train)

        condexptest <- (mutest + mu1_vec_test)
      }else{ # above_cens !=Inf
        condexptrain <-
          (mu + mu1_vec_train)*(1 - probcensabove_train ) +
          sigma1_vec_train*(  - fastnormdens(above_cens, mean = mu + mu1_vec_train, sd = sigma1_vec_train) ) +
          above_cens*probcensabove_train

        condexptest <-
          (mutest + mu1_vec_test)*(1 - probcensabove ) +
          sigma1_vec_test*(  - fastnormdens(above_cens, mean = mutest + mu1_vec_test, sd = sigma1_vec_test) ) +
          above_cens*probcensabove
      }
    }else{ # below_cens != - Inf
      if(above_cens == Inf){
        condexptrain <- below_cens*probcensbelow_train +
          (mu + mu1_vec_train)*(1 - probcensbelow_train) +
          sigma1_vec_train*( fastnormdens(below_cens, mean = mu + mu1_vec_train, sd = sigma1_vec_train)  )

        condexptest <- below_cens*probcensbelow +
          (mutest + mu1_vec_test)*(1 - probcensbelow) +
          sigma1_vec_test*( fastnormdens(below_cens, mean = mutest + mu1_vec_test, sd = sigma1_vec_test)  )


      }else{ # above_cens !=Inf
        condexptrain <- below_cens*probcensbelow_train +
          (mu + mu1_vec_train)*(1 - probcensabove_train - probcensbelow_train) +
          sigma1_vec_train*( fastnormdens(below_cens, mean = mu + mu1_vec_train, sd = sigma1_vec_train) -
                               fastnormdens(above_cens, mean = mu + mu1_vec_train, sd = sigma1_vec_train) ) +
          above_cens*probcensabove_train

        condexptest <- below_cens*probcensbelow +
          (mutest + mu1_vec_test)*(1 - probcensabove - probcensbelow) +
          sigma1_vec_test*( fastnormdens(below_cens, mean = mutest + mu1_vec_test, sd = sigma1_vec_test) -
                              fastnormdens(above_cens, mean = mutest + mu1_vec_test, sd = sigma1_vec_test) ) +
          above_cens*probcensabove
      }
    }



    if(iter > n.burnin){
      iter_min_burnin <- iter-n.burnin
      draw$Z.mat[,iter_min_burnin] = z
      draw$Z.matcens[,iter_min_burnin] = z[cens_inds]
      # draw$Z.matuncens[,iter_min_burnin] = z[uncens_inds]
      draw$Z.matcensbelow[,iter_min_burnin] = z[censbelow_inds]
      draw$Z.matcensabove[,iter_min_burnin] = z[censabove_inds]
      draw$mu[,iter_min_burnin] = mu + mu1_vec_train
      draw$mucens[,iter_min_burnin] = mu[cens_inds] + mu1_vec_train[cens_inds]
      draw$muuncens[,iter_min_burnin] = mu[uncens_inds] + mu1_vec_train[uncens_inds]
      draw$mucensbelow[,iter_min_burnin] = mu[censbelow_inds] + mu1_vec_train[censbelow_inds]
      draw$mucensabove[,iter_min_burnin] = mu[censabove_inds] + mu1_vec_train[censabove_inds]
      draw$ystar[,iter_min_burnin] = ystar
      draw$ystarcens[,iter_min_burnin] = ystar[cens_inds]
      draw$ystaruncens[,iter_min_burnin] = ystar[uncens_inds]
      draw$ystarcensbelow[,iter_min_burnin] = ystar[censbelow_inds]
      draw$ystarcensabove[,iter_min_burnin] = ystar[censabove_inds]
      draw$test.mu[,iter_min_burnin] = mutest + mu1_vec_test
      draw$test.y_nocensoring[,iter_min_burnin] = ystartest
      draw$test.y_withcensoring[,iter_min_burnin] = ystartestcens
      draw$test.probcensbelow[,iter_min_burnin] = probcensbelow
      draw$test.probcensabove[,iter_min_burnin] = probcensabove
      draw$sigmavecs_train[, iter_min_burnin] = sigma1_vec_train
      draw$sigmavecs_test[, iter_min_burnin] = sigma1_vec_test
      draw$error_mu_train[, iter_min_burnin] = mu1_vec_train
      draw$error_mu_test[, iter_min_burnin] = mu1_vec_test

      draw$cond_exp_train[, iter_min_burnin] = condexptrain
      draw$cond_exp_test[, iter_min_burnin] = condexptest
      draw$alpha[iter_min_burnin] <- alpha
    }

    if(iter %% print.opt == 0){
      print(paste("Gibbs Iteration", iter))
      # print(c(sigma2.alpha, sigma2.beta))
    }


  }#end iterations of Gibbs sampler



  return(draw)



}

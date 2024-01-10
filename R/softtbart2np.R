

#' @title Nonparametric Type II Tobit Soft Bayesian Additive Regression Trees with sparsity inducing hyperprior implemented using MCMC
#'
#' @description Nonparametric Type II Tobit Soft Bayesian Additive Regression Trees with sparsity inducing hyperprior implemented using MCMC. The errors in the selection and outcome equations are modelled by a Dirichlet Process mixture of bivariate normal distributions.
#' @import dbarts
#' @import truncnorm
#' @import MASS
#' @import GIGrvg
#' @import mvtnorm
#' @import Rfast
#' @param x.train The outcome model training covariate data for all training observations. Number of rows equal to the number of observations. Number of columns equal to the number of covariates.
#' @param x.test The outcome model test covariate data for all test observations. Number of rows equal to the number of observations. Number of columns equal to the number of covariates.
#' @param w.train The censoring model training covariate data for all training observations. Number of rows equal to the number of observations. Number of columns equal to the number of covariates.
#' @param w.test The censoring model test covariate data for all test observations. Number of rows equal to the number of observations. Number of columns equal to the number of covariates.
#' @param y The training data vector of outcomes. A continuous, censored outcome variable. Censored observations must be included with values equal to censored_value
#' @param n.iter Number of iterations excluding burnin.
#' @param n.burnin Number of burnin iterations.
#' @param censored_value The value taken by censored observations
#' @param gamma0 The mean of the normal prior on the covariance of the errors in the censoring and outcome models.
#' @param G0 The variance of the normal prior on the covariance of the errors in the censoring and outcome models.
#' @param nzero A prior parameter which when divided by 2 gives the mean of the normal prior on phi, where phi*gamma is the variance of the errors of the outcome model.
#' @param S0 A prior parameter which when divided by 2 gives the variance of the normal prior on phi, where phi*gamma is the variance of the errors of the outcome model.
#' @param sigest If variance of the error term is the
#' @param n.trees_outcome (dbarts control option) A positive integer giving the number of trees used in the outcome model sum-of-trees formulation.
#' @param n.trees_censoring (dbarts control option) A positive integer giving the number of trees used in the censoring model sum-of-trees formulation.
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
#' @param eq_by_eq If TRUE, implements sampler equation by equation (as in BAVART by Huber and Rossini (2021)). If FALSE, implements sampler in similar approach to SUR-BART (Chakraborty 2016) or MPBART (Kindo 2016).
#' @param accelerate If TRUE, add extra parameter for accelerated sampler as descibed by Omori (2007).
#' @param cov_prior Prior for the covariance of the error terms. If VH, apply the prior of van Hasselt (2011), N(gamma0, tau*phi), imposing dependence between gamma and phi. If Omori, apply the prior N(gamma0,G0). If mixture, then a mixture of the VH and Omori priors with probability mixprob applied to the VH prior.
#' @param tau Parameter for the prior of van Hasselt (2011) on the covariance of the error terms.
#' @param M_mat Base distribution covariace for errors in outcome and selection equation for Dirichlet Process mixture.
#' @param alpha_prior The prior for the alpha parameter of the Dirichlet Process mixture of normals. If "vh" then apply the Gamma(c1,c2) prior of van Hasselt (2011) and Escobar (1994). If "george", then apply the prior of George (2019), McCulloch (2021), Conley (2008), and Antoniak (1974).
#' @param c1 If alpha_prior == "vh", then c1 is the shape parameter of the Gamma distribution.
#' @param c2 If alpha_prior == "vh", then c2 is the rate parameter of the Gamma distribution.
#' @param alpha_gridsize If alpha_prior = "george", this is the size of the grid to use for the discretized samples of alpha
#' @export
#' @return The following objects are returned:
#' \item{Z.mat_train}{Matrix of draws of censoring model latent outcomes for training observations. Number of rows equals number of training observations. Number of columns equals n.iter . Rows are ordered in order of observations in the training data.}
#' \item{Z.mat_test}{Matrix of draws of censoring model latent outcomes for test observations. Number of rows equals number of test observations. Number of columns equals n.iter . Rows are ordered in order of observations in the test data.}
#' \item{Y.mat_train}{Matrix of draws of outcome model latent outcomes for training observations. Number of rows equals number of training observations. Number of columns equals n.iter . Rows are ordered in order of observations in the training data.}
#' \item{Y.mat_test}{Matrix of draws of outcome model latent outcomes for test observations. Number of rows equals number of test observations. Number of columns equals n.iter . Rows are ordered in order of observations in the test data.}
#' \item{mu_y_train}{Matrix of draws of the outcome model sums of terminal nodes, i.e. f(x_i), for all training observations. Number of rows equals number of training observations. Number of columns equals n.iter .}
#' \item{mu_y_test}{Matrix of draws of the outcome model sums of terminal nodes, i.e. f(x_i), for all test observations. Number of rows equals number of test observations. Number of columns equals n.iter .}
#' \item{mucens_y_train}{Matrix of draws of the outcome model sums of terminal nodes, i.e. f(x_i), for all censored training observations. Number of rows equals number of censored training observations. Number of columns equals n.iter .}
#' \item{muuncens_y_train}{Matrix of draws of the outcome model sums of terminal nodes, i.e. f(x_i), for all uncensored training observations. Number of rows equals number of uncensored training observations. Number of columns equals n.iter .}
#' \item{mu_z_train}{Matrix of draws of the censoring model sums of terminal nodes, i.e. f(w_i), for all training observations. Number of rows equals number of training observations. Number of columns equals n.iter .}
#' \item{mu_z_test}{Matrix of draws of the censoring model sums of terminal nodes, i.e. f(w_i), for all test observations. Number of rows equals number of test observations. Number of columns equals n.iter .}
#' \item{train.probcens}{Matrix of draws of probabilities of training sample observations being censored. Number of rows equals number of training observations. Number of columns equals n.iter .}
#' \item{test.probcens}{Matrix of draws of probabilities of test sample observations being censored. Number of rows equals number of test observations. Number of columns equals n.iter .}
#' \item{cond_exp_train}{Matrix of draws of the conditional (i.e. possibly censored) expectations of the outcome for all training observations. Number of rows equals number of training observations. Number of columns equals n.iter .}
#' \item{cond_exp_test}{Matrix of draws of the conditional (i.e. possibly censored) expectations of the outcome for all test observations. Number of rows equals number of test observations. Number of columns equals n.iter .}
#' \item{uncond_exp_train}{Only defined if censored_value is a number. Matrix of draws of the unconditional (i.e. possibly censored) expectations of the outcome for all training observations. Number of rows equals number of training observations. Number of columns equals n.iter .}
#' \item{uncond_exp_test}{Only defined if censored_value is a number. Matrix of draws of the unconditional (i.e. possibly censored) expectations of the outcome for all test observations. Number of rows equals number of test observations. Number of columns equals n.iter .}
#' \item{ystar_train}{Matrix of training sample draws of the outcome assuming uncensored. Number of rows equals number of training observations. Number of columns equals n.iter .}
#' \item{ystar_test}{Matrix of test sample draws of the outcome assuming uncensored . Number of rows equals number of test observations. Number of columns equals n.iter .}
#' \item{zstar_train}{Matrix of training sample draws of the censoring model latent outcome. Number of rows equals number of training observations. Number of columns equals n.iter.}
#' \item{zstar_test}{Matrix of test sample draws of the censoring model latent outcome. Number of rows equals number of test observations. Number of columns equals n.iter.}
#' \item{ydraws_train}{Only defined if censored_value is a number. Matrix of training sample unconditional (i.e. possibly censored) draws of the outcome. Number of rows equals number of training observations. Number of columns equals n.iter .}
#' \item{ydraws_test}{Only defined if censored_value is a number. Matrix of test sample unconditional (i.e. possibly censored) draws of the outcome . Number of rows equals number of test observations. Number of columns equals n.iter .}
#' \item{ycond_draws_train}{List of training sample conditional (i.e. zstar >0 for draw) draws of the outcome. Number of rows equals number of training observations. Number of columns equals n.iter .}
#' \item{ycond_draws_test}{List of test sample conditional (i.e. zstar >0 for draw) draws of the outcome . Number of rows equals number of test observations. Number of columns equals n.iter .}
#' \item{Sigma_draws}{3 dimensional array of MCMC draws of the covariance matrix for the censoring and outcome error terms. The numbers of rows and columns equal are equal to 2. The first row and column correspond to the censoring model. The second row and column correspond to the outcome model. The number of slices equals n.iter . }
#' @examples
#'
#'#example taken from Zhang, J., Li, Z., Song, X., & Ning, H. (2021). Deep Tobit networks: A novel machine learning approach to microeconometrics. Neural Networks, 144, 279-296.
#'
#'
#'
#' #Type II tobit simulation
#'
#' num_train <- 5000
#'
#' #consider increasing the number of covariates
#'
#' Xmat_train <- matrix(NA,nrow = num_train,
#'                      ncol = 8)
#'
#' Xmat_train[,1] <- runif(num_train, min = -1, max = 1)
#' Xmat_train[,2] <- rf(num_train,20,20)
#' Xmat_train[,3] <- rbinom(num_train, size = 1, prob = 0.75)
#' Xmat_train[,4] <- rnorm(num_train, mean = 1, sd = 1)
#' Xmat_train[,5] <- rnorm(num_train)
#' Xmat_train[,6] <- rbinom(num_train, size = 1, prob = 0.5)
#' Xmat_train[,7] <- rf(num_train,20,200)
#' Xmat_train[,8] <- runif(num_train, min = 0, max = 2)
#'
#' #it would be better to test performance of the models when there is correlation in the error terms.
#' varepsilon1_train <- rnorm(num_train, mean = 0, sd = sqrt(0.00025))
#' varepsilon2_train <- rnorm(num_train, mean = 0, sd = sqrt(0.00025))
#'
#' y1star_train <- 1 - 0.75*Xmat_train[,1] + 0.75*Xmat_train[,2] -
#'   0.5*Xmat_train[,4] -  0.5*Xmat_train[,6] - 0.25*Xmat_train[,1]^2 -
#'   0.75*Xmat_train[,1]*Xmat_train[,4] - 0.25*Xmat_train[,1]*Xmat_train[,2] -
#'   1*Xmat_train[,1]*Xmat_train[,6] + 0.5*Xmat_train[,2]*Xmat_train[,6] +
#'   varepsilon1_train
#'
#' y2star_train <- 1 + 0.25*Xmat_train[,4] - 0.75*Xmat_train[,6] +
#'   0.5*Xmat_train[,7] + 0.25*Xmat_train[,8] +
#'   0.25*Xmat_train[,4]^2 + 0.75*Xmat_train[,7]^2 + 0.5*Xmat_train[,8]^2 -
#'   1*Xmat_train[,4]*Xmat_train[,6] + 0.5*Xmat_train[,4]*Xmat_train[,8] +
#'   1*Xmat_train[,6]*Xmat_train[,7] - 0.25*Xmat_train[,7]*Xmat_train[,8] +
#'   varepsilon2_train
#'
#' y2obs_train <- ifelse(y1star_train>0, y2star_train,0)
#'
#' #Type II tobit simulation
#'
#' num_test <- 5000
#'
#' #consider increasing the number of covariates
#'
#' Xmat_test <- matrix(NA,nrow = num_test,
#'                     ncol = 8)
#'
#' Xmat_test[,1] <- runif(num_test, min = -1, max = 1)
#' Xmat_test[,2] <- rf(num_test,20,20)
#' Xmat_test[,3] <- rbinom(num_test, size = 1, prob = 0.75)
#' Xmat_test[,4] <- rnorm(num_test, mean = 1, sd = 1)
#' Xmat_test[,5] <- rnorm(num_test)
#' Xmat_test[,6] <- rbinom(num_test, size = 1, prob = 0.5)
#' Xmat_test[,7] <- rf(num_test,20,200)
#' Xmat_test[,8] <- runif(num_test, min = 0, max = 2)
#'
#' #it would be better to test performance of the models when there is correlation in the error terms.
#' varepsilon1_test <- rnorm(num_test, mean = 0, sd = sqrt(0.00025))
#' varepsilon2_test <- rnorm(num_test, mean = 0, sd = sqrt(0.00025))
#'
#' y1star_test <- 1 - 0.75*Xmat_test[,1] + 0.75*Xmat_test[,2] -
#'   0.5*Xmat_test[,4] -  0.5*Xmat_test[,6] - 0.25*Xmat_test[,1]^2 -
#'   0.75*Xmat_test[,1]*Xmat_test[,4] - 0.25*Xmat_test[,1]*Xmat_test[,2] -
#'   1*Xmat_test[,1]*Xmat_test[,6] + 0.5*Xmat_test[,2]*Xmat_test[,6] +
#'   varepsilon1_test
#'
#' y2star_test <- 1 + 0.25*Xmat_test[,4] - 0.75*Xmat_test[,6] +
#'   0.5*Xmat_test[,7] + 0.25*Xmat_test[,8] +
#'   0.25*Xmat_test[,4]^2 + 0.75*Xmat_test[,7]^2 + 0.5*Xmat_test[,8]^2 -
#'   1*Xmat_test[,4]*Xmat_test[,6] + 0.5*Xmat_test[,4]*Xmat_test[,8] +
#'   1*Xmat_test[,6]*Xmat_test[,7] - 0.25*Xmat_test[,7]*Xmat_test[,8] +
#'   varepsilon2_test
#'
#' y2obs_test <- ifelse(y1star_test>0, y2star_test,0)
#'
#' y2response_test <- ifelse(y1star_test>0, 1,0)
#'
#' tbartII_example <- tbart2c(Xmat_train,
#'                            Xmat_test,
#'                            Xmat_train,
#'                            Xmat_test,
#'                            y2obs_train,
#'                            n.iter=5000,
#'                            n.burnin=1000,
#'                            censored_value = 0,
#'                            eq_by_eq = TRUE)
#'
#'
#' pred_probs_tbart2_test <- rowMeans(tbartII_example$test.probcens)
#'
#' #Training (within-sample) Prediction Realization Table
#'
#' cutoff_point <- mean(y2obs_train>0)
#'
#' test_bin_preds <- ifelse(1 - pred_probs_tbart2_test > cutoff_point,1,0)
#'
#' #Training (within-sample) Prediction Realization Table
#'
#' pred_realization_test <- rbind(cbind(table(y2response_test, test_bin_preds)/length(y2response_test),
#'                                      apply(table(y2response_test, test_bin_preds)/length(y2response_test),1,sum)),
#'                                c(t(apply(table(y2response_test, test_bin_preds)/length(y2response_test),2,sum)), 1))
#'
#' hit_rate_test <- pred_realization_test[1,1] +pred_realization_test[2,2]
#'
#' testpreds_tbart2 <- rowMeans(tbartII_example$uncond_exp_test)
#'
#' sqrt(mean((y2obs_test - testpreds_tbart2  )^2 ))
#'
#' @export

softtbart2np <- function(x.train,
                         x.test,
                         w.train,
                         w.test,
                         y,
                         n.iter=1000,
                         n.burnin=100,
                         censored_value = NA,
                         gamma0 = 0,
                         G0=10,
                         nzero = 6,#0.002,
                         S0= 12,#0.002,
                         sigest = NA,
                         n.trees_outcome = 50L,
                         n.trees_censoring = 50L,
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
                         eq_by_eq = TRUE,
                         accelerate = FALSE,
                         cov_prior = "VH",
                         tau = 0.5,
                         M_mat = 2*diag(2),#matrix(c(1, 0,0, 1),nrow = 2, ncol = 2, byrow = TRUE),
                         alpha_prior = "vh",
                         c1 = 2,
                         c2 = 2,
                         alpha_gridsize = 100L,
                         selection_test = 1,
                         init.many.clust = TRUE,
                         nu0 = 3,
                         quantsig = 0.95){



  if(!(cov_prior %in% c("VH","Omori","Mixture", "Ding"))){
    stop("cov_prior must equal VH, Omori, Mixture, or Ding")
  }

  if(!(is.integer(alpha_gridsize))){
    stop("alpha_gridsize must be an integer")
  }

  # if(is.vector(x.train) | is.factor(x.train)| is.data.frame(x.train)) x.train = as.matrix(x.train)
  # if(is.vector(x.test) | is.factor(x.test)| is.data.frame(x.test)) x.test = as.matrix(x.test)

  # if((!is.matrix(x.train))) stop("argument x.train must be a double matrix")
  # if((!is.matrix(x.test)) ) stop("argument x.test must be a double matrix")

  if(nrow(x.train) != length(y)) stop("number of rows in x.train must equal length of y.train")
  if((ncol(x.test)!=ncol(x.train))) stop("input x.test must have the same number of columns as x.train")

  if((ncol(w.test)!=ncol(w.train))) stop("input w.test must have the same number of columns as w.train")

  if((nrow(w.test)!=nrow(x.test))) stop("input w.test must have the same number of rows as x.test")
  if((nrow(w.train)!=nrow(x.train))) stop("input w.train must have the same number of rows as x.train")


  #indexes of censored observations
  if(is.na(censored_value)){
    cens_boolvec <- is.na(y)
    uncens_boolvec <- !(is.na(y))

    cens_inds <- which(is.na(y))
    uncens_inds <- which(!(is.na(y)))
  }else{
    cens_boolvec <- (y == censored_value)
    uncens_boolvec <- (y != censored_value)

    cens_inds <- which(y == censored_value)
    uncens_inds <- which(y != censored_value)
  }

  if(length(cens_inds)==0) stop("No censored observations")


  # normalize the outcome
  tempmean <- mean(y[uncens_inds])
  tempsd <- sd(y[uncens_inds])
  originaly <- y
  y <- (y - tempmean)/tempsd

  if(is.numeric(censored_value)){
    censored_value <- (censored_value - tempmean)/tempsd
  }


  x.train <- as.matrix(x.train)
  x.test <- as.matrix(x.test)
  w.train <- as.matrix(w.train)
  w.test <- as.matrix(w.test)


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

  ecdfs   <- list()
  for(i in 1:ncol(w.train)) {
    ecdfs[[i]] <- ecdf(w.train[,i])
    if(length(unique(w.train[,i])) == 1) ecdfs[[i]] <- identity
    if(length(unique(w.train[,i])) == 2) ecdfs[[i]] <- make_01_norm(w.train[,i])
  }
  for(i in 1:ncol(w.train)) {
    w.train[,i] <- ecdfs[[i]](w.train[,i])
    w.test[,i] <- ecdfs[[i]](w.test[,i])
  }

  rm(ecdfs)


  #create z vector


  #create ystar vector
  ystar <- rep(mean(y[uncens_inds]), length(y))
  ystar[uncens_inds] <- y[uncens_inds]


  n <- length(y)
  n0 <- length(cens_inds)
  n1 <- length(uncens_inds)

  ntest = nrow(x.test)

  offsetz <- 0 #qnorm(n1/n)

  z <- rep(offsetz, length(y))

  # z[cens_inds] <- qnorm(0.001) #rtruncnorm(n0, a= -Inf, b = 0, mean = offsetz, sd = 1)
  #
  # z[uncens_inds] <- qnorm(0.999) #rtruncnorm(n1, a= 0, b = Inf, mean = offsetz, sd = 1)

  z[cens_inds] <- rtruncnorm(n0, a= -Inf, b = 0, mean = offsetz, sd = 1)
  z[uncens_inds] <- rtruncnorm(n1, a= 0, b = Inf, mean = offsetz, sd = 1)

  # z <- rnorm(n = length(y), mean = offsetz, sd =1)


  # meanmu_z <- (min(z - offsetz) +max(z- offsetz))/(2*n.trees_censoring)
  # sigmu_z <- (max(z- offsetz) - min(z- offsetz))/(2*2*sqrt(n.trees_censoring))
  #set prior parameter values


  # if(is.null(S0)){
  #
  #   #use uncensored observations
  #   if(is.na(sigest)) {
  #     if(ncol(x.train) < n1) {
  #       df = data.frame(x = x.train[uncens_inds,],y = y[uncens_inds])
  #       lmf = lm(y~.,df)
  #       sigest = summary(lmf)$sigma
  #     } else {
  #       sigest = sd(y[uncens_inds])
  #     }
  #   }
  #
  #   S0 <- 2*(sigest^2 -   (1/(8*(G0^2))) - 4*(gamma0^2)*G0   )
  #
  #   # S0 <- 2*(sigest^2)# -   (1/(8*(G0^2))) - 4*(gamma0^2)*G0   )
  #
  #   # S0 <- 0.002
  #
  # }

  # #use uncensored observations
  # if(is.na(sigest)) {
  #   if(ncol(x.train) < n1) {
  #     df = data.frame(x = x.train[uncens_inds,],y = y[uncens_inds])
  #     lmf = lm(y~.,df)
  #     sigest = summary(lmf)$sigma
  #   } else {
  #     sigest = sd(y[uncens_inds])
  #   }
  # }
  #
  # if(is.null(nzero)){
  #
  #   nzero <- 2*(sigest^2)
  #
  # }

  #use uncensored observations
  if(is.na(sigest)) {
    if(ncol(x.train) < n1) {
      # df = data.frame(x = x.train[uncens_inds,],y = y[uncens_inds])
      # lmf = lm(y~.,df)
      # sigest = summary(lmf)$sigma
      if(is.na(censored_value)){
        dtemp <-  1*(!(is.na(y)))

      }else{
        dtemp <-  1*(y != censored_value)

      }

      df = data.frame(x = cbind(x.train,w.train)  , y = y, d = dtemp )

      # seleq <- paste0("d ~ " , paste(paste("x",(ncol(x.train)+1):(ncol(x.train) + ncol(w.train)),sep = "."),collapse = " + "))
      # outeq <- paste0("y ~ " , paste(paste("x",1:ncol(x.train),sep = "."),collapse = " + "))

      # colnames(df)[1:ncol(x.train)] <- paste("x",1:ncol(x.train),sep = ".")
      # colnames(df)[(ncol(x.train)+1):(ncol(x.train) + ncol(w.train))] <- paste("x",(ncol(x.train)+1):(ncol(x.train) + ncol(w.train)),sep = ".")
      #
      # seleq <- paste0("d ~ " , paste(paste("x",(ncol(x.train)+1):(ncol(x.train) + ncol(w.train)),sep = "."),collapse = " + "))
      # outeq <- paste0("y ~ " , paste(paste("x",1:ncol(x.train),sep = "."),collapse = " + "))

      seleq <- paste0("d ~ " , paste(colnames(df)[(ncol(x.train)+1):(ncol(x.train) + ncol(w.train))],
                                     collapse = " + "))
      outeq <- paste0("y ~ " , paste(colnames(df)[1:ncol(x.train)],
                                     collapse = " + "))

      heckit.ml <- heckit(selection = as.formula(seleq),
                          outcome = as.formula(outeq),
                          data = df,
                          method = "ml")


      # heckit.ml$estimate["rho"]

      correst <- heckit.ml$estimate["rho"]
      sigest <- heckit.ml$estimate["sigma"]

      # correst <- heckit.2step$coefficients["rho"]
      # sigest <- heckit.2step$coefficients["sigma"]

      # print("heckit.2step$coefficients = ")
      # print(heckit.2step$coefficients)
      # print("heckit.2step$lm$coefficients = ")
      # print(heckit.2step$lm$coefficients)
      # print("correst = ")
      # print(correst)
      # print("correst = ")
      # print(correst)
      # print("correst = ")
      # print(correst)
    } else {
      sigest = sd(y[uncens_inds])
      correst <- 0
    }
  }


  # set the prior based on sigest

  #
  # M_mat = 10*matrix(c(1, 0,0, 1),nrow = 2, ncol = 2, byrow = TRUE)

  # M_mat = ((sigest^2))*matrix(c(1, 0,0, 1),nrow = 2, ncol = 2, byrow = TRUE)

  M_inv <- solve(M_mat)

  S0 <- (sigest^2)*(1 - correst^2)*(nzero-2)/(1+tau)

  #alternatively, draw this from the prior
  Sigma_mat <- cbind(c(1,0),c(0,sigest^2))

  #set initial gamma
  gamma1 <- correst*sigest  #0#cov(ystar,z)

  #set initial phi
  phi1 <- sigest^2 - gamma1^2

  # print("phi1 = ")
  # print(phi1)
  # print("correst = ")
  # print(correst)
  # print("sigest = ")
  # print(sigest)
  # print("gamma1 = ")
  # print(gamma1)

  gamma0 <- correst*sigest

  if(cov_prior == "Ding"){
    gamma0 <- 0

    # sigquant <- 0.9
    qchi <- qchisq(1.0-quantsig,nu0-1)
    cdivnu <- (sigest*sigest*qchi)/(nu0-1) #lambda parameter for sigma prior
    cding <- cdivnu*(nu0-1)

    rhoinit <- 0
    siginit <- sigest
  }

  #set initial sigma

  #alternatively, draw this from the prior
  Sigma_mat <- cbind(c(1,0),c(0,sigest^2))

  #set initial gamma
  #gamma1 <- 0 # cov(ystar,z)

  # gamma1_vec_train <- rep(0,n)
  # gamma1_vec_test <- rep(0,ntest)
  gamma1_vec_train <- rep(gamma0,n)
  gamma1_vec_test <- rep(gamma0,ntest)

  #set initial phi
  # phi1 <- sigest^2

  # phi1_vec_train <- rep(sigest^2, n)
  # phi1_vec_test <- rep(sigest^2, ntest)
  phi1_vec_train <- rep(phi1, n)
  phi1_vec_test <- rep(phi1, ntest)

  mu1_vec_train <- rep(0,n)
  mu1_vec_test <- rep(0,ntest)

  mu2_vec_train <- rep(0,n)
  mu2_vec_test <- rep(0,ntest)

  varthetamat <- cbind(mu1_vec_train, mu2_vec_train, phi1_vec_train, gamma1_vec_train)
  varthetamat_test <- cbind(mu1_vec_test, mu2_vec_test, phi1_vec_test, gamma1_vec_test)

  #########  Initialize mixture vartheta values ######################################################

  varthetamat <- cbind(mu1_vec_train, mu2_vec_train, phi1_vec_train, gamma1_vec_train)

  # essentially assuming predictions all equal to zero initially

  u_z <- z - offsetz #- mutemp_z

  u_y <- ystar[uncens_inds] #- mutemp_y[uncens_inds]

  if(init.many.clust == TRUE){
    uncens_count <- 0

    for(i in 1:n){

      if(i %in% cens_inds){
        #censored observation

        if(cov_prior == "Ding"){
          tempsigma <- rinvwishart(nu = nu0,
                                   S = cding*diag(2))

          transmat <- cbind(c(1,0),c(0,1/sqrt(tempsigma[2,2])))
          tempomega <- (transmat %*% tempsigma) %*% transmat

          temprho <- tempomega[1,2]/(sqrt(tempomega[1,1]))

          # gamma1 <- tempomega[1,2]
          # phi1 <- tempomega[1,1] - (gamma1^2)

          gamma1_vec_train[i] <- tempomega[1,2]
          phi1_vec_train[i] <- tempomega[1,1] - (gamma1_vec_train[i]^2)

        }else{

          phi1_vec_train[i] <- 1/rgamma(n = 1, shape =  nzero/2, rate = S0/2)

          phitilde <- phi1_vec_train[i]

          gamma1_vec_train[i] <- rnorm(1,
                                       mean = gamma0,
                                       sd = sqrt(tau*phitilde))
        }


        mutilde <- c(NA,NA)

        mu1_vec_train[i] <- rnorm(n = 1,
                                  mean = u_z[i]/( (1/M_mat[1,1]) + 1  ),
                                  sd = sqrt(1/( (1/M_mat[1,1]) + 1  )) )

        mutilde[1] <- mu1_vec_train[i]

        mu2_vec_train[i] <- rnorm(n = 1,
                                  mean =   mutilde[1]*M_mat[1,2]/M_mat[1,1],
                                  sd =  sqrt( M_mat[2,2] - (M_mat[1,2]^2/M_mat[1,1])   ))

        mutilde[2] <- mu2_vec_train[i]


      }else{
        # uncensored observation, must use Gibbs sampler to obtain initial value
        # Maybe it would be more efficient to just sample from the prior?

        #increase count of uncensored observation index
        uncens_count <- uncens_count + 1

        num_sample <- 1

        # initialize (could even use draws from prior to initialize this)
        mutilde <- c(mu1_vec_train[i],mu2_vec_train[i])

        phitilde <- phi1_vec_train[i]
        gammatilde <- gamma1_vec_train[i]

        for(samp_ind in 1:num_sample){
          # print("samp_ind = ")
          # print(samp_ind)

          Sigma_mat_temp_j <- cbind(c(1,
                                      gammatilde),
                                    c(gammatilde,
                                      phitilde+gammatilde^2))

          # print("Sigma_mat_temp_j = ")
          # print(Sigma_mat_temp_j)

          tempsigmainv <- solve(Sigma_mat_temp_j)
          tempsigmamat <- solve(M_inv + tempsigmainv )

          mutilde <- Rfast::rmvnorm(n = 1,
                                    mu = (tempsigmamat%*%tempsigmainv) %*%
                                      c((u_z[i]),(u_y[uncens_count])),
                                    sigma = tempsigmamat )

          if(cov_prior == "Ding"){

            rhotilde <- gammatilde/sqrt(phitilde + gammatilde^2)


            # print( "nu0 = " )
            # print(nu0)
            #
            # print( "cding = " )
            # print(cding)
            #
            # print( "rhotilde = " )
            # print(rhotilde)


            sigz2 <- 1/rgamma(n = 1,
                              shape = nu0/2,
                              rate = cding/(2*(1- (rhotilde^2))))


            u_z2 <- sqrt(sigz2)*(u_z[i] - mutilde[1])
            u_y2 <- u_y[uncens_count] - mutilde[2]#- mutemp_y[uncens_inds]

            zsquares <- crossprod(u_z2, u_z2)[1]
            ysquares <- crossprod(u_y2, u_y2)[1]
            zycross <- crossprod(u_z2, u_y2)[1]

            Stemp <- cbind(c(ysquares,zycross),
                           c(zycross, zsquares))

            tempsigma <- rinvwishart(nu = 1 + nu0,
                                     S = Stemp+cding*diag(2))

            transmat <- cbind(c(1,0),c(0,1/sqrt(tempsigma[2,2])))
            tempomega <- (transmat %*% tempsigma) %*% transmat

            temprho <- tempomega[1,2]/(sqrt(tempomega[1,1]))

            gammatilde <- tempomega[1,2]
            phitilde <- tempomega[1,1] - (gammatilde^2)

          }else{ # if VH prior, use VH full conditional

            dbar_temp <- (S0/2) + (gammatilde^2/tau) +
              (1/2)*( (u_y[uncens_count] - mutilde[2] - gammatilde*(u_z[i] - mutilde[1]) )^2 )

            phitilde <- 1/rgamma(n = 1, shape =  (nzero/2) +  1, rate = dbar_temp)

            gammabar_temp <- ( (u_z[i] - mutilde[1])*(u_y[uncens_count] - mutilde[2]) ) /
              ((1/tau) + ( (u_z[i] - mutilde[1])^2 ) )


            gammatilde <- rnorm(1,
                                mean = gammabar_temp,
                                sd = sqrt( phitilde / ((1/tau) + ( (u_z[i] - mutilde[1])^2 ) )   ))

          }

        } # end of Gibbs sampler loop

        mu1_vec_train[i] <- mutilde[1]
        mu2_vec_train[i] <- mutilde[2]

        phi1_vec_train[i] <- phitilde
        gamma1_vec_train[i] <- gammatilde

      } # end else statement for uncensrored observations

    } # end loop over i
  }else{ #just one cluster for initial values
    # for both selected and unselected observations?

    # phi1_vec_train <- phi1 #sigest #  1/rgamma(n = 1, shape =  nzero/2, rate = S0/2)
    phi1_vec_train <- rep(phi1, n)

    phitilde <- phi1_vec_train[1]

    # gamma1_vec_train <- gamma0 # rnorm(1, mean = gamma0, sd = sqrt(tau*phitilde))
    gamma1_vec_train <- rep(gamma0,n)

    # mutilde <- c(NA,NA)
    #
    # mu1_vec_train[i] <- rnorm(n = 1,
    #                           mean = u_z[i]/( (1/M_mat[1,1]) + 1  ),
    #                           sd = sqrt(1/( (1/M_mat[1,1]) + 1  )) )
    #
    # mutilde[1] <- mu1_vec_train[i]
    # mu2_vec_train[i] <- rnorm(n = 1,
    #                           mean =   mutilde[1]*M_mat[1,2]/M_mat[1,1],
    #                           sd =  sqrt( M_mat[2,2] - (M_mat[1,2]^2/M_mat[1,1])   ))
    #
    # mutilde[2] <- mu2_vec_train[i]

    mutilde <- c(0,0)

    # mu1_vec_train[i] <- 0
    #
    # mu2_vec_train[i] <- 0

    mu1_vec_train <- rep(0,n)
    mu2_vec_train <- rep(0,n)



    # mutilde[2] <- 0

  }


  varthetamat <- cbind(mu1_vec_train, mu2_vec_train, phi1_vec_train, gamma1_vec_train)



  #intiailize alpha

  #can take a draw form the prior, or set equal to mode

  alpha <- rgamma(n = 1, shape = c1, rate = c2)

  alpha <- c1/c2


  draw = list(
    # Z.mat_train = array(NA, dim = c(n, n.iter)),
    # Z.mat_test = array(NA, dim = c(ntest, n.iter)),
    # Y.mat_train = array(NA, dim = c(n, n.iter)),
    # Y.mat_test = array(NA, dim = c(ntest, n.iter)),
    mu_y_train = array(NA, dim = c(n1, n.iter)),# array(NA, dim = c(n, n.iter)),
    mu_y_test = array(NA, dim = c(ntest, n.iter)),
    mu_y_train_noerror = array(NA, dim = c(n1, n.iter)),# array(NA, dim = c(n, n.iter)),
    mu_y_test_noerror = array(NA, dim = c(ntest, n.iter)),
    # mucens_y_train = array(NA, dim = c(n0, n.iter)),
    muuncens_y_train = array(NA, dim = c(n1, n.iter)),
    mu_z_train = array(NA, dim = c(n, n.iter)),
    mu_z_test = array(NA, dim = c(ntest, n.iter)),
    train.probcens =  array(NA, dim = c(n1, n.iter)),#array(NA, dim = c(n, n.iter)),#,
    test.probcens =  array(NA, dim = c(ntest, n.iter)),#,
    cond_exp_train = array(NA, dim = c(n1, n.iter)),#cond_exp_train = array(NA, dim = c(n, n.iter)),
    cond_exp_test = array(NA, dim = c(ntest, n.iter)),
    ystar_train = array(NA, dim = c(n, n.iter)),
    ystar_test = array(NA, dim = c(ntest, n.iter)),
    zstar_train = array(NA, dim = c(n, n.iter)),
    zstar_test = array(NA, dim = c(ntest, n.iter)),
    ycond_draws_train = list(),
    ycond_draws_test = list(),
    vartheta_draws = array(NA, dim = c(n, 4, n.iter)),
    vartheta_test_draws = array(NA, dim = c(ntest, 4, n.iter)),
    alpha = rep(NA,n.iter)
    # Sigma_draws = array(NA, dim = c(2, 2, n.iter))
  )

  if(is.numeric(censored_value)){
    draw$uncond_exp_train <- array(NA, dim = c(n1, n.iter)) #array(NA, dim = c(n, n.iter))
    draw$uncond_exp_test <- array(NA, dim = c(ntest, n.iter))
    # draw$ydraws_train <- array(NA, dim = c(n, n.iter))
    draw$ydraws_test <- array(NA, dim = c(ntest, n.iter))
  }


  if(selection_test == 1){

    #error draws
    # save in array
    #first slice is selection equation error
    #second slice is outcome equation error
    draw$error_draws <- array(NA, dim = c(n, n.iter,2))
    draw$error_draws_test <- array(NA, dim = c(n.iter,2))

    # draw correlations and other dependence measures
    # just one value per iteration, so it is a vector
    draw$pearsoncorr_draws <- array(NA, dim = c(n.iter)) #array(NA, dim = c(n, n.iter))
    draw$kendalltau_draws <- array(NA, dim = c(n.iter)) #array(NA, dim = c(n, n.iter))
    draw$spearmanrho_draws <- array(NA, dim = c(n.iter)) #array(NA, dim = c(n, n.iter))


  }


  ########## Initialize SoftBart #####################

  hypers_y <- Hypers(x.train[uncens_inds,], ystar[uncens_inds],
                     num_tree = n.trees_outcome, #sigma_hat = 1,
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


  opts_y <- Opts(update_sigma = TRUE, num_print = n.burnin + n.iter + 1)

  sampler_forest_y <- MakeForest(hypers_y, opts_y, warn = FALSE)


  hypers_z <- Hypers(w.train, z - offsetz,
                     num_tree = n.trees_censoring, #sigma_hat = 1,
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


  opts_z <- Opts(update_sigma = TRUE, num_print = n.burnin + n.iter + 1)

  sampler_forest_z <- MakeForest(hypers_z, opts_z, warn = FALSE)



  # ########## Initialize dbarts #####################
  #
  # control_z <- dbartsControl(updateState = updateState, verbose = FALSE,  keepTrainingFits = TRUE,
  #                            keepTrees = TRUE,
  #                            n.trees = n.trees_censoring,
  #                            n.burn = n.burn,
  #                            n.samples = n.samples,
  #                            n.thin = n.thin,
  #                            n.chains = n.chains,
  #                            n.threads = n.threads,
  #                            printEvery = printEvery,
  #                            printCutoffs = printCutoffs,
  #                            rngKind = rngKind,
  #                            rngNormalKind = rngNormalKind,
  #                            rngSeed = rngSeed)
  #
  # control_y <- dbartsControl(updateState = updateState, verbose = FALSE,  keepTrainingFits = TRUE,
  #                            keepTrees = TRUE,
  #                            n.trees = n.trees_outcome,
  #                            n.burn = n.burn,
  #                            n.samples = n.samples,
  #                            n.thin = n.thin,
  #                            n.chains = n.chains,
  #                            n.threads = n.threads,
  #                            printEvery = printEvery,
  #                            printCutoffs = printCutoffs,
  #                            rngKind = rngKind,
  #                            rngNormalKind = rngNormalKind,
  #                            rngSeed = rngSeed)
  # print(colnames(Xmat.train))

  # print("begin dbarts")


  # not clear how to set this in initial step
  # maybe should actually sample vartheta first?

  weightstemp <- rep(1,n)

  weightstemp[uncens_inds] <- (gamma1_vec_train[uncens_inds]^2 + phi1_vec_train[uncens_inds])/phi1_vec_train[uncens_inds]

  weightstemp_y  <- 1/phi1_vec_train[uncens_inds]


  if(nrow(x.test )==0){


    xdf_y <- data.frame(y = ystar[uncens_inds], x = x.train[uncens_inds,])

    # sampler_y <- dbarts(y ~ .,
    #                     data = xdf_y,
    #                     #test = x.test,
    #                     weights = weightstemp_y,
    #                     control = control_y,
    #                     tree.prior = tree.prior,
    #                     node.prior = node.prior,
    #                     resid.prior = resid.prior,
    #                     proposal.probs = proposal.probs,
    #                     sigma = sigmadbarts
    #                     )

    xdf_z <- data.frame(y = z - offsetz, x = w.train)

    # sampler_z <- dbarts(y ~ .,
    #                     data = xdf_z,
    #                     #test = x.test,
    #                     weights = weightstemp,
    #                     control = control_z,
    #                     tree.prior = tree.prior,
    #                     node.prior = node.prior,
    #                     resid.prior = resid.prior,
    #                     proposal.probs = proposal.probs,
    #                     sigma = 1#sigmadbarts
    #                     )

  }else{

    xdf_y <- data.frame(y = ystar[uncens_inds], x = x.train[uncens_inds,])
    xdf_y_test <- data.frame(x = x.test)

    # sampler_y <- dbarts(y ~ .,
    #                     data = xdf_y,
    #                     test = xdf_y_test,
    #                     # weights = weightstemp_y,
    #                     control = control_y,
    #                     tree.prior = tree.prior,
    #                     node.prior = node.prior,
    #                     resid.prior = resid.prior,
    #                     proposal.probs = proposal.probs,
    #                     sigma = sigmadbarts
    #                     )


    # print("length(z - offsetz) = ")
    # print(length(z - offsetz))
    #
    # print("length(weightstemp) = ")
    # print(length(weightstemp))

    xdf_z <- data.frame(y = z - offsetz, x = w.train)
    xdf_z_test <- data.frame(x = w.test)

    # sampler_z <- dbarts(y ~ .,
    #                     data = xdf_z,
    #                     test = xdf_z_test,
    #                     # weights = weightstemp,
    #                     control = control_z,
    #                     tree.prior = tree.prior,
    #                     node.prior = node.prior,
    #                     resid.prior = resid.prior,
    #                     proposal.probs = proposal.probs,
    #                     sigma = 1#sigmadbarts
    #                     )

  }

  preds.train_ystar <- matrix(NA, n, 1)
  preds.train_z <- matrix(NA, n, 1)

  preds.test_ystar <- matrix(NA, ntest, 1)
  preds.test_z <- matrix(NA, ntest, 1)


  #initialize sum-of-tree sampler


  z_resids <- z - offsetz - mu1_vec_train #z_epsilon
  z_resids[uncens_inds] <- z[uncens_inds] - offsetz - mu1_vec_train[uncens_inds] -
    (ystar[uncens_inds] - mu2_vec_train[uncens_inds] - 0)*gamma1_vec_train[uncens_inds]/(phi1_vec_train[uncens_inds] + gamma1_vec_train[uncens_inds]^2)

  sampler_forest_z$set_sigma(1)

  mutemp_z <- sampler_forest_z$do_gibbs_weighted(w.train,
                                                 z_resids,
                                                 weightstemp,
                                                 w.train,
                                                 1)

  mutemp_test_z <- sampler_forest_z$do_predict(w.test)



  # #set the response for draws of z trees
  # sampler_z$setResponse(y = z_resids)
  # #set the standard deivation
  # sampler_z$setSigma(sigma = 1)
  #
  # # weightstemp[uncens_inds] <- (gamma1_vec_train[uncens_inds]^2 + phi1_vec_train[uncens_inds])/phi1_vec_train[uncens_inds]
  #
  # # print("weightstemp = ")
  # # print(weightstemp)
  #
  # # sampler_z$setSigma(sigma = 1)
  # sampler_z$setWeights(weights = weightstemp)
  #
  # # sampler_z$sampleTreesFromPrior()
  #
  # # priormean_z <- sampler_z$predict(xdf_z)[1,]
  #
  # # sampler_z$sampleNodeParametersFromPrior()
  #
  # samplestemp_z <- sampler_z$run()
  #
  # # mutemp_z <- rep(0,n) # samplestemp_z$train[,1]
  # # mutemp_test_z <- rep(0,ntest) #samplestemp_z$test[,1]
  #
  # mutemp_z <- samplestemp_z$train[,1]
  # mutemp_test_z <- samplestemp_z$test[,1]
  #
  # # mutemp_test_z <- sampler_z$predict(xdf_z_test)[,1]#samplestemp_z$test[,1]



  # print("length(mutemp_test_z) = ")
  # print(length(mutemp_test_z))
  #
  # print("mutemp_test_z[1000:1010] = ")
  # print(mutemp_test_z[1000:1010])
  #
  # print("mutemp_test_z[1:10] = ")
  # print(mutemp_test_z[1:10])
  #
  # print("nrow(xdf_z_test) = ")
  # print(nrow(xdf_z_test))

  # print("weightstemp_y = ")
  # print(weightstemp_y)



  y_resids <- ystar[uncens_inds] - mu2_vec_train[uncens_inds] -
    gamma1_vec_train[uncens_inds]*(z[uncens_inds] - offsetz - mu1_vec_train[uncens_inds]- mutemp_z[uncens_inds])
  # sd_ydraw <- sqrt(phi1_vec_train[uncens_inds])


  # print("y_resids = ")
  #
  # print(y_resids)

  sampler_forest_y$set_sigma(1)

  mutemp_y <- sampler_forest_y$do_gibbs_weighted(x.train[uncens_inds,],
                                                 y_resids,
                                                 weightstemp_y,
                                                 x.train[uncens_inds,],
                                                 1)

  mutemp_test_y <- sampler_forest_y$do_predict(x.test)

  # #set the response for draws of z trees
  # sampler_y$setResponse(y = y_resids)
  # sampler_y$setSigma(sigma = 1)
  #
  # # sampler_y$setSigma(sigma = sigest)
  # sampler_y$setWeights(weights = weightstemp_y)
  #
  # # sampler_y$sampleTreesFromPrior()
  #
  # # priormean_y <- sampler_y$predict(xdf_y)[1,]
  #
  # # sampler_y$sampleNodeParametersFromPrior()
  #
  # samplestemp_y <- sampler_y$run()
  #
  # # mutemp_y <- rep(mean(y),n) #samplestemp_y$train[,1]
  # # mutemp_test_y <- rep(mean(y),ntest) # samplestemp_y$test[,1]
  #
  # mutemp_y <- samplestemp_y$train[,1]
  # mutemp_test_y <- samplestemp_y$test[,1]

  # print("length(mutemp_test_y) = ")
  # print(length(mutemp_test_y))

  # if(sigest != samplestemp_y$sigma){
  #   print("sigest = ")
  #   print(sigest)
  #   print("dbarts sigma estimate =")
  #   print(samplestemp_y$sigma)
  #
  #   df = data.frame(x.train,y)
  #   lmf = lm(y~.,df)
  #   sigest2 = summary(lmf)$sigma
  #
  #   print("sigest2 = ")
  #   print(sigest2)
  #
  #   # stop("sigest != samplestemp_y$sigma")
  #
  # }

  # sigest <-  samplestemp_y$sigma
  # S0 <- 2*(sigest^2 -   (1/(8*(G0^2))) - 4*(gamma0^2)*G0   )




  #COMMENTING OUT INITIALIZATION FOR FIRST ITERATION
  #MIGHT NEED TO EDIT THIS

  # sampler$setResponse(y = z)
  # sampler$setSigma(sigma = 1)

  #sampler$setPredictor(x= Xmat.train$x, column = 1, forceUpdate = TRUE)

  #mu = as.vector( alpha + X.mat %*% beta )

  # sampler$sampleTreesFromPrior()
  # samplestemp <- sampler$run()

  #mutemp <- samplestemp$train[,1]
  #suppose there are a number of samples

  # print("sigma = ")
  # sigma <- samplestemp$sigma
  #
  # mu <- samplestemp$train[,1]
  # mutest <- samplestemp$test[,1]
  #
  # ystar <- rnorm(n,mean = mu, sd = sigma)
  # ystartest <- rnorm(ntest,mean = mutest, sd = sigma)
  #
  # ystartestcens <-rtruncnorm(ntest, a = below_cens, b = above_cens, mean = mutest, sd = sigma)
  #
  # probcensbelow <- pnorm(below_cens, mean = mutest, sd = sigma)
  # probcensabove <- 1 - pnorm(above_cens, mean = mutest, sd = sigma)



  #save the first round of values
  # if(n.burnin == 0){
  #   draw$Z.mat[,1] = z
  #   draw$Z.matcens[,1] = z[cens_inds]
  #   # draw$Z.matuncens[,1] = z[uncens_inds]
  #   draw$Z.matcensbelow[,1] = z[censbelow_inds]
  #   draw$Z.matcensabove[,1] = z[censabove_inds]
  #   draw$mu[,1] = mu
  #   draw$mucens[,1] = mu[cens_inds]
  #   draw$muuncens[,1] = mu[uncens_inds]
  #   draw$mucensbelow[,1] = mu[censbelow_inds]
  #   draw$mucensabove[,1] = mu[censabove_inds]
  #   draw$ystar[,1] = ystar
  #   draw$ystarcens[,1] = ystar[cens_inds]
  #   draw$ystaruncens[,1] = ystar[uncens_inds]
  #   draw$ystarcensbelow[,1] = ystar[censbelow_inds]
  #   draw$ystarcensabove[,1] = ystar[censabove_inds]
  #   draw$test.mu[,1] = mutest
  #   draw$test.y_nocensoring[,1] = ystartest
  #   draw$test.y_withcensoring[,1] = ystartestcens
  #   draw$test.probcensbelow[,1] = probcensbelow
  #   draw$test.probcensabove[,1] = probcensabove
  #   draw$sigma[1] <- sigma
  # }


  if(any(phi1_vec_train <= 0)){
    stop("Line 801. some phi1_vec_train values <= 0")
  }

  #########  Begin Gibbs sampler ######################################################


  #loop through the Gibbs sampler iterations
  for(iter in 1:(n.iter+n.burnin)){


    # if( (iter %% 100) == 0){
    #   print("iteration number = ")
    #
    #   print(iter)
    # }

    # if(eq_by_eq){
    #   sig_zdraw <- 1
    #   sig_ydraw <- phi1
    #
    # }else{
    #   sig_zdraw <- phi1/(gamma1^2+phi1)
    #   sig_ydraw <- phi1
    #
    # }


    ###### sample Z #################


    temp_sd_z <- sqrt( phi1_vec_train/(phi1_vec_train+gamma1_vec_train^2)   )

    if(any(phi1_vec_train <= 0)){

      print("phi1_vec_train[phi1_vec_train <= 0] = ")
      print(phi1_vec_train[phi1_vec_train <= 0])
      stop("any(phi1_vec_train <= 0)")
    }else{
      # print("phi1_vec_train = ")
      # print(phi1_vec_train)
      #
      # print("gamma1_vec_train = ")
      # print(gamma1_vec_train)
    }

    if(any(is.nan(temp_sd_z))){

      print("temp_sd_z = ")
      print(temp_sd_z)
      stop("temp_sd_z contains NaNs")
    }



    #draw the latent outcome
    # z[cens_inds] <- rtruncnorm(n0, a= below_cens, b = above_cens, mean = mu[cens_inds], sd = sigma)
    if(length(cens_inds)>0){
      # temp_sd_y <- sqrt(phi1 + gamma1^2)

      # print("mutemp_y[cens_inds] = ")
      # print(mutemp_y[cens_inds])
      # print("temp_sd_y = ")
      # print(temp_sd_y)
      #
      # print("cens_inds = ")
      # print(cens_inds)

      # ystar[cens_inds] <- rnorm(n0,  mean = mutemp_y[cens_inds], sd = temp_sd_y)

      # temp_zmean_cens <- offsetz + mutemp_z[cens_inds] + (ystar[cens_inds]  - mutemp_y[cens_inds])*gamma1/(phi1 + gamma1^2)
      temp_zmean_cens <- offsetz + mutemp_z[cens_inds] + mu1_vec_train[cens_inds] #+ (ystar[cens_inds]  - mutemp_y[cens_inds])*gamma1/(phi1 + gamma1^2)


      z[cens_inds] <- rtruncnorm(n0, a= -Inf, b = 0, mean = temp_zmean_cens, sd = 1)
    }

    # temp_zmean_uncens <- offsetz + mutemp_z[uncens_inds] + (ystar[uncens_inds]  - mutemp_y[uncens_inds])*gamma1/(phi1 + gamma1^2)
    temp_zmean_uncens <- offsetz + mutemp_z[uncens_inds] +
      mu1_vec_train[uncens_inds] +
      (ystar[uncens_inds]  - mutemp_y - mu2_vec_train[uncens_inds])*
      gamma1_vec_train[uncens_inds]/(phi1_vec_train[uncens_inds] + gamma1_vec_train[uncens_inds]^2)

    z[uncens_inds] <- rtruncnorm(n1, a= 0, b = Inf, mean = temp_zmean_uncens,
                                 sd = temp_sd_z[uncens_inds])



    # z_epsilon <- z - offsetz - mutemp_z
    # y_epsilon <- ystar - mutemp_y

    z_epsilon <- z - offsetz - mutemp_z - mu1_vec_train

    y_epsilon <- rep(0, n)
    y_epsilon[uncens_inds] <- ystar[uncens_inds] - mutemp_y - mu2_vec_train[uncens_inds]

    # print("temp_sd_z = ")
    # print(temp_sd_z)
    #
    # print("mutemp_z = ")
    # print(mutemp_z)
    #
    # print("temp_zmean_cens = ")
    # print(temp_zmean_cens)
    #
    # print("z = ")
    # print(z)


    ####### draw sums of trees for z #######################################################

    #create residuals for z and set variance

    # if(eq_by_eq){
    #   z_resids <- z - offsetz #z_epsilon
    #   sd_zdraw <- 1
    # }else{
    #   #not sure about this step for tobit2b
    #   z_resids <- z - offsetz - y_epsilon*(gamma1/(phi1+gamma1^2))
    #   sd_zdraw <- sqrt(phi1 / (phi1 + gamma1^2)  )
    # }

    z_resids <- z - offsetz - mu1_vec_train #z_epsilon
    z_resids[uncens_inds] <- z[uncens_inds] - offsetz - mu1_vec_train[uncens_inds] -
      (ystar[uncens_inds] - mu2_vec_train[uncens_inds] - mutemp_y)*gamma1_vec_train[uncens_inds]/(phi1_vec_train[uncens_inds] + gamma1_vec_train[uncens_inds]^2)

    # print("z_resids = ")
    # print(z_resids)

    # print("length(z_resids) =")
    # print(length(z_resids))
    #
    # print("length(mu1_vec_train) =")
    # print(length(mu1_vec_train))
    #
    # print("length(z) =")
    # print(length(z))

    # #set the response for draws of z trees
    # sampler_z$setResponse(y = z_resids)
    # #set the standard deivation
    # sampler_z$setSigma(sigma = 1)

    weightstemp[uncens_inds] <- (gamma1_vec_train[uncens_inds]^2 + phi1_vec_train[uncens_inds])/phi1_vec_train[uncens_inds]


    if(any(is.nan(weightstemp))){

      print("weightstemp = ")
      print(weightstemp)
      stop("weightstemp contains NaNs")
    }

    # print("length(weightstemp) = ")
    # print(length(weightstemp))

    sampler_forest_z$set_sigma(1)

    mutemp_z <- sampler_forest_z$do_gibbs_weighted(w.train,
                                                   z_resids,
                                                   weightstemp,
                                                   w.train,
                                                   1)

    mutemp_test_z <- sampler_forest_z$do_predict(w.test)

    # sampler_z$setWeights(weights = weightstemp)
    #
    # samplestemp_z <- sampler_z$run()
    #
    # mutemp_z <- samplestemp_z$train[,1]
    # mutemp_test_z <- sampler_z$predict(xdf_z_test)[,1]#samplestemp_z$test[,1]

    # print("length(mutemp_test_z) = ")
    # print(length(mutemp_test_z))
    #
    # print("nrow(xdf_z_test) = ")
    # print(nrow(xdf_z_test))

    #update z_epsilon
    z_epsilon <- z - offsetz - mutemp_z - mu1_vec_train


    ####### draw sums of trees for y #######################################################

    #create residuals for z and set variance

    # print("y_epsilon = ")
    #
    # print(y_epsilon)
    #
    #
    # print("z_epsilon = ")
    #
    # print(z_epsilon)
    #
    # print("gamma1 = ")
    #
    # print(gamma1)


    # if(eq_by_eq){
    #   y_resids <- ystar[uncens_inds] - gamma1*z_epsilon[uncens_inds]
    #   sd_ydraw <- sqrt(phi1)
    # }else{
    #   y_resids <- ystar[uncens_inds] - gamma1*z_epsilon[uncens_inds]
    #   sd_ydraw <- sqrt(phi1)
    # }

    y_resids <- ystar[uncens_inds] - mu2_vec_train[uncens_inds] - gamma1_vec_train[uncens_inds]*(z[uncens_inds] - offsetz - mu1_vec_train[uncens_inds]- mutemp_z[uncens_inds])
    # sd_ydraw <- sqrt(phi1_vec_train[uncens_inds])


    # print("y_resids = ")
    #
    # print(y_resids)


    sampler_forest_y$set_sigma(1)

    mutemp_y <- sampler_forest_y$do_gibbs_weighted(x.train[uncens_inds,],
                                                   y_resids,
                                                   weightstemp_y,
                                                   x.train[uncens_inds,],
                                                   1)

    mutemp_test_y <- sampler_forest_y$do_predict(x.test)


    # #set the response for draws of z trees
    # sampler_y$setResponse(y = y_resids)
    # #set the standard deviation
    # # sampler_y$setSigma(sigma = sd_ydraw)
    #
    # sampler_y$setSigma(sigma = 1)
    #
    # weightstemp_y  <- 1/phi1_vec_train[uncens_inds]
    # sampler_y$setWeights(weights = weightstemp_y)
    #
    # samplestemp_y <- sampler_y$run()
    #
    # mutemp_y <- samplestemp_y$train[,1]
    # mutemp_test_y <- samplestemp_y$test[,1]


    #update z_epsilon
    y_epsilon[uncens_inds] <- ystar[uncens_inds] - mutemp_y - mu2_vec_train[uncens_inds]



    #########  Step 5: Sample alpha  ######################################################

    # This step can be implemented using the prior of van Hasselt (2011)
    # or the prior of George et al. (2019), McCulloch et al. (2021) ans Conley et al. (2008)

    # count number of unique mixture components
    # there is probably a more efficient way of doing this

    # create a matrix in which each row contains all of an individual's mixture component parameters

    varthetamat <- cbind(mu1_vec_train, mu2_vec_train, phi1_vec_train, gamma1_vec_train)
    varthetamat_test <- cbind(mu1_vec_test, mu2_vec_test, phi1_vec_test, gamma1_vec_test)

    #obtain the unique rows (components)
    vartheta_unique_mat <- unique(varthetamat)

    #number of unique components
    k_uniq <- nrow(vartheta_unique_mat)


    if(alpha_prior == "vh"){

      #########  VH Step 5 a: Sample auxiliary variable kappa  ######################################################

      kappa_aux = rbeta(n = 1, shape1 = alpha+1, shape2 = n)

      #########  VH Step 5 b: Sample alpha from a mixture of gamma distributions  ######################################################

      #obtain the mixing probability
      p_kappa <- (c1+k_uniq-1)/(n*(c2-log(kappa_aux))+c1+k_uniq-1)

      #sample a mixture component
      mix_draw <- rbinom(1,1,p_kappa)

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


        if(floor(0.1*n)<=1){
          stop("Not enough observations for Prior of George et al. (2019)")
        }

        psiprior <- 0.5

        # from ivbart package code
        # https://github.com/rsparapa/bnptools/blob/master/ivbart/R/amode.R
        egamm = 0.5772156649
        tempmin= digamma(Imin) - log(egamm+log(n))
        tempmax= digamma(Imax) - log(egamm+log(n))

        alpha_min <- exp(tempmin)
        alpha_max <- exp(tempmax)
        alpha_values =  seq(from=alpha_min,to=alpha_max,length.out=alpha_gridsize)
        temp_aprior = 1 - (alpha_values-alpha_min)/(alpha_max-alpha_min)
        temp_aprior = temp_aprior^psiprior
        # temp_aprior = temp_aprior/sum(temp_aprior)


        log_tempvals <- k_uniq*log(alpha_values) + lgamma(alpha_values) - lgamma(n+alpha_values)

        # print("log_tempvals = ")
        # print(log_tempvals)
        # print("temp_aprior = ")
        # print(temp_aprior)

        # temp_kgivenalpha <- exp(log_tempvals)
        #
        # # temp_kgivenalpha <- ((alpha_values)^(k_uniq))*gamma(alpha_values)/gamma(n+alpha_values)
        # temp_alpha_postprobs <- temp_kgivenalpha*temp_aprior


        logtemp_alpha_postprobs <- log_tempvals + log(temp_aprior)

        maxll <- max(logtemp_alpha_postprobs)

        temp_alpha_postprobs <- exp(logtemp_alpha_postprobs- maxll)

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

        post_probs_alphs = temp_alpha_postprobs/sum(temp_alpha_postprobs)

        # print("post_probs_alphs = ")
        # print(post_probs_alphs)

        #sample from 1 to alpha_gridsize
        index_alpha <- sample.int(n = alpha_gridsize, size = 1, prob = post_probs_alphs, replace = TRUE)

        alpha <- alpha_values[index_alpha]


      }else{
        stop("Alpha prior must be vh or george")
      }

    }

    #########  Sample new cluster parameter values  ######################################################


    # print("sample new clusters step")

    # varthetamat <- cbind(mu1_vec_train, mu2_vec_train, phi1_vec_train, gamma1_vec_train)
    #
    # #obtain the unique rows (components)
    # vartheta_unique_mat <- unique(varthetamat)
    #
    # create matrix for which rows correspond to observations (zstar_i, y_i)
    # columns correspond to parameter values v_j


    #including densities for censored and uncensored observations
    jointdens_mat <- matrix(NA, nrow = n , ncol = n)

    itemp <- 0


    #     print("calculating matrix of probabilities for uncensored observations")
    #
    #     # uncens_inds
    #
    #
    #     for(j in 1:n){
    #       # print("j = ")
    #       # print(j)
    #
    #       # for(i in uncens_inds){
    #       #   itemp <- itemp +1
    #       #   print("itemp")
    #       #
    #       #   print(itemp)
    #       #
    #       #   print("i = ")
    #       #   print(i)
    #
    #         #can calculate manually using the formula for the bivariate probability
    #
    #
    #       # print("length(mutemp_y) = ")
    #       # print(length(mutemp_y))
    #       #
    #       # print("length(uncens_inds = ")
    #       # print(length(uncens_inds))
    #
    #         jointdens_mat[uncens_inds,j] <- (1/(2*pi*sqrt(phi1_vec_train[j])))*
    #           exp(- ((phi1_vec_train[j]+gamma1_vec_train[j]^2)/(2*phi1_vec_train[j]))*(
    #            ( (z - offsetz)[uncens_inds] - mutemp_z[uncens_inds] - mu1_vec_train[j] )^2 -
    #              2*(gamma1_vec_train[j] /( (phi1_vec_train[j]+gamma1_vec_train[j]^2) )  )*
    #              ((z - offsetz)[uncens_inds] - mutemp_z[uncens_inds] - mu1_vec_train[j] )*
    #              (ystar[uncens_inds] - mutemp_y  - mu2_vec_train[j] ) +
    #              ((ystar[uncens_inds] - mutemp_y  - mu2_vec_train[j])^2 )/( phi1_vec_train[j]+gamma1_vec_train[j]^2)
    #           ) )
    #
    # #
    # #         Sigma_mat_temp_j <- cbind(c(1,
    # #                                     gamma1_vec_train[j]),
    # #                                   c(gamma1_vec_train[j],
    # #                                     phi1_vec_train[j]+gamma1_vec_train[j]^2))
    # #
    # #
    # #         # print("calcaulte density")
    # #
    # #         #or can use a pre-defined function
    # #         jointdens_mat[i,j] <-  dmvnorm(c((z - offsetz)[i],
    # #                                          ystar[i]),
    # #                                        mean = c(mutemp_z[i] + mu1_vec_train[j],
    # #                                                 mutemp_y[itemp]+ mu2_vec_train[j]),
    # #                                        sigma = Sigma_mat_temp_j)
    #
    #
    #       # }
    #
    #         if(any(is.nan(jointdens_mat[uncens_inds,j]))){
    #
    #           print("j = ")
    #           print(j)
    #
    #           print("phi1_vec_train[j]= ")
    #           print(phi1_vec_train[j])
    #
    #           print("phi1_vec_train = ")
    #           print(phi1_vec_train)
    #
    #
    #           print("gamma1_vec_train[j] = ")
    #           print(gamma1_vec_train[j])
    #
    #           print("mu1_vec_train[j] =")
    #           print(mu1_vec_train[j])
    #
    #           print("mu2_vec_train[j] =")
    #           print(mu2_vec_train[j])
    #
    #
    #           print("jointdens_mat[uncens_inds,j] = ")
    #           print(jointdens_mat[uncens_inds,j])
    #           stop("There are some NAN probabilities")
    #         }
    #
    #
    #     }
    #
    #
    #     print("calculating matrix of probabilities for censored observations")
    #
    #
    #     # cens_inds
    #     # for(i in cens_inds){
    #       # itemp <- itemp +1
    #       for(j in 1:n){
    #
    #
    #
    #         #can calculate manually using the formula for the biavariate probability
    #         # jointdens_mat[i,j] <- (1/(2*pi*sqrt(phi1_vec_train[j])))*
    #         #   exp(- ((phi1_vec_train[j]+gamma1_vec_train[j]^2)/(2*phi1_vec_train[j]))*(
    #         #    ( (z - offsetz)[i] - mu1_vec_train[j] )^2 -
    #         #      2*(gamma1_vec_train[j] /( (phi1_vec_train[j]+gamma1_vec_train[j]^2) )  )*
    #         #      ((z - offsetz)[i] - mu1_vec_train[j] )*
    #         #      (ystar[i] - mutemp_y[itemp]  - mu2_vec_train[j] ) +
    #         #      ((ystar[i] - mutemp_y[itemp]  - mu2_vec_train[j])^2 )/( phi1_vec_train[j]+gamma1_vec_train[j]^2)
    #         #   ) )
    #
    #
    #         # Sigma_mat_temp_j <- cbind(c(1,
    #         #                             gamma1_vec_train[j]),
    #         #                           c(gamma1_vec_train[j],
    #         #                             phi1_vec_train[j]+gamma1_vec_train[j]^2))
    #
    #
    #         #or can use a pre-defined function
    #
    #         # tempvec <- dnorm((z - offsetz)[cens_inds] ,
    #         #       mean = mutemp_z[cens_inds] + mu1_vec_train[j],
    #         #       sd = 1)
    #         #
    #         # print("length(tempvec) = ")
    #         # print(length(tempvec) )
    #         #
    #         # print("length(mutemp_z[cens_inds]) = ")
    #         # print(length(mutemp_z[cens_inds]) )
    #         #
    #         # print("length((z - offsetz)[cens_inds]) = ")
    #         # print(length((z - offsetz)[cens_inds]) )
    #
    #
    #         jointdens_mat[cens_inds,j] <- dnorm((z - offsetz)[cens_inds] ,
    #                                     mean = mutemp_z[cens_inds] + mu1_vec_train[j],
    #                                     sd = 1)
    #
    #         if(any(is.nan(jointdens_mat[cens_inds,j]))){
    #           print("jointdens_mat[cens_inds,j] = ")
    #           print(jointdens_mat[cens_inds,j])
    #           stop("There are some NAN probabilities")
    #         }
    #
    #
    #
    #       }
    #
    #     # }


    # print("jointdens_mat = ")
    #
    # print(jointdens_mat)

    #loop over all individuals and draw (or do not draw) new values


    # can't vectorize, must update vartheta in each step
    # # vectorize to speed up
    # nonunique_inds_vec <- which(duplicated(varthetamat))
    # num_nonunique <-
    #
    # #first consider non-unique observations
    # mutilde_vec <- mvrnorm(n = l,
    #                        mu = c(0, 0),
    #                        Sigma = M_mat)
    #



    # print("now loop through each observation for new samples")

    #reset index for uncensored observations
    itemp <- 0




    for(i in 1:n){
      # find number of individuals in the same cluster

      # if((i %%200) == 0){
      #   print("i = ")
      #   print(i)
      #
      # }

      # temp_samp_probs <- rep(NA,n)
      log_temp_samp_probs <- rep(NA,n)

      if(i %in% uncens_inds){
        itemp <- itemp +1
      }

      #there is probably a more efficient way of coding this
      temp_params <- varthetamat[i,]

      # colSums(t(varthetamat) == temp_params) == ncol(varthetamat)

      # code form https://stackoverflow.com/questions/32640682/check-whether-matrix-rows-equal-a-vector-in-r-vectorized
      #this gives the same number
      # n_rho_i <- sum(colSums(t(varthetamat) == temp_params) == ncol(varthetamat))

      #recalculate in each iteration because vartheta has been updated
      # n_rho_i <- sum(!Rfast::colsums(t(varthetamat) != temp_params))

      #continuous parameters, so fine to check just one value
      n_rho_i <- sum(varthetamat[,4] == temp_params[4])

      # n_rho_i_temp <- sum(!Rfast::colsums(t(varthetamat) != temp_params))
      #
      # if(n_rho_i != n_rho_i_temp){
      #   stop("n_rho_i != n_rho_i_temp")
      # }

      if(n_rho_i > 1){
        # sample from the base distribution H0

        # calculate probabilities
        if(i %in% cens_inds){
          # jointdens_mat[i , ] <- dnorm(rep((z - offsetz)[i], n) ,
          #       mean = mutemp_z[i] + mu1_vec_train,
          #       sd = 1)

          # temp_samp_probs <- exp(-((z - offsetz)[i] - (mutemp_z[i] + mu1_vec_train) )^2/2)/sqrt(2*pi)
          log_temp_samp_probs <- (-((z - offsetz)[i] - (mutemp_z[i] + mu1_vec_train) )^2/2) -
            0.5*log(2*pi)

          # jointdens_tilde <-  dnorm((z - offsetz)[i] ,
          #                           mean = mutemp_z[i] +  mutilde[1],
          #                           sd = 1)
        }else{

          temp_zdiff <- (z - offsetz)[i] - mutemp_z[i] - mu1_vec_train
          temp_ydiff <- ystar[i] - mutemp_y[itemp]  - mu2_vec_train

          # temp_samp_probs <-  (1/(2*pi*sqrt(phi1_vec_train)))*
          #   exp(- (1/(2*phi1_vec_train))*(
          #     (phi1_vec_train+gamma1_vec_train^2)*( temp_zdiff )^2 -
          #       2*(gamma1_vec_train   )*
          #       (temp_zdiff )*
          #       (temp_ydiff ) +
          #       ((temp_ydiff)^2 )
          #   ) )

          log_temp_samp_probs <-  -log(2*pi) - 0.5*log(phi1_vec_train)- (1/(2*phi1_vec_train))*(
            (phi1_vec_train+gamma1_vec_train^2)*( temp_zdiff )^2 -
              2*(gamma1_vec_train   )*
              (temp_zdiff )*
              (temp_ydiff ) +
              ((temp_ydiff)^2 ))

          # jointdens_tilde <-  dmvnorm(c((z - offsetz)[i],
          #                               ystar[i]),
          #                             mean = c(mutemp_z[i] + mutilde[1],
          #                                      mutemp_y[itemp]+ mutilde[2]),
          #                             sigma = Sigma_mat_temp_tilde)
        }

        mutilde <- Rfast::rmvnorm(n = 1,
                                  mu = c(0, 0),
                                  sigma = M_mat)

        if(cov_prior == "Ding"){
          tempsigma <- rinvwishart(nu = nu0,
                                   S = cding*diag(2))

          transmat <- cbind(c(1,0),c(0,1/sqrt(tempsigma[2,2])))
          tempomega <- (transmat %*% tempsigma) %*% transmat

          temprho <- tempomega[1,2]/(sqrt(tempomega[1,1]))

          # gamma1 <- tempomega[1,2]
          # phi1 <- tempomega[1,1] - (gamma1^2)

          gammatilde <- tempomega[1,2]
          phitilde <- tempomega[1,1] - (gammatilde^2)

        }else{

          phitilde <- 1/rgamma(n = 1, shape =  nzero/2, rate = S0/2)

          gammatilde <- rnorm(n = 1,
                              mean = gamma0,
                              sd = sqrt(tau*phitilde))
        }

        # calculate the density of the sampled value
        #can calculate manually using the formula for the biavariate probability
        # jointdens_mat[i,j] <- (1/(2*pi*sqrt(phi1_vec_train[j])))*
        #   exp(- ((phi1_vec_train[j]+gamma1_vec_train[j]^2)/(2*phi1_vec_train[j]))*(
        #    ( (z - offsetz)[i] - mu1_vec_train[i] )^2 -
        #      2*(gamma1_vec_train[j] /( (phi1_vec_train[j]+gamma1_vec_train[j]^2) )  )*
        #      ((z - offsetz)[i] - mu1_vec_train[i] )*
        #      (ystar[i] - mutemp_y[itemp]  - mu2_vec_train[i] ) +
        #      ((ystar[i] - mutemp_y[itemp]  - mu2_vec_train[i])^2 )/( phi1_vec_train[j]+gamma1_vec_train[j]^2)
        #   ) )


        Sigma_mat_temp_tilde <- cbind(c(1,
                                        gammatilde),
                                      c(gammatilde,
                                        phitilde+gammatilde^2))


        #or can use a pre-defined function

        if(i %in% cens_inds){
          # jointdens_tilde <-  dnorm((z - offsetz)[i] ,
          #                           mean = mutemp_z[i] +  mutilde[1],
          #                           sd = 1)

          # jointdens_tilde <- exp(-((z - offsetz)[i] - (mutemp_z[i] + mutilde[1]) )^2/2)/sqrt(2*pi)
          log_jointdens_tilde <- (-((z - offsetz)[i] - (mutemp_z[i] + mutilde[1]) )^2/2) -
            0.5*log(2*pi)

        }else{
          # jointdens_tilde <-  dmvnorm(c((z - offsetz)[i],
          #                               ystar[i]),
          #                             mean = c(mutemp_z[i] + mutilde[1],
          #                                      mutemp_y[itemp]+ mutilde[2]),
          #                             sigma = Sigma_mat_temp_tilde)

          temp_zdiff <- (z - offsetz)[i] - mutemp_z[i] - mutilde[1]
          temp_ydiff <- ystar[i] - mutemp_y[itemp]  - mutilde[2]

          # jointdens_tilde <-  (1/(2*pi*sqrt(phitilde)))*
          #   exp(- (1/(2*phitilde))*(
          #     (phitilde+gammatilde^2)*( temp_zdiff )^2 -
          #       2*(gammatilde  )*
          #       (temp_zdiff )*
          #       (temp_ydiff ) +
          #       (temp_ydiff^2 )
          #   ) )

          log_jointdens_tilde <- - log(2*pi) - 0.5*log(phitilde) - # (1/(2*pi*sqrt(phitilde)))*
            (1/(2*phitilde))*(
              (phitilde+gammatilde^2)*( temp_zdiff )^2 -
                2*(gammatilde  )*
                (temp_zdiff )*
                (temp_ydiff ) +
                (temp_ydiff^2 )
            )

        }


        #unnecessary to calculate this normalizing constant if sample functionautomatically normalizes
        # Ctemp_i <- (alpha+n-1)/( alpha*jointdens_tilde + sum(jointdens_mat[-i,j]) )

        # Ctemp_i <- 1/( alpha*jointdens_tilde + sum(jointdens_mat[-i,j]) )



        # temp_samp_probs <- jointdens_mat[i , ]

        # temp_samp_probs[i] <- alpha*jointdens_tilde

        log_temp_samp_probs[i] <- log(alpha)+log_jointdens_tilde

        maxll <- max(log_temp_samp_probs)

        temp_samp_probs <- exp(log_temp_samp_probs- maxll)

        temp_samp_probs <- temp_samp_probs/sum(temp_samp_probs) # this step is probably unnecessary

        # print("temp_samp_probs= ")
        # print(temp_samp_probs)
        # some of the observations not equal to i have the same vartheta values
        # so it is possible to sample the same aprameters again

        # new_vartheta <- sample(Ctemp_i*temp_samp_probs, size = 1)
        new_varind <- sample.int(n = n, size = 1, prob = temp_samp_probs, replace = TRUE)

        new_vartheta <- NA

        if(new_varind == i){
          #accept new proposal
          mu1_vec_train[i] <- mutilde[1]
          mu2_vec_train[i] <- mutilde[2]

          phi1_vec_train[i] <- phitilde
          gamma1_vec_train[i] <- gammatilde


          #update densities

          # if(i %in% uncens_inds){
          #   # itemp <- itemp +1
          #
          #   #only need to update rows from i+1 because yet to be sampled
          #
          #   if(i==n){
          #
          #   }else{
          #
          #     for(indtemp in (i+1):n){
          #
          #
          #
          #       #can calculate manually using the formula for the biavariate probability
          #       # jointdens_mat[i,j] <- (1/(2*pi*sqrt(phi1_vec_train[j])))*
          #       #   exp(- ((phi1_vec_train[j]+gamma1_vec_train[j]^2)/(2*phi1_vec_train[j]))*(
          #       #    ( (z - offsetz)[i] - mu1_vec_train[j] )^2 -
          #       #      2*(gamma1_vec_train[j] /( (phi1_vec_train[j]+gamma1_vec_train[j]^2) )  )*
          #       #      ((z - offsetz)[i] - mu1_vec_train[j] )*
          #       #      (ystar[i] - mutemp_y[itemp]  - mu2_vec_train[j] ) +
          #       #      ((ystar[i] - mutemp_y[itemp]  - mu2_vec_train[j])^2 )/( phi1_vec_train[j]+gamma1_vec_train[j]^2)
          #       #   ) )
          #
          #
          #       Sigma_mat_temp_j <- cbind(c(1,
          #                                   gamma1_vec_train[i]),
          #                                 c(gamma1_vec_train[i],
          #                                   phi1_vec_train[i]+gamma1_vec_train[i]^2))
          #
          #
          #       #or can use a pre-defined function
          #       jointdens_mat[indtemp,i] <-  dmvnorm(c((z - offsetz)[indtemp],
          #                                              ystar[indtemp]),
          #                                            mean = c(mutemp_z[indtemp] + mu1_vec_train[i],
          #                                                     mutemp_y[itemp]+ mu2_vec_train[i]),
          #                                            sigma = Sigma_mat_temp_j)
          #
          #
          #       if(any(is.nan(jointdens_mat[indtemp,i]))){
          #         print("jointdens_mat[indtemp,i] = ")
          #         print(jointdens_mat[indtemp,i])
          #         stop("There are some NAN probabilities")
          #       }
          #
          #
          #     }
          #
          #
          #   }
          #
          # }
          #
          # # cens_inds
          # if(i %in% cens_inds){
          #   # itemp <- itemp +1
          #   for(indtemp in (i+1):n){
          #
          #
          #
          #     #can calculate manually using the formula for the biavariate probability
          #     # jointdens_mat[i,j] <- (1/(2*pi*sqrt(phi1_vec_train[j])))*
          #     #   exp(- ((phi1_vec_train[j]+gamma1_vec_train[j]^2)/(2*phi1_vec_train[j]))*(
          #     #    ( (z - offsetz)[i] - mu1_vec_train[j] )^2 -
          #     #      2*(gamma1_vec_train[j] /( (phi1_vec_train[j]+gamma1_vec_train[j]^2) )  )*
          #     #      ((z - offsetz)[i] - mu1_vec_train[j] )*
          #     #      (ystar[i] - mutemp_y[itemp]  - mu2_vec_train[j] ) +
          #     #      ((ystar[i] - mutemp_y[itemp]  - mu2_vec_train[j])^2 )/( phi1_vec_train[j]+gamma1_vec_train[j]^2)
          #     #   ) )
          #
          #
          #     # Sigma_mat_temp_j <- cbind(c(1,
          #     #                             gamma1_vec_train[i]),
          #     #                           c(gamma1_vec_train[i],
          #     #                             phi1_vec_train[i]+gamma1_vec_train[i]^2))
          #
          #
          #     #or can use a pre-defined function
          #     jointdens_mat[indtemp,i] <- dnorm((z - offsetz)[indtemp] ,
          #                                       mean = mutemp_z[indtemp] + mu1_vec_train[i],
          #                                       sd = 1)
          #
          #
          #     if(any(is.nan(jointdens_mat[indtemp,i]))){
          #       print("jointdens_mat[indtemp,i] = ")
          #       print(jointdens_mat[indtemp,i])
          #       stop("There are some NAN probabilities")
          #     }
          #
          #
          #   }
          #
          # }


        }else{
          #set parameters equal to those of cluster new_varind
          mu1_vec_train[i] <- mu1_vec_train[new_varind]
          mu2_vec_train[i] <- mu2_vec_train[new_varind]

          phi1_vec_train[i] <- phi1_vec_train[new_varind]
          gamma1_vec_train[i] <- gamma1_vec_train[new_varind]

          # jointdens_mat[,i] <- jointdens_mat[,new_varind]

        }


        varthetamat[i,] <- c(mu1_vec_train[i], mu2_vec_train[i], phi1_vec_train[i], gamma1_vec_train[i])



      }else{ #nrho equal to 1

        if(n_rho_i != 1){
          stop("n_rho_i < 1 and not equal to 1")
        }


        #calculate probabilties
        if(i %in% cens_inds){


          # jointdens_mat[i , ] <- dnorm(rep((z - offsetz)[i], n) ,
          #                              mean = mutemp_z[i] + mu1_vec_train,
          #                              sd = 1)

          # temp_samp_probs <-exp(-((z - offsetz)[i] - (mutemp_z[i] + mu1_vec_train) )^2/2)/sqrt(2*pi)
          log_temp_samp_probs <- (-((z - offsetz)[i] - (mutemp_z[i] + mu1_vec_train) )^2/2) -
            0.5*log(2*pi)

          # jointdens_tilde <-  dnorm((z - offsetz)[i] ,
          #                           mean = mutemp_z[i] +  mutilde[1],
          #                           sd = 1)
        }else{

          temp_zdiff <- (z - offsetz)[i] - mutemp_z[i] - mu1_vec_train
          temp_ydiff <- ystar[i] - mutemp_y[itemp]  - mu2_vec_train

          # temp_samp_probs <-  (1/(2*pi*sqrt(phi1_vec_train)))*
          #   exp(- (1/(2*phi1_vec_train))*(
          #     (phi1_vec_train+gamma1_vec_train^2)*(temp_zdiff  )^2 -
          #       2*(gamma1_vec_train   )*
          #       (temp_zdiff )*
          #       temp_ydiff +
          #       (temp_ydiff^2 )
          #   ) )

          log_temp_samp_probs <-  -log(2*pi) - 0.5*log(phi1_vec_train) -
            (1/(2*phi1_vec_train))*(
              (phi1_vec_train+gamma1_vec_train^2)*(temp_zdiff  )^2 -
                2*(gamma1_vec_train   )*
                (temp_zdiff )*
                temp_ydiff +
                (temp_ydiff^2 ))

          # jointdens_tilde <-  dmvnorm(c((z - offsetz)[i],
          #                               ystar[i]),
          #                             mean = c(mutemp_z[i] + mutilde[1],
          #                                      mutemp_y[itemp]+ mutilde[2]),
          #                             sigma = Sigma_mat_temp_tilde)
        }




        # temp_samp_probs <- jointdens_mat[i, ]
        # temp_samp_probs[i] <- alpha*temp_samp_probs[i]

        log_temp_samp_probs[i] <- log(alpha)+log_temp_samp_probs[i]

        maxll <- max(log_temp_samp_probs)

        temp_samp_probs <- exp(log_temp_samp_probs- maxll)

        temp_samp_probs <- temp_samp_probs/sum(temp_samp_probs) # this step is probably unnecessary

        # print("temp_samp_probs= ")
        # print(temp_samp_probs)

        new_varind <- sample.int(n = n, size = 1, prob = temp_samp_probs, replace = TRUE)

        new_vartheta <- NA

        if(new_varind == i){

          #parameters unchanged

          # mu1_vec_train[i] <- mutilde[1]
          # mu2_vec_train[i] <- mutilde[2]
          #
          # phi1_vec_train[i] <- phitilde
          # gamma1_vec_train[i] <- gammatilde
          #
        }else{
          mu1_vec_train[i] <- mu1_vec_train[new_varind]
          mu2_vec_train[i] <- mu2_vec_train[new_varind]

          phi1_vec_train[i] <- phi1_vec_train[new_varind]
          gamma1_vec_train[i] <- gamma1_vec_train[new_varind]

          varthetamat[i,] <- c(mu1_vec_train[i], mu2_vec_train[i], phi1_vec_train[i], gamma1_vec_train[i])

          # jointdens_mat[,i] <- jointdens_mat[,new_varind]



          # if(i %in% uncens_inds){
          #   # itemp <- itemp +1
          #   for(indtemp in 1:n){
          #
          #
          #
          #     #can calculate manually using the formula for the biavariate probability
          #     # jointdens_mat[i,j] <- (1/(2*pi*sqrt(phi1_vec_train[j])))*
          #     #   exp(- ((phi1_vec_train[j]+gamma1_vec_train[j]^2)/(2*phi1_vec_train[j]))*(
          #     #    ( (z - offsetz)[i] - mu1_vec_train[j] )^2 -
          #     #      2*(gamma1_vec_train[j] /( (phi1_vec_train[j]+gamma1_vec_train[j]^2) )  )*
          #     #      ((z - offsetz)[i] - mu1_vec_train[j] )*
          #     #      (ystar[i] - mutemp_y[itemp]  - mu2_vec_train[j] ) +
          #     #      ((ystar[i] - mutemp_y[itemp]  - mu2_vec_train[j])^2 )/( phi1_vec_train[j]+gamma1_vec_train[j]^2)
          #     #   ) )
          #
          #
          #     Sigma_mat_temp_j <- cbind(c(1,
          #                                 gamma1_vec_train[i]),
          #                               c(gamma1_vec_train[i],
          #                                 phi1_vec_train[i]+gamma1_vec_train[i]^2))
          #
          #
          #     #or can use a pre-defined function
          #     jointdens_mat[indtemp,i] <-  dmvnorm(c((z - offsetz)[indtemp],
          #                                            ystar[indtemp]),
          #                                          mean = c(mutemp_z[indtemp] + mu1_vec_train[i],
          #                                                   mutemp_y[itemp]+ mu2_vec_train[i]),
          #                                          sigma = Sigma_mat_temp_j)
          #
          #
          #   }
          #
          # }
          #
          # # cens_inds
          # if(i %in% cens_inds){
          #   # itemp <- itemp +1
          #   for(indtemp in 1:n){
          #
          #
          #
          #     #can calculate manually using the formula for the biavariate probability
          #     # jointdens_mat[i,j] <- (1/(2*pi*sqrt(phi1_vec_train[j])))*
          #     #   exp(- ((phi1_vec_train[j]+gamma1_vec_train[j]^2)/(2*phi1_vec_train[j]))*(
          #     #    ( (z - offsetz)[i] - mu1_vec_train[j] )^2 -
          #     #      2*(gamma1_vec_train[j] /( (phi1_vec_train[j]+gamma1_vec_train[j]^2) )  )*
          #     #      ((z - offsetz)[i] - mu1_vec_train[j] )*
          #     #      (ystar[i] - mutemp_y[itemp]  - mu2_vec_train[j] ) +
          #     #      ((ystar[i] - mutemp_y[itemp]  - mu2_vec_train[j])^2 )/( phi1_vec_train[j]+gamma1_vec_train[j]^2)
          #     #   ) )
          #
          #
          #     # Sigma_mat_temp_j <- cbind(c(1,
          #     #                             gamma1_vec_train[i]),
          #     #                           c(gamma1_vec_train[i],
          #     #                             phi1_vec_train[i]+gamma1_vec_train[i]^2))
          #
          #
          #     #or can use a pre-defined function
          #     jointdens_mat[indtemp,i] <- dnorm((z - offsetz)[indtemp] ,
          #                                       mean = mutemp_z[indtemp] + mu1_vec_train[i],
          #                                       sd = 1)
          #
          #     if(any(is.nan(jointdens_mat[indtemp,i]))){
          #       print("jointdens_mat[indtemp,i] = ")
          #       print(jointdens_mat[indtemp,i])
          #       stop("There are some NAN probabilities")
          #     }
          #
          #   }
          #
          # } # end loop over cens_inds
          #


        } #end else new var ind !=i

      } # end else statement nrho equal to 1

    } # end loop over uncensored observations observations




    # loop over censored observations

    # for(i in cens_inds){
    #   # find number of individuals in the same cluster
    #   # itemp <- itemp +1
    #
    #
    #   #there is probably a more efficient way of coding this
    #   temp_params <- varthetamat[i,]
    #
    #   # colSums(t(varthetamat) == temp_params) == ncol(varthetamat)
    #
    #   # code form https://stackoverflow.com/questions/32640682/check-whether-matrix-rows-equal-a-vector-in-r-vectorized
    #   #this gives the same number
    #   # n_rho_i <- sum(colSums(t(varthetamat) == temp_params) == ncol(varthetamat))
    #
    #
    #   n_rho_i <- sum(!colSums(t(varthetamat) != temp_params))
    #
    #
    #
    #
    #   if(n_rho_i >1){
    #     # sample from the base distribution H0
    #
    #
    #     mutilde <- mvrnorm(n = 1,
    #                        mu = c(0, 0),
    #                        Sigma = M_mat)
    #
    #
    #     gammatilde <- rnorm(1,
    #                         mean = gamma0,
    #                         sd = sqrt(G0))
    #
    #     phitilde <- 1/rgamma(n = 1, shape =  nzero/2, rate = S0/2)
    #
    #
    #     # calculate the density of the sampled value
    #     #can calculate manually using the formula for the biavariate probability
    #     # jointdens_mat[i,j] <- (1/(2*pi*sqrt(phi1_vec_train[j])))*
    #     #   exp(- ((phi1_vec_train[j]+gamma1_vec_train[j]^2)/(2*phi1_vec_train[j]))*(
    #     #    ( (z - offsetz)[i] - mu1_vec_train[i] )^2 -
    #     #      2*(gamma1_vec_train[j] /( (phi1_vec_train[j]+gamma1_vec_train[j]^2) )  )*
    #     #      ((z - offsetz)[i] - mu1_vec_train[i] )*
    #     #      (ystar[i] - mutemp_y[itemp]  - mu2_vec_train[i] ) +
    #     #      ((ystar[i] - mutemp_y[itemp]  - mu2_vec_train[i])^2 )/( phi1_vec_train[j]+gamma1_vec_train[j]^2)
    #     #   ) )
    #
    #
    #     # Sigma_mat_temp_tilde <- cbind(c(1,
    #     #                                 gammatilde),
    #     #                               c(gammatilde,
    #     #                                 phitilde+gammatilde^2))
    #
    #
    #     #or can use a pre-defined function
    #     # jointdens_tilde <-  dmvnorm(c((z - offsetz)[i],
    #     #                               ystar[i]),
    #     #                             mean = c(mutemp_z[i] + mutilde[1],
    #     #                                      mutemp_y[itemp]+ mutilde[2]),
    #     #                             sigma = Sigma_mat_temp_tilde)
    #
    #     jointdens_tilde <-  dnorm((z - offsetz)[i] ,
    #           mean = mutemp_z[i] +  mutilde[1],
    #           sd = 1)
    #
    #
    #     #unnecessary to calculate this normalizing constant if sample functionautomatically normalizes
    #     # Ctemp_i <- (alpha+n-1)/( alpha*jointdens_tilde + sum(jointdens_mat[-i,j]) )
    #
    #     # Ctemp_i <- 1/( alpha*jointdens_tilde + sum(jointdens_mat[-i,j]) )
    #
    #     temp_samp_probs <- jointdens_mat[i , ]
    #
    #     temp_samp_probs[i] <- alpha*jointdens_tilde
    #
    #
    #     # new_vartheta <- sample(Ctemp_i*temp_samp_probs, size = 1)
    #     new_varind <- sample(1:n, size = 1, prob = temp_samp_probs)
    #
    #     new_vartheta <- NA
    #
    #     if(new_varind == i){
    #       #accept new proposal
    #       mu1_vec_train[i] <- mutilde[1]
    #       mu2_vec_train[i] <- mutilde[2]
    #
    #       phi1_vec_train[i] <- phitilde
    #       gamma1_vec_train[i] <- gammatilde
    #
    #     }else{
    #       #set parameters equal to those of cluster new_varind
    #       mu1_vec_train[i] <- mu1_vec_train[new_varind]
    #       mu2_vec_train[i] <- mu2_vec_train[new_varind]
    #
    #       phi1_vec_train[i] <- phi1_vec_train[new_varind]
    #       gamma1_vec_train[i] <- gamma1_vec_train[new_varind]
    #     }
    #
    #
    #   }else{ #nrho equal to 1
    #
    #     if(n_rho_i != 1){
    #       stop("n_rho_i < 1 and not equal to 1")
    #     }
    #
    #
    #     temp_samp_probs <- jointdens_mat[, i]
    #     temp_samp_probs[i] <- alpha*jointdens_mat[i, i]
    #
    #     new_varind <- sample(1:n, size = 1, prob = temp_samp_probs)
    #
    #     new_vartheta <- NA
    #
    #     if(new_varind == i){
    #
    #       #parameters unchanged
    #
    #       # mu1_vec_train[i] <- mutilde[1]
    #       # mu2_vec_train[i] <- mutilde[2]
    #       #
    #       # phi1_vec_train[i] <- phitilde
    #       # gamma1_vec_train[i] <- gammatilde
    #       #
    #     }else{
    #       mu1_vec_train[i] <- mu1_vec_train[new_varind]
    #       mu2_vec_train[i] <- mu2_vec_train[new_varind]
    #
    #       phi1_vec_train[i] <- phi1_vec_train[new_varind]
    #       gamma1_vec_train[i] <- gamma1_vec_train[new_varind]
    #     }
    #
    #
    #
    #   } # end else statement nrho equal to 1
    #
    #
    #
    # } # end lop over censored observations

    #########  Resample or remix vartheta values ######################################################
    if(any(phi1_vec_train <= 0)){
      stop("Line 1797 some phi1_vec_train values <= 0")
    }

    # print("remix step")

    # varthetamat <- cbind(mu1_vec_train, mu2_vec_train, phi1_vec_train, gamma1_vec_train)

    u_z <- z - offsetz - mutemp_z

    u_y <- ystar[uncens_inds] - mutemp_y


    # obtain the unique rows (components)
    # Must do this again because sa,pled new values in previous step
    vartheta_unique_mat <- unique(varthetamat)

    #number of unique components
    k_uniq <- nrow(vartheta_unique_mat)


    #The values are conditionally independent

    #loop through the unique values and take draws

    for(j in 1:k_uniq){

      # there are 3 cases
      # Only censored observations in cluster j
      # Only uncensored observations in cluster j
      # Both censored and uncensored observations in cluster j

      #First find the indices of the observations in cluster j

      temp_params <- vartheta_unique_mat[j,]
      mutilde <- c(temp_params[1], temp_params[2])
      phitilde <- temp_params[3]
      gammatilde <- temp_params[4]


      # print("temp_params = ")
      # print(temp_params)

      # clust_boolvec <- !Rfast::colsums(t(varthetamat) != temp_params)
      #
      clust_boolvec <- (varthetamat[,4] == temp_params[4])

      # clust_boolvec_temp <- !Rfast::colsums(t(varthetamat) != temp_params)
      #
      # if(any(clust_boolvec != clust_boolvec_temp)){
      #   stop("any(clust_boolvec != clust_boolvec_temp)")
      # }


      n_rho_j <- sum(clust_boolvec)


      num_cens_temp <- sum(clust_boolvec[cens_inds])
      num_uncens_temp <- sum(clust_boolvec[uncens_inds])


      if( (num_cens_temp > 0) & (num_uncens_temp == 0)){
        #only censored observations

        # mutilde <- mvrnorm(n = 1,
        #                    mu = c(0, 0),
        #                    Sigma = M_mat)

        if(cov_prior == "Ding"){
          tempsigma <- rinvwishart(nu = nu0,
                                   S = cding*diag(2))

          transmat <- cbind(c(1,0),c(0,1/sqrt(tempsigma[2,2])))
          tempomega <- (transmat %*% tempsigma) %*% transmat

          temprho <- tempomega[1,2]/(sqrt(tempomega[1,1]))

          # gamma1 <- tempomega[1,2]
          # phi1 <- tempomega[1,1] - (gamma1^2)

          gammatilde <- tempomega[1,2]
          phitilde <- tempomega[1,1] - (gammatilde^2)

        }else{
          phitilde <- 1/rgamma(n = 1, shape =  nzero/2, rate = S0/2)

          gammatilde <- rnorm(1,
                              mean = gamma0,
                              sd = sqrt(tau*phitilde))
        }

        mutilde[1] <- rnorm(n = 1,
                            mean = sum(u_z[clust_boolvec])/( (1/M_mat[1,1]) + n_rho_j  ),
                            sd = sqrt(1/( (1/M_mat[1,1]) + n_rho_j  )) )

        mutilde[2] <- rnorm(n = 1,
                            mean =   mutilde[1]*M_mat[1,2]/M_mat[1,1],
                            sd =  sqrt( M_mat[2,2] - (M_mat[1,2]^2/M_mat[1,1])   ))


      }
      if( (num_cens_temp == 0) & (num_uncens_temp > 0)){
        #only uncensored observations

        clust_uncens_for_y <- which(uncens_inds %in% which(clust_boolvec))


        # print("uncens_inds = ")
        # print(uncens_inds)
        # print("which(clust_boolvec) = ")
        # print(which(clust_boolvec))
        # print("clust_uncens_for_y = ")
        # print(clust_uncens_for_y)

        # clust_uncens_for_z <- clust_boolvec[uncens_inds]
        clust_uncens_boolvec <- uncens_boolvec & clust_boolvec

        # Exact sampling is not feasible, make Gibbs samples

        # not clear if can just take one Gibbs sample, i.e. num_sample <- 1
        # or if correlation across samples might be an issue
        num_sample <- 1



        for(samp_ind in 1:num_sample){

          Sigma_mat_temp_j <- cbind(c(1,
                                      gammatilde),
                                    c(gammatilde,
                                      phitilde+gammatilde^2))

          tempsigmainv <- solve(Sigma_mat_temp_j)
          tempsigmamat <- solve(M_inv+ n_rho_j*tempsigmainv )

          mutilde <- Rfast::rmvnorm(n = 1,
                                    mu = (tempsigmamat%*%tempsigmainv) %*%
                                      c(sum(u_z[clust_uncens_boolvec]),sum(u_y[clust_uncens_for_y])),
                                    sigma = tempsigmamat )



          # print("clust_uncens_boolvec = ")
          # print(clust_uncens_boolvec)

          # print("length(u_y) = ")
          # print(length(u_y))
          #
          # print("clust_uncens_for_y = ")
          # print(clust_uncens_for_y)
          #
          # print("u_y[clust_uncens_for_y] = ")
          # print(u_y[clust_uncens_for_y])
          #
          # print("u_z[clust_uncens_boolvec] = ")
          # print(u_z[clust_uncens_boolvec])
          #
          # print("gammatilde= ")
          # print(gammatilde)
          #
          # print("mutilde= ")
          # print(mutilde)
          #
          # print("dbar_temp= ")
          # print(dbar_temp)


          # print("(nzero/2) +  (n_rho_j + 1)/2 = ")
          # print((nzero/2) +  (n_rho_j + 1)/2)



          if(cov_prior == "Ding"){

            y_epsilon2 <- u_y[clust_uncens_for_y]  - mutilde[2]

            rho1 <- gammatilde/sqrt(Sigma_mat_temp_j[2,2])

            # print( "nu0 = " )
            # print(nu0)
            #
            # print( "cding = " )
            # print(cding)
            #
            # print( "rho1 = " )
            # print(rho1)

            sigz2 <- 1/rgamma(n = 1,
                              shape = nu0/2,
                              rate = cding/(2*(1- (rho1^2))))


            z_epsilon2 <- sqrt(sigz2)*(u_z[clust_uncens_boolvec] - mutilde[1])

            zsquares <- crossprod(z_epsilon2, z_epsilon2)[1]
            ysquares <- crossprod(y_epsilon2, y_epsilon2)[1]
            zycross <- crossprod(z_epsilon2, y_epsilon2)[1]

            Stemp <- cbind(c(ysquares,zycross),
                           c(zycross, zsquares))

            # print(" nu0 = ")
            # print( nu0)
            # print(" n_rho_j  = ")
            # print( n_rho_j )
            #
            # print(" Stemp = ")
            # print( Stemp)
            # print(" cding*diag(2) = ")
            # print( cding*diag(2))
            # print(" Stemp+cding*diag(2) = ")
            # print( Stemp+cding*diag(2))

            tempsigma <- rinvwishart(nu = n_rho_j + nu0,
                                     S = Stemp+cding*diag(2))


            transmat <- cbind(c(1,0),c(0,1/sqrt(tempsigma[2,2])))
            tempomega <- (transmat %*% tempsigma) %*% transmat

            temprho <- tempomega[1,2]/(sqrt(tempomega[1,1]))

            gammatilde <- tempomega[1,2]
            phitilde <- tempomega[1,1] - (gammatilde^2)



          }else{

            dbar_temp <- (S0/2) + (gammatilde^2/tau) +
              (1/2)*sum( (u_y[clust_uncens_for_y] - mutilde[2] - gammatilde*(u_z[clust_uncens_boolvec] - mutilde[1]) )^2 )

            phitilde <- 1/rgamma(n = 1, shape =  (nzero/2) +  (n_rho_j + 1)/2 , rate = dbar_temp)


            gammabar_temp <- sum( (u_z[clust_uncens_boolvec] - mutilde[1])*(u_y[clust_uncens_for_y] - mutilde[2]) ) /
              ((1/tau) + sum( (u_z[clust_uncens_boolvec] - mutilde[1])^2 ) )

            # print("gammabar_temp= ")
            # print(gammabar_temp)
            #
            # print("sqrt( phitilde / ((1/tau) + sum( (u_z[clust_uncens_for_z] - mutilde[1])^2 ) )   ) = ")
            # print(sqrt( phitilde / ((1/tau) + sum( (u_z[clust_uncens_boolvec] - mutilde[1])^2 ) )   ))

            gammatilde <- rnorm(1,
                                mean = gammabar_temp,
                                sd = sqrt( phitilde / ((1/tau) + sum( (u_z[clust_uncens_boolvec] - mutilde[1])^2 ) )   ))

          }


        } # end of Gibbs sampler loop


      }
      if( (num_cens_temp > 0) & (num_uncens_temp > 0)){
        #Both censored and uncensored observations




        #obtain number censored and uncensored

        clust_cens_boolvec <- cens_boolvec & clust_boolvec
        clust_uncens_boolvec <- uncens_boolvec & clust_boolvec

        # clust_uncens_for_y <- which(uncens_boolvec %in% clust_boolvec)
        clust_uncens_for_y <- which(uncens_inds %in% which(clust_boolvec))
        # print("clust_uncens_for_y = ")
        # print(clust_uncens_for_y)

        # num_cens_inclust <- sum(clust_cens_boolvec)
        # num_uncens_inclust <- sum(clust_uncens_boolvec)




        # The mean of the error in the z equation can be sampled directly
        mutilde[1] <- rnorm(n = 1,
                            mean = sum(u_z[clust_boolvec])/( (1/M_mat[1,1]) + n_rho_j  ),
                            sd = sqrt(1/( (1/M_mat[1,1]) + n_rho_j  )) )

        # The other cluster parameters must be sampled in a Gibbs sampler

        M2bar <- 1/( ( M_mat[1,1]/(M_mat[1,1]*M_mat[2,2] - M_mat[1,2]^2) )  + (num_uncens_temp/phitilde )  )

        # sum(u_y[clust_cens_inds] -  gammatilde*(u_z[clust_cens_inds]  -mutilde[1] ))

        mu2bar <- M2bar*( ( M_mat[1,1]*mutilde[1]/(M_mat[1,1]*M_mat[2,2] - M_mat[1,2]^2) )  +
                            (sum(u_y[clust_uncens_for_y] -  gammatilde*(u_z[clust_uncens_boolvec]  -mutilde[1] ))/phitilde ) )


        mutilde[2] <- rnorm(n = 1,
                            mean = mu2bar,
                            sd = sqrt(M2bar) )


        if(cov_prior == "Ding"){

          Sigma_mat_temp_j <- cbind(c(1,
                                      gammatilde),
                                    c(gammatilde,
                                      phitilde+gammatilde^2))

          y_epsilon2 <- u_y[clust_uncens_for_y]  - mutilde[2]

          rho1 <- gammatilde/sqrt(Sigma_mat_temp_j[2,2])

          # print( "gammatilde = " )
          # print(gammatilde)
          #
          # print( "sqrt(Sigma_mat_temp_j[2,2]) = " )
          # print(sqrt(Sigma_mat_temp_j[2,2]))
          #
          # print( "nu0 = " )
          # print(nu0)
          #
          # print( "cding = " )
          # print(cding)
          #
          # print( "rho1 = " )
          # print(rho1)

          sigz2 <- 1/rgamma(n = 1,
                            shape = nu0/2,
                            rate = cding/(2*(1- (rho1^2))))


          z_epsilon2 <- sqrt(sigz2)*(u_z[clust_uncens_boolvec] - mutilde[1])

          zsquares <- crossprod(z_epsilon2, z_epsilon2)[1]
          ysquares <- crossprod(y_epsilon2, y_epsilon2)[1]
          zycross <- crossprod(z_epsilon2, y_epsilon2)[1]

          Stemp <- cbind(c(ysquares,zycross),
                         c(zycross, zsquares))


          # print(" nu0 = ")
          # print( nu0)
          # print(" num_uncens_temp  = ")
          # print( num_uncens_temp )
          #
          # print(" Stemp = ")
          # print( Stemp)
          # print(" cding*diag(2) = ")
          # print( cding*diag(2))
          # print(" Stemp+cding*diag(2) = ")
          # print( Stemp+cding*diag(2))


          tempsigma <- rinvwishart(nu = num_uncens_temp + nu0,
                                   S = Stemp+cding*diag(2))


          transmat <- cbind(c(1,0),c(0,1/sqrt(tempsigma[2,2])))
          tempomega <- (transmat %*% tempsigma) %*% transmat

          temprho <- tempomega[1,2]/(sqrt(tempomega[1,1]))

          gammatilde <- tempomega[1,2]
          phitilde <- tempomega[1,1] - (gammatilde^2)



        }else{
          gammabar_temp <- sum( (u_z[clust_uncens_boolvec] - mutilde[1])*(u_y[clust_uncens_for_y] - mutilde[2]) ) /
            ((1/tau) + sum( (u_z[clust_uncens_boolvec] - mutilde[1])^2 ) )


          gammatilde <- rnorm(1,
                              mean = gammabar_temp,
                              sd = sqrt( phitilde / ((1/tau) + sum( (u_z[clust_uncens_boolvec] - mutilde[1])^2 ) )   ))


          dbar_temp <- (S0/2) + (gammatilde^2/tau) +
            (1/2)*sum( (u_y[clust_uncens_for_y] - mutilde[2] - gammatilde*(u_z[clust_uncens_boolvec] - mutilde[1]) )^2 )

          phitilde <- 1/rgamma(n = 1, shape =  (nzero/2) +  (num_uncens_temp + 1)/2 , rate = dbar_temp)
        }



      }

      # now must update vartheta_mat and vartheta_unique_mat


      # tempmat <- matrix( rnorm(4*10), nrow = 4, ncol = 10)
      #
      # tempmat[c(1,3),1 ] <- 1
      #

      #presumably this is less efficient because creates a new matrix
      # varthetamat[clust_boolvec,] <- rep(c(mutilde, phitilde, gammatilde), each = length(clust_boolvec))

      varthetamat[clust_boolvec,1] <- mutilde[1]
      varthetamat[clust_boolvec,2] <- mutilde[2]
      varthetamat[clust_boolvec,3] <- phitilde
      varthetamat[clust_boolvec,4] <- gammatilde


    } #end of loop over unique cluster values

    # presumably only update this after loop over all unique clusters because conditionally independent

    vartheta_unique_mat <- unique(varthetamat)

    #number of unique components
    k_uniq <- nrow(vartheta_unique_mat)


    #update the vectors of parameter values

    # varthetamat <- cbind(mu1_vec_train, mu2_vec_train, phi1_vec_train, gamma1_vec_train)

    if(any(phi1_vec_train <= 0)){
      stop("Line 1797 some phi1_vec_train values <= 0")
    }

    mu1_vec_train <- varthetamat[,1]
    mu2_vec_train <- varthetamat[,2]
    phi1_vec_train <- varthetamat[,3]
    gamma1_vec_train <- varthetamat[,4]

    if(any(phi1_vec_train <= 0)){
      stop("Line 1797 some phi1_vec_train values <= 0")
    }

    ###### Sample test vartheta values   ###############################




    if(ntest>0){

      temp_sample_probs <- c(rep(1/(alpha+n) , n), alpha/(alpha+n) )



      for(i in 1:ntest){

        temp_ind <- sample.int(n = (n+1), size = 1, prob = temp_sample_probs, replace = TRUE)

        if(temp_ind == n+1){
          # sample from the base distribution

          if(cov_prior == "Ding"){


            mutilde <- Rfast::rmvnorm(n = 1,
                                      mu = c(0, 0),
                                      sigma = M_mat)

            tempsigma <- rinvwishart(nu = nu0,
                                     S = cding*diag(2))

            transmat <- cbind(c(1,0),c(0,1/sqrt(tempsigma[2,2])))
            tempomega <- (transmat %*% tempsigma) %*% transmat

            temprho <- tempomega[1,2]/(sqrt(tempomega[1,1]))

            # gamma1 <- tempomega[1,2]
            # phi1 <- tempomega[1,1] - (gamma1^2)

            gammatilde <- tempomega[1,2]
            phitilde <- tempomega[1,1] - (gammatilde^2)

          }else{

            phitilde <- 1/rgamma(n = 1, shape =  nzero/2, rate = S0/2)

            gammatilde <- rnorm(1,
                                mean = gamma0,
                                sd = sqrt(tau*phitilde))

          }



          if( (is.na(gammatilde)) | !is.finite(gammatilde)   ){
            print("gammatilde = ")
            print(gammatilde)

            print("nzero = ")
            print(nzero)


            print("S0 = ")
            print(S0)


            print("gamma0 = ")
            print(gamma0)
            print("tau = ")
            print(tau)
            print("phitilde = ")
            print(phitilde)

            stop("gamma tilde stop")
          }

          varthetamat_test[i,] <- c(mutilde, phitilde, gammatilde)


        }else{
          varthetamat_test[i,] <- varthetamat[temp_ind,]

          if( any(is.na(gammatilde)) | any(!is.finite(gammatilde))   ){

            print("varthetamat_test = ")
            print(varthetamat_test)

            print("i = ")
            print(i)

            print("temp_ind = ")
            print(temp_ind)

          }


        }

      } # end of for-loop of length ntest


    }




    # #########  set parameters for gamma draw  ######################################################
    #
    # if(vh_prior == TRUE){
    #   G0 <- tau*phi1
    # }
    #
    #
    # # G1inv <- (1/G0) + (1/phi1)*crossprod(z_epsilon)
    # G1inv <- (1/G0) + (1/phi1)*crossprod(z_epsilon[uncens_inds])
    # G1 <- (1/G1inv)[1,1]
    #
    # # gamma_one <- (G1*( (1/G0)*gamma0 + (1/phi1)*crossprod(z_epsilon , y_epsilon   )   ))[1,1]
    # gamma_one <- (G1*( (1/G0)*gamma0 + (1/phi1)*crossprod(z_epsilon[uncens_inds] , y_epsilon[uncens_inds]   )   ))[1,1]
    #
    # # print("phi1 = ")
    # # print(phi1)
    # # print("G0 = ")
    # # print(G0)
    # #
    # # print("(G1*( (1/G0)*gamma0 + (1/phi1)*crossprod(z_epsilon[uncens_inds] , y_epsilon[uncens_inds]   )   )) = ")
    # # print((G1*( (1/G0)*gamma0 + (1/phi1)*crossprod(z_epsilon[uncens_inds] , y_epsilon[uncens_inds]   )   )))
    # #
    # # print("gamma_one = ")
    # # print(gamma_one)
    # #
    # # print("crossprod(z_epsilon , y_epsilon   ) = ")
    # # print(crossprod(z_epsilon , y_epsilon   ))
    # #
    # # print("crossprod(z_epsilon    ) = ")
    # # print(crossprod(z_epsilon   ))
    # #
    # # print("z_epsilon     = ")
    # # print(z_epsilon   )
    # #
    # # print("G1 = ")
    # # print(G1)
    #
    # gamma1 <- rnorm(n = 1, mean = gamma_one, sd =  sqrt(G1) )
    #
    # #########  set parameters for phi draw  ######################################################
    #
    # n_one <- nzero + n1 + 1
    #
    # # print("S0 = ")
    # # print(S0)
    # # print("(gamma1^2)*crossprod(z_epsilon) = ")
    # # print((gamma1^2)*crossprod(z_epsilon))
    # #
    # # print("2*gamma1*crossprod(z_epsilon , y_epsilon   ) = ")
    # # print(2*gamma1*crossprod(z_epsilon , y_epsilon   ))
    # #
    # # print("crossprod(y_epsilon) = ")
    # # print(crossprod(y_epsilon))
    #
    # # S1 <- S0 + (gamma1^2)*crossprod(z_epsilon) - 2*gamma1*crossprod(z_epsilon , y_epsilon   )  + crossprod(y_epsilon)
    #
    # S1 <- S0 + (gamma1^2)/G0 + gamma1*crossprod( y_epsilon[uncens_inds] - gamma1*z_epsilon[uncens_inds]  )  + crossprod(y_epsilon)
    #
    # if(vh_prior == TRUE){
    #   S1 <- S0 + (gamma1^2)/tau + gamma1*crossprod( y_epsilon[uncens_inds] - gamma1*z_epsilon[uncens_inds]  )  + crossprod(y_epsilon)
    # }
    # # print("S1 = ")
    # # print(S1)
    # # print("n_one = ")
    # # print(n_one)
    #
    # # draw from inverse gamma
    # phi1 <- 1/rgamma(n = 1, shape =  n_one/2, rate = S1/2)
    #
    #
    #
    #
    #
    # # print("gamma1 = ")
    # # print(gamma1)
    #
    #
    # ######### update Sigma matrix #####################################################
    #
    # Sigma_mat <- cbind(c(1,gamma1),c(gamma1,phi1+gamma1^2))
    #
    #
    #
    #
    # ###### Accelerated sampler  ###############################
    #
    #
    # if(accelerate){
    #
    #
    #   #if prior mean for mu parameters is zero (does this make sense? require an offset for y?)
    #
    #   nu1 <- sum(sampler_z$getTrees@.Data()$var ==-1) - nzero + 1
    #
    #
    #   asquared <- (1/phi1)*(S0 + crossprod(y_epsilon[uncens_inds]))
    #
    #   znodestemp <- sampler_z$getTrees@.Data()$value[sampler_z$getTrees@.Data()$var!=-1]
    #
    #   bsquared <- (1 + (gamma1^2)/phi1)*crossprod(z_epsilon) +
    #     (1/sigmu_z)*crossprod(znodestemp) + (gamma1^2)*(1/G0)
    #
    #
    #   if(sqrt(asquared*bsquared) > 150^2){
    #     print("GIG sample will be slow.")
    #   }
    #
    #   #candidate g parameer value
    #   gprime <- rgig(n = 1, lambda = nu1/2, chi = asquared, psi = bsquared )
    #
    #
    #
    #
    #   probaccept <- min(1, exp((gprime-1)* ((1/sigmu_z)*sum(znodestemp)*meanmu_z      +
    #                                           gamma1*gamma0/G0 )    ) )
    #
    #   g_accepted <- 1
    #
    #   #check if accept
    #   accept_bin <- rbinom(n = 1,size = 1, prob = probaccept)
    #
    #   if(is.na(accept_bin)){
    #
    #     print("accept_bin is na.  probaccept =")
    #     print(probaccept)
    #
    #     print("gprime = ")
    #     print(gprime)
    #
    #     print("sigmu_z = ")
    #     print(sigmu_z)
    #
    #     print("sum(znodestemp) = ")
    #     print(sum(znodestemp))
    #
    #     print("meanmu_z = ")
    #     print(meanmu_z)
    #
    #   }
    #
    #   if(accept_bin ==1){
    #     g_accepted <- gprime
    #
    #     phi1 <- (gprime^2)*phi1
    #
    #     gamma1 <- gprime*gamma1
    #
    #     mutemp_z <- gprime*mutemp_z
    #
    #     z <- gprime*z
    #
    #   }
    #
    #
    #
    #
    # }






    ###### Store results   ###############################






    if(iter>n.burnin){
      iter_min_burnin <- iter-n.burnin



      if(selection_test ==1){
        utrain <- matrix(NA, nrow = n, ncol = 2)

        for(i in 1:n){

          Sigma_mattemp <- cbind(c(1,
                                   varthetamat[i,4]),
                                 c(varthetamat[i,4],
                                   varthetamat[i,3]+varthetamat[i,4]^2))


          utrain[i,] <- Rfast::rmvnorm(n = 1,
                                       mu = c(varthetamat[i,1],
                                              varthetamat[i,2]),
                                       sigma = Sigma_mattemp)
        }



        draw$error_draws[,iter_min_burnin,] <- utrain

        # draw correlations
        draw$pearsoncorr_draws[iter_min_burnin] <- cor(utrain[,1],utrain[,2])
        draw$kendalltau_draws[iter_min_burnin] <- cor(utrain[,1],utrain[,2], method = "kendall") #array(NA, dim = c(n, n.iter))
        draw$spearmanrho_draws[iter_min_burnin] <-  cor(utrain[,1],utrain[,2], method = "spearman") #array(NA, dim = c(n, n.iter))


        # for(i in 1:ntest){

        # varthetamat_test[i,] <- cbind(mutilde, phitilde, gammatilde)

        #can use first element, just need one draw from posterior predictive

        Sigma_mattemp <- cbind(c(1,
                                 varthetamat_test[1,4]),
                               c(varthetamat_test[1,4],
                                 varthetamat_test[1,3]+varthetamat_test[1,4]^2))


        draw$error_draws_test[iter_min_burnin,] <- Rfast::rmvnorm(n = 1,
                                                                  mu = c(varthetamat_test[1,1],
                                                                         varthetamat_test[1,2]),
                                                                  sigma = Sigma_mattemp)
        # }





      }




      #NOTE y and z training sample values saved here
      #do not correspond to the the same means and errors as
      #the test values and expectations saved here.
      #However, they are the values to which the trees in this round were fitted.

      #draw z and y for test observations
      zytest <- matrix(NA, nrow = ntest, ncol = 2)

      # zytest <- mvrnorm(n = ntest,
      #                   mu = c(0, 0),
      #                   Sigma = Sigma_mat)

      # print("length(mutemp_test_z) = ")
      # print(length(mutemp_test_z))
      #
      # print("offsetz = ")
      # print(offsetz)
      #
      # print("length(zytest[,1]) = ")
      # print(length(zytest[,1]))

      # zytest[,1] <- zytest[,1] + offsetz + mutemp_test_z
      # zytest[,2] <- zytest[,2] + mutemp_test_y

      for(i in 1:ntest){

        # varthetamat_test[i,] <- cbind(mutilde, phitilde, gammatilde)


        Sigma_mattemp <- cbind(c(1,
                                 varthetamat_test[i,4]),
                               c(varthetamat_test[i,4],
                                 varthetamat_test[i,3]+varthetamat_test[i,4]^2))


        zytest[i,] <- Rfast::rmvnorm(n = 1,
                                     mu = c(offsetz + mutemp_test_z[i] + varthetamat_test[i,1],
                                            mutemp_test_y[i]+ varthetamat_test[i,2]),
                                     sigma = Sigma_mattemp)
      }


      probcens_train <- pnorm(- mutemp_z[uncens_inds] - offsetz - varthetamat[uncens_inds,1] )
      probcens_test <- pnorm(- mutemp_test_z - offsetz- varthetamat_test[,1])

      #calculate conditional expectation

      # condexptrain <- mutemp_y + gamma1*(dnorm(- mutemp_z - offsetz ))/(1-probcens_train)

      temp_ztrain <- mutemp_z[uncens_inds] + offsetz + varthetamat[uncens_inds,1]

      IMR_train <- exp( dnorm(temp_ztrain,log=T) - pnorm(temp_ztrain,log.p = T) )


      condexptrain <- mutemp_y + varthetamat[uncens_inds,2] + varthetamat[uncens_inds,4]*IMR_train#*
      # (dnorm(- mutemp_z[uncens_inds] - offsetz - varthetamat[uncens_inds,1]))/(1-probcens_train)

      # (dnorm(- mutemp_test_z - offsetz - varthetamat_test[,1] ))/(1-probcens_test)


      temp_ztest <- mutemp_test_z + offsetz + varthetamat_test[,1]

      IMR_test <- exp( dnorm(temp_ztest,log=T) - pnorm(temp_ztest,log.p = T) )

      condexptest <- mutemp_test_y + varthetamat_test[,2] +
        varthetamat_test[,4]*IMR_test#(dnorm(- mutemp_test_z - offsetz - varthetamat_test[,1] ))/(1-probcens_test)

      # condexptest <- ifelse(probcens_test ==1,
      #                       mutemp_test_y + varthetamat_test[,2],
      #                       mutemp_test_y + varthetamat_test[,2] +
      #                         varthetamat_test[,4]*(dnorm(- mutemp_test_z - offsetz - varthetamat_test[,1] ))/(1-probcens_test)
      #                       )

      if( any(is.na(condexptest)) | any(!is.finite(condexptest))   ){

        print("condexptest = ")
        print(condexptest)

        print("mutemp_test_y = ")
        print(mutemp_test_y)

        print("mutemp_test_z = ")
        print(mutemp_test_z)

        print(" varthetamat_test[,4] = ")
        print( varthetamat_test[,4])

        print("which(is.na(condexptest)) = ")
        print(which(is.na(condexptest)))


        print("which(!is.finite(condexptest)) = ")
        print(which(!is.finite(condexptest)))

        err_inds <- which(!is.finite(condexptest))

        print("offsetz= ")
        print(offsetz)

        print("condexptest[err_inds] = ")
        print(condexptest[err_inds])

        print("mutemp_test_y[err_inds] = ")
        print(mutemp_test_y[err_inds])

        print("mutemp_test_z[err_inds] = ")
        print(mutemp_test_z[err_inds])

        print(" varthetamat_test[err_inds,] = ")
        print( varthetamat_test[err_inds,])

        print(" probcens_test[err_inds] = ")
        print( probcens_test[err_inds])



        stop("any(is.na(condexptest)) | any(!is.finite(condexptest))")

      }



      # draw$Z.mat_train[,iter_min_burnin] <- z
      # draw$Z.mat_test[,iter_min_burnin] <-  zytest[,1]
      # draw$Y.mat_train = array(NA, dim = c(n, n.iter)),
      # draw$Y.mat_test = array(NA, dim = c(ntest, n.iter)),
      draw$mu_y_train[, iter_min_burnin] <- mutemp_y + varthetamat[uncens_inds,2]
      draw$mu_y_test[, iter_min_burnin] <- mutemp_test_y + varthetamat_test[,2]

      draw$mu_y_train_noerror[, iter_min_burnin] <- mutemp_y #+ varthetamat[uncens_inds,2]
      draw$mu_y_test_noerror[, iter_min_burnin] <- mutemp_test_y #+ varthetamat_test[,2]


      # draw$mucens_y_train[, iter_min_burnin] <- mutemp_y[cens_inds]
      # draw$muuncens_y_train[, iter_min_burnin] <- mutemp_y[uncens_inds]
      draw$muuncens_y_train[, iter_min_burnin] <- mutemp_y+ varthetamat[uncens_inds,2]

      draw$mu_z_train[, iter_min_burnin] <- mutemp_z + offsetz + varthetamat[,1]
      draw$mu_z_test[, iter_min_burnin] <- mutemp_test_z + offsetz + varthetamat_test[,1]

      draw$train.probcens[, iter_min_burnin] <-  probcens_train
      draw$test.probcens[, iter_min_burnin] <-  probcens_test

      draw$cond_exp_train[, iter_min_burnin] <- condexptrain
      draw$cond_exp_test[, iter_min_burnin] <- condexptest

      draw$ystar_train[, iter_min_burnin] <- ystar
      draw$ystar_test[, iter_min_burnin] <- zytest[,2]
      draw$zstar_train[,iter_min_burnin] <- z
      draw$zstar_test[,iter_min_burnin] <-  zytest[,1]

      draw$ycond_draws_train[[iter_min_burnin]] <-  ystar[z >=0]
      draw$ycond_draws_test[[iter_min_burnin]] <-  zytest[,2][zytest[,1] >= 0]

      draw$vartheta_draws[,, iter_min_burnin] <- varthetamat
      draw$vartheta_test_draws[,, iter_min_burnin] <- varthetamat_test


      if(is.numeric(censored_value)){

        # uncondexptrain <- censored_value*probcens_train +  mutemp_y*(1- probcens_train ) + gamma1*dnorm(- mutemp_z - offsetz )
        uncondexptrain <- censored_value*probcens_train +  mutemp_y*(1- probcens_train ) + varthetamat[uncens_inds,4]*dnorm(- mutemp_z[uncens_inds] - offsetz - varthetamat[uncens_inds,1] )
        uncondexptest <- censored_value*probcens_test +  mutemp_test_y*(1- probcens_test ) + varthetamat_test[,4]*dnorm(- mutemp_test_z - offsetz -  varthetamat_test[,1])

        draw$uncond_exp_train[, iter_min_burnin] <- uncondexptrain
        draw$uncond_exp_test[, iter_min_burnin] <- uncondexptest


        # draw$ydraws_train[, iter_min_burnin] <- ifelse(z < 0, censored_value, ystar )
        draw$ydraws_test[, iter_min_burnin] <- ifelse(zytest[,1] < 0, censored_value, zytest[,2] )
      }

      draw$alpha[iter_min_burnin] <- alpha


    } # end if iter > burnin

    if(iter %% print.opt == 0){
      print(paste("Gibbs Iteration", iter))
      # print(c(sigma2.alpha, sigma2.beta))
    }


  }#end iterations of Giibs sampler



  return(draw)



}

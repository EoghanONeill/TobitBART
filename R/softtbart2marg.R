
#' @title Type II Tobit Soft Bayesian Additive Regression Trees implemented using MCMC and marginalization of all terminal node parameters for tree sampling
#'
#' @description Type II Tobit Soft Bayesian Additive Regression Trees implemented using MCMC and marginalization of all terminal node parameters for tree sampling
#' @import dbarts
#' @import truncnorm
#' @import MASS
#' @import GIGrvg
#' @import mvtnorm
#' @import sampleSelection
#' @import progress
#' @import LaplacesDemon
#' @param x.train The outcome model training covariate data for all training observations. Number of rows equal to the number of observations. Number of columns equal to the number of covariates.
#' @param x.test The outcome model test covariate data for all test observations. Number of rows equal to the number of observations. Number of columns equal to the number of covariates.
#' @param w.train The censoring model training covariate data for all training observations. Number of rows equal to the number of observations. Number of columns equal to the number of covariates.
#' @param w.test The censoring model test covariate data for all test observations. Number of rows equal to the number of observations. Number of columns equal to the number of covariates.
#' @param y The training data vector of outcomes. A continuous, censored outcome variable. Censored observations must be included with values equal to censored_value
#' @param n.iter Number of iterations excluding burnin.
#' @param n.burnin Number of burnin iterations.
#' @param censored_value The value taken by censored observations
#' @param gamma0 The mean of the normal prior on the covariance of the errors in the censoring and outcome models.
#' @param G0 The variance of the normal prior on the covariance of the errors in the censoring and outcome models (only if cov_prior equals Omori).
#' @param nzero A prior parameter which when divided by 2 gives the mean of the normal prior on phi, where phi*gamma is the variance of the errors of the outcome model.
#' @param S0 A prior parameter which when divided by 2 gives the variance of the normal prior on phi, where phi*gamma is the variance of the errors of the outcome model.
#' @param sigest Estimate of variance of hte error term.
#' @param n.trees_outcome (dbarts control option) A positive integer giving the number of trees used in the outcome model sum-of-trees formulation.
#' @param n.trees_censoring (dbarts control option) A positive integer giving the number of trees used in the censoring model sum-of-trees formulation.
#' @param tree_power_y Tree prior parameter for outcome model.
#' @param tree_base_y Tree prior parameter for outcome model.
#' @param tree_power_z Tree prior parameter for selection model.
#' @param tree_base_z Tree prior parameter for selection model.
#' @param k_z Hyperparameter for calibration of sigma2_mu_z.
#' @param k_y Hyperparameter for calibration of sigma2_mu_y.
#' @param node.prior (dbarts option) An expression of the form dbarts:::normal or dbarts:::normal(k) that sets the prior used on the averages within nodes.
#' @param resid.prior (dbarts option) An expression of the form dbarts:::chisq or dbarts:::chisq(df,quant) that sets the prior used on the residual/error variance
#' @param proposal.probs (dbarts option) Named numeric vector or NULL, optionally specifying the proposal rules and their probabilities. Elements should be "birth_death", "change", and "swap" to control tree change proposals, and "birth" to give the relative frequency of birth/death in the "birth_death" step.
#' @param sigmadbarts (dbarts option) A positive numeric estimate of the residual standard deviation. If NA, a linear model is used with all of the predictors to obtain one.
#' @param print.opt Print every print.opt number of Gibbs samples.
#' @param eq_by_eq If TRUE, implements sampler equation by equation (as in BAVART by Huber and Rossini (2021)). If FALSE, implements sampler in similar approach to SUR-BART (Chakraborty 2016) or MPBART (Kindo 2016).
#' @param accelerate If TRUE, add extra parameter for accelerated sampler as descibed by Omori (2007).
#' @param cov_prior Prior for the covariance of the error terms. If VH, apply the prior of van Hasselt (2011), N(gamma0, tau*phi), imposing dependence between gamma and phi. If Omori, apply the prior N(gamma0,G0). If mixture, then a mixture of the VH and Omori priors with probability mixprob applied to the VH prior.
#' @param mixprob If cov_prior equals Mixture, then mixprob is the probability applied to the Van Hasselt covariance prior, and one minus mixprob is the probability applied to the Omori prior.
#' @param tau Parameter for the prior of van Hasselt (2011) on the covariance of the error terms.
#' @param simultaneous_covmat If TRUE, jointly sample the parameters that determine the covariance matrix instead of sampling from separate full conditionals.
#' @param fast If equal to true, takes faster samples of z and y and makes faster approximate calculations of selection probabilities.
#' @param nu0 For the inverseWishart prior Winv(nu0,c*I_2) of Ding (2014) on an unidentified unrestricted covariance matrix. nu = 3 corresponds to a uniform correlation prior. nu > 3 centers the correlation prior at 0, while nu < 3 places more prior probability on selection on unobservables.
#' @param quantsig For the inverseWishart prior Winv(nu0,c*I_2) of Ding (2014). The parameter c is determined by quantsig so that the marginal prior on the standard deviation of the outcome has 90% quantile equal to an estimate from a linear tobit model or sigest if sigest is not NA.
#' @param sparse If equal to TRUE, use Linero Dirichlet prior on splitting probabilities
#' @param alpha_a_y Linero alpha prior parameter for outcome equation splitting probabilities
#' @param alpha_b_y Linero alpha prior parameter for outcome equation splitting probabilities
#' @param alpha_a_z Linero alpha prior parameter for selection equation splitting probabilities
#' @param alpha_b_z Linero alpha prior parameter for selection equation splitting probabilities
#' @param alpha_split_prior If true, set hyperprior for Linero alpha parameter
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
#' \item{ystar_test}{Matrix of test sample draws of the outcome assuming uncensored . Number of rows equals number of test observations. Number of columns equals n.iter .}
#' \item{zstar_train}{Matrix of training sample draws of the censoring model latent outcome. Number of rows equals number of training observations. Number of columns equals n.iter.}
#' \item{zstar_test}{Matrix of test sample draws of the censoring model latent outcome. Number of rows equals number of test observations. Number of columns equals n.iter.}
#' \item{ydraws_train}{Only defined if censored_value is a number. Matrix of training sample unconditional (i.e. possibly censored) draws of the outcome. Number of rows equals number of training observations. Number of columns equals n.iter .}
#' \item{ydraws_test}{Only defined if censored_value is a number. Matrix of test sample unconditional (i.e. possibly censored) draws of the outcome . Number of rows equals number of test observations. Number of columns equals n.iter .}
#' \item{ycond_draws_test}{List of test sample conditional (i.e. zstar >0 for draw) draws of the outcome . Number of rows equals number of test observations. Number of columns equals n.iter .}
#' \item{Sigma_draws}{3 dimensional array of MCMC draws of the covariance matrix for the censoring and outcome error terms. The numbers of rows and columns equal are equal to 2. The first row and column correspond to the censoring model. The second row and column correspond to the outcome model. The number of slices equals n.iter . }
#' \item{alpha_s_y_store}{For Dirichlet prior on splitting probabilities in outcome equation, vector of alpha hyperparameter draws for each iteration.}
#' \item{alpha_s_z_store}{For Dirichlet prior on splitting probabilities in selection equation, vector of alpha hyperparameter draws for each iteration }
#' \item{var_count_y_store}{Matrix of counts of splits on each variable in outcome observation. The number of rows is the number of potential splitting variables. The number of columns is the number of post-burn-in iterations.}
#' \item{var_count_z_store}{Matrix of counts of splits on each variable in selection observation. The number of rows is the number of potential splitting variables. The number of columns is the number of post-burn-in iterations. }
#' \item{s_prob_y_store}{Splitting probabilities for the outcome equation. The number of rows is the number of potential splitting variables. The number of columns is the number of post-burn-in iterations. }
#' \item{s_prob_z_store}{Splitting probabilities for the selection equation. The number of rows is the number of potential splitting variables. The number of columns is the number of post-burn-in iterations. }
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
#' xmat_train <- matrix(NA,nrow = num_train,
#'                      ncol = 8)
#'
#' xmat_train[,1] <- runif(num_train, min = -1, max = 1)
#' xmat_train[,2] <- rf(num_train,20,20)
#' xmat_train[,3] <- rbinom(num_train, size = 1, prob = 0.75)
#' xmat_train[,4] <- rnorm(num_train, mean = 1, sd = 1)
#' xmat_train[,5] <- rnorm(num_train)
#' xmat_train[,6] <- rbinom(num_train, size = 1, prob = 0.5)
#' xmat_train[,7] <- rf(num_train,20,200)
#' xmat_train[,8] <- runif(num_train, min = 0, max = 2)
#'
#' #it would be better to test performance of the models when there is correlation in the error terms.
#' varepsilon1_train <- rnorm(num_train, mean = 0, sd = sqrt(0.00025))
#' varepsilon2_train <- rnorm(num_train, mean = 0, sd = sqrt(0.00025))
#'
#' y1star_train <- 1 - 0.75*xmat_train[,1] + 0.75*xmat_train[,2] -
#'   0.5*xmat_train[,4] -  0.5*xmat_train[,6] - 0.25*xmat_train[,1]^2 -
#'   0.75*xmat_train[,1]*xmat_train[,4] - 0.25*xmat_train[,1]*xmat_train[,2] -
#'   1*xmat_train[,1]*xmat_train[,6] + 0.5*xmat_train[,2]*xmat_train[,6] +
#'   varepsilon1_train
#'
#' y2star_train <- 1 + 0.25*xmat_train[,4] - 0.75*xmat_train[,6] +
#'   0.5*xmat_train[,7] + 0.25*xmat_train[,8] +
#'   0.25*xmat_train[,4]^2 + 0.75*xmat_train[,7]^2 + 0.5*xmat_train[,8]^2 -
#'   1*xmat_train[,4]*xmat_train[,6] + 0.5*xmat_train[,4]*xmat_train[,8] +
#'   1*xmat_train[,6]*xmat_train[,7] - 0.25*xmat_train[,7]*xmat_train[,8] +
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
#' tbartII_example <- tbart2c(xmat_train,
#'                            Xmat_test,
#'                            xmat_train,
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

softtbart2marg <- function(x.train,
                       x.test,
                       w.train,
                       w.test,
                       y,
                       n.iter=1000,
                       n.burnin=100,
                       censored_value = NA,
                       gamma0 = 0,
                       G0=1,
                       nzero = 6,#0.002,
                       S0= 12,#0.002,
                       sigest = NA,
                       n.trees_outcome = 50L,
                       n.trees_censoring = 50L,
                       trans_prob = c(2.5, 2.5, 4) / 9, # Probabilities to grow, prune or change, respectively
                       max_bad_trees = 10,
                       tree_power_z = 2,
                       tree_power_y = 2,
                       tree_base_z = 0.95,
                       tree_base_y = 0.95,
                       k_z = 2,
                       k_y = 2,
                       alpha_z = 0.95,
                       beta_z = 2,
                       alpha_y = 0.95,
                       beta_y = 2,
                       node.prior = dbarts:::normal,
                       resid.prior = dbarts:::chisq,
                       proposal.probs = c(birth_death = 0.5, swap = 0, change = 0.5, birth = 0.5),
                       sigmadbarts = NA_real_,
                       print.opt = 100,
                       eq_by_eq = TRUE,
                       # accelerate = FALSE,
                       cov_prior = "Ding",
                       tau = 1/3,
                       mixprob = 0.5,
                       simultaneous_covmat = TRUE,
                       fast = TRUE,
                       nu0 = 3, offsetz = FALSE,
                       quantsig = 0.9,
                       sparse = FALSE,
                       alpha_a_y = 0.5,
                       alpha_b_y = 1,
                       alpha_a_z = 0.5,
                       alpha_b_z = 1,
                       alpha_split_prior = TRUE,
                       sigma_mu_prior = FALSE, sigma_mu_dist = "Cauchy",
                       node_min_size = 5,
                       centre_y = TRUE,
                       splitting_rules = "continuous",
                       marginalize = TRUE,
                       one_chol = FALSE,
                       tau_hyperprior = TRUE, alpha_tau = 1, beta_tau = 10,
                       jointgammanodes = FALSE, linearterms = FALSE, jointbetagamma = FALSE,
                       mh_tau_bandwidth = TRUE,
                       tau_rate = 10){


  if(marginalize == FALSE){
    stop("Code does not currently support marginalize == FALSE. Use softtbart2 instead.")
  }

  if(one_chol){
    stop("This function does not currently cupport one_chol == TRUE")
  }

  # if(jointbetagamma & jointbetagamma){
  #   stop("Can't have both jointbetagamma & jointbetagamma ")
  # }

  if((marginalize == FALSE) & (jointgammanodes == TRUE)){
    stop("Currently cannot jointly sample gamma with terminal node parameters without marginalizing out these parameter in outcome tree draws")
  }

  # if((marginalize == FALSE) & (linearterms == TRUE)){
  #   stop("Currently code only written for marginalized model with linear terms. Use tbart2pluslinear for non-marginalized linear function plus trees.")
  # }

  if(!(splitting_rules %in% c("discrete", "continuous"))){
    stop("splitting_rules must be 'discrete' or 'continuous'.")
  }

  if(!(cov_prior %in% c("VH","Omori","Mixture", "Ding"))){
    stop("cov_prior must equal VH, Omori, Mixture, or Ding")
  }
  if((mixprob < 0)| (mixprob > 1) ){
    stop("mixprob must be between zero and one.")
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
    cens_inds <- which(is.na(y))
    uncens_inds <- which(!(is.na(y)))
  }else{
    cens_inds <- which(y == censored_value)
    uncens_inds <- which(y != censored_value)
  }

  if(length(cens_inds)==0) stop("No censored observations")




  ######### set up things for myBART implementation ####################

  # Extract control parameters
  # we only have to allow for empty nodes when updating Z (and therefore Zlag is updated and splits on Zlag are affected)
  # Therefore there is still a minimum node size criterion for the purpose of proposing new splits
  node_min_size = node_min_size

  # Storage containers
  store_size = n.iter # npost # code currently written to save all output, so no nburnin or npost
  tree_store = vector('list', store_size)
  sigma2_store = rep(NA, store_size)
  # y_hat_store = matrix(NA, ncol = length(y), nrow = store_size)
  # var_count = rep(0, ncol(x))
  # var_count_store = matrix(0, ncol = ncol(x), nrow = store_size)
  # s_prob_store = matrix(0, ncol = ncol(x), nrow = store_size)
  # tree_fits_store = matrix(0, ncol = n.trees, nrow = length(y))







  # normalize the outcome
  tempmean <- mean(y[uncens_inds])
  tempsd <- sd(y[uncens_inds])
  originaly <- y
  y <- (y-tempmean)/tempsd

  if(is.numeric(censored_value)){
    censored_value <- (censored_value- tempmean)/tempsd
  }



  ecdfsx   <- list()
  for(i in 1:ncol(x.train)) {
    ecdfsx[[i]] <- ecdf(x.train[,i])
    if(length(unique(x.train[,i])) == 1) ecdfsx[[i]] <- identity
    if(length(unique(x.train[,i])) == 2) ecdfsx[[i]] <- make_01_norm(x.train[,i])
  }
  for(i in 1:ncol(x.train)) {
    x.train[,i] <- ecdfsx[[i]](x.train[,i])
    x.test[,i] <- ecdfsx[[i]](x.test[,i])
  }

  rm(ecdfsx)

  ecdfsw   <- list()
  for(i in 1:ncol(w.train)) {
    ecdfsw[[i]] <- ecdf(w.train[,i])
    if(length(unique(w.train[,i])) == 1) ecdfsw[[i]] <- identity
    if(length(unique(w.train[,i])) == 2) ecdfsw[[i]] <- make_01_norm(w.train[,i])
  }
  for(i in 1:ncol(w.train)) {
    w.train[,i] <- ecdfsw[[i]](w.train[,i])
    w.test[,i] <- ecdfsw[[i]](w.test[,i])
  }

  rm(ecdfsw)






  # y_min <- min(y_scale)
  # y_max <- max(y_scale)

  if(centre_y){
    y_max <- max(y[uncens_inds])
    y_min <- min(y[uncens_inds])
  }else{
    y_max <- 0
    y_min <- 0
  }

  # Other variables
  # sigma2 <- 1                          # !!!!!!!!!!!!!!
  # mu_mu <- 0 # (y_min + y_max) / (2 * m)

  y_scale <- y - (y_max + y_min)/2
  # tau=(max(y.train)-min(y.train))/(2*k*sqrt(ntree))
  if(is.numeric(censored_value)){
    censored_value <- censored_value - (y_max + y_min)/2
  }
  # sigma2_mu <- (max(y_scale)-min(y_scale))/((2 * k * sqrt(m))^2)
  # sigma2_mu <- ((max(y_scale)-min(y_scale))/(2 * k* sqrt(m)))^2


  # sigma2_mu <- ((y_max - y_min) / (2 * k * sqrt(m)))^2



  #create z vector

  #create ystar vector
  ystar <- rep(mean(y_scale[uncens_inds]), length(y_scale)) # this line is unimportant because only uncensored values are used
  ystar[uncens_inds] <- y_scale[uncens_inds]




  n <- length(y)
  n0 <- length(cens_inds)
  n1 <- length(uncens_inds)

  ntest = nrow(x.test)

  p_y <- ncol(x.train)
  p_z <- ncol(w.train)
  s_y <- rep(1 / p_y, p_y) # probability vector to be used during the growing process for DART feature weighting
  s_z <- rep(1 / p_z, p_z) # probability vector to be used during the growing process for DART feature weighting


  if(linearterms){
    xmat_train <- cbind(1, x.train[uncens_inds, ,drop = FALSE])
    wmat_train <- cbind(1, w.train)
    xmat_test <- cbind(1, x.test)
    wmat_test <- cbind(1, w.test)
    Amean_p <- rep(0, p_z + 1)
    Bmean_p <- rep(0, p_y + 1)

    Avar_p <- 10*diag(p_z + 1)
    Bvar_p <- 10*diag(p_y + 1)
    invAvar_p <- diag(p_z + 1)/10
    invBvar_p <- diag(p_y + 1)/10

    alpha_vec <- rep(0, p_z+1)
    beta_vec <- rep(0, p_y+1)
  }


  if(sparse){
    rho_y <- p_y # For DART

    if(alpha_split_prior){
      alpha_s_y <- p_y
    }else{
      alpha_s_y <- 1
    }
    alpha_scale_y <- p_y


    rho_z <- p_z # For DART

    if(alpha_split_prior){
      alpha_s_z <- p_z
    }else{
      alpha_s_z <- 1
    }
    alpha_scale_z <- p_z

  }


  if(!marginalize){
    tree_fits_store_z = matrix(0, ncol = n.trees_censoring, nrow = n)
    tree_fits_store_y = matrix(0, ncol = n.trees_outcome, nrow = n1)
  }



  if(offsetz){
    offsetz <- qnorm(n1/n)
  }else{
    offsetz <- 0 # qnorm(n1/n)
  }
  z <- rep(offsetz, length(y))

  # z[cens_inds] <- qnorm(0.001) #rtruncnorm(n0, a= -Inf, b = 0, mean = offsetz, sd = 1)
  #
  # z[uncens_inds] <- qnorm(0.999) #rtruncnorm(n1, a= 0, b = Inf, mean = offsetz, sd = 1)

  z[cens_inds] <- rtruncnorm(n0, a= -Inf, b = 0, mean = offsetz, sd = 1)
  z[uncens_inds] <- rtruncnorm(n1, a= 0, b = Inf, mean = offsetz, sd = 1)


  # z <- rnorm(n = length(y), mean = offsetz, sd =1)



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

  #use uncensored observations
  if(is.na(sigest)) {
    if(ncol(x.train) < n1) {
      # df = data.frame(x = x.train[uncens_inds,],y = y[uncens_inds])
      # lmf = lm(y~.,df)
      # sigest = summary(lmf)$sigma
      if(is.na(censored_value)){
        dtemp <-  1*(!(is.na(y_scale)))
      }else{
        dtemp <-  1*(y_scale != censored_value)
      }

      df = data.frame(x = cbind(x.train,w.train), y = y_scale, d = dtemp )

      # print("x.train = ")
      # print(x.train)
      # print("colnames(df) = ")
      # print(colnames(df))

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
      gamma0 <- correst*sigest

    } else {
      sigest = sd(y_scale[uncens_inds])
      correst <- 0
      gamma0 <- 0
    }
  }else{
    correst <- 0
    gamma0 <- 0
  }

  # if(is.null(nzero)){
  #
  #   nzero <- 2*(sigest^2)
  #
  # }

  #set initial sigma

  #alternatively, draw this from the prior
  Sigma_mat <- cbind(c(1,0),c(0,sigest^2))

  #set initial gamma
  gamma1 <- 0 # correst*sigest  #0#cov(ystar,z)
  # gamma1 <- 0
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


  # if(sigest > G0){
  #   S0 <- (nzero - 2)*(sigest-G0)
  # }else{
  # sigquant <- 0.9
  # qchi <- qchisq(1.0-sigquant,nzero/2)
  # S0 <- 2*(sigest*sigest*qchi)/(nzero/2)
  # }


  # S0 <- (sigest^2)*(nzero-2)/(1+tau)

  S0 <- (sigest^2)*(1 - correst^2)*(nzero-2)/(1+tau)
  # S0 <- 0.5*(sigest^2 - gamma0^2)*(nzero-2)/(2+tau)

  print("S0 = ")
  print(S0)

  # # alternative: calibrate prior on phi as if gamma equals zero
  # qchi = qchisq(1.0-quantsig,nzero)
  # lambda = (sigest*sigest*qchi)/nzero #lambda parameter for sigma prior
  # S0 <- nzero*lambda
  # S0 <- 2*(sigest^2) * (nzero/2 - 1)

  print("2* sigest*(n0/2 - 1) = ")
  print(2* sigest*(n0/2 - 1))
  print("(sigest^2)*(1 - correst^2)*(nzero-2)/(1+tau) = ")
  print((sigest^2)*(1 - correst^2)*(nzero-2)/(1+tau))

  print("sigest = ")
  print(sigest)
  print("sigest^2 = ")
  print(sigest^2)

  print("correst = ")
  print(correst)

  print("S0 = ")
  print(S0)

  print("(tempsd^2)*sigest^2 = ")
  print((tempsd^2)*sigest^2)

  print("(tempsd^2)*prior mean outcome variance = ")
  print((tempsd^2)*S0*(1+tau)/(nzero-2) + gamma0^2)


  print("prior mean outcome variance = ")
  print(S0*(1+tau)/(nzero-2) + gamma0^2)

  # S0 <- 2
  # nzero <- 2

  if(cov_prior == "Ding"){
    gamma0 <- 0
    sigquant <- 0.9
    qchi <- qchisq(1.0-quantsig,nu0-1)
    cdivnu <- (sigest*sigest*qchi)/(nu0-1) #lambda parameter for sigma prior
    cding <- cdivnu*(nu0-1)

    print("cding old = ")
    print(cding)


    qig_noscale <- qgamma(p =  1- quantsig, shape = (nu0-1)/2, rate = 1/2)
    cding <- sigest*sigest*qig_noscale
    print("cding = ")
    print(cding)

    print("1/qgamma(p = 1- quantsig, shape = (nu0-1)/2, rate = cding/2) = ")
    print(1/qgamma(p = 1- quantsig, shape = (nu0-1)/2, rate = cding/2))

    print("sigest^2 = ")
    print(sigest^2)
    # rhoinit <- 0
    # siginit <- sigest
    Sigma_mat <- cbind(c(1,gamma1),c(gamma1,sigest^2))

  }

  if(cov_prior == "Omori"){
    # S0 <- (sigest^2 - G0*(1 + gamma0^2))*(nzero-2)
    # can be negative if G0 not chosen appropriately
    S0 <- (sigest^2)*(1 - correst^2)*(nzero-2)/(1+tau)
    G0 <- tau*S0/(nzero-2) # tau*E[phi]
    print("S0 = ")
    print(S0)
  }
  print("gamma0 = ")
  print(gamma0)

  print("tempsd*gamma0 = ")
  print(tempsd*gamma0)

  # gamma0 <- 0


  draw = list(
    # Z.mat_train = array(NA, dim = c(n, n.iter)),
    # Z.mat_test = array(NA, dim = c(ntest, n.iter)),
    # Y.mat_train = array(NA, dim = c(n, n.iter)),
    # Y.mat_test = array(NA, dim = c(ntest, n.iter)),
    mu_y_train = array(NA, dim = c(n1, n.iter)),# array(NA, dim = c(n, n.iter)),
    mu_y_test = array(NA, dim = c(ntest, n.iter)),
    # mucens_y_train = array(NA, dim = c(n0, n.iter)),
    muuncens_y_train = array(NA, dim = c(n1, n.iter)),
    mu_z_train = array(NA, dim = c(n, n.iter)),
    mu_z_test = array(NA, dim = c(ntest, n.iter)),
    train.probcens =  array(NA, dim = c(n1, n.iter)),#array(NA, dim = c(n, n.iter)),#,
    test.probcens =  array(NA, dim = c(ntest, n.iter)),#,
    cond_exp_train = array(NA, dim = c(n1, n.iter)),#cond_exp_train = array(NA, dim = c(n, n.iter)),
    cond_exp_test = array(NA, dim = c(ntest, n.iter)),
    # ystar_train = array(NA, dim = c(n, n.iter)),
    ystar_test = array(NA, dim = c(ntest, n.iter)),
    zstar_train = array(NA, dim = c(n, n.iter)),
    zstar_test = array(NA, dim = c(ntest, n.iter)),
    # ycond_draws_train = list(),
    ycond_draws_test = list(),
    Sigma_draws = array(NA, dim = c(2, 2, n.iter))
  )

  if(is.numeric(censored_value)){
    draw$uncond_exp_train <- array(NA, dim = c(n1, n.iter)) #array(NA, dim = c(n, n.iter))
    draw$uncond_exp_test <- array(NA, dim = c(ntest, n.iter))
    # draw$ydraws_train <- array(NA, dim = c(n, n.iter))
    draw$ydraws_test <- array(NA, dim = c(ntest, n.iter))
  }

  draw$var_count_y_store <- matrix(0, ncol = p_y, nrow = n.iter)
  draw$var_count_z_store <- matrix(0, ncol = p_z, nrow = n.iter)
  var_count_y <- rep(0, p_y)
  var_count_z <- rep(0, p_z)
  if(sparse){

    draw$alpha_s_y_store <- rep(NA, n.iter)
    draw$alpha_s_z_store <- rep(NA, n.iter)
    draw$s_prob_y_store <- matrix(0, ncol = p_y, nrow = n.iter)
    draw$s_prob_z_store <- matrix(0, ncol = p_z, nrow = n.iter)

  }


  tau_vec_censoring <- rep(1/tau_rate, n.trees_censoring)
  tau_vec_outcome <- rep(1/tau_rate, n.trees_outcome)



  ########## Initialize dbarts #####################

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


  weightstemp <- rep(1,n)

  weightstemp[uncens_inds] <- (gamma1^2 + phi1)/phi1

  # print("weightstemp = ")
  # print(weightstemp)
  #
  # print("length(weightstemp) = ")
  # print(length(weightstemp))
  #
  # print("length(uncens_inds) = ")
  # print(length(uncens_inds))

  # if(nrow(x.test ) == 0){
  #
  #
  #   xdf_y <- data.frame(y = ystar[uncens_inds], x = x.train[uncens_inds,])
  #   sampler_y <- dbarts(y ~ .,
  #                       data = xdf_y,
  #                       #test = x.test,
  #                       control = control_y,
  #                       tree.prior = dbarts:::cgm(power = tree_power_y, base =  tree_base_y,  split.probs = rep(1 / p_y, p_y)),
  #                       node.prior = node.prior,
  #                       resid.prior = resid.prior,
  #                       proposal.probs = proposal.probs,
  #                       sigma = sigmadbarts
  #   )
  #
  #   # print("Line 425")
  #
  #   xdf_z <- data.frame(y = z - offsetz, x = w.train)
  #
  #   sampler_z <- dbarts(y ~ .,
  #                       data = xdf_z,
  #                       #test = x.test,
  #                       weights = weightstemp,
  #                       control = control_z,
  #                       tree.prior = dbarts:::cgm(power = tree_power_z, base = tree_base_z,  split.probs = rep(1 / p_z, p_z)),
  #                       node.prior = node.prior,
  #                       resid.prior = resid.prior,
  #                       proposal.probs = proposal.probs,
  #                       sigma = 1
  #   )
  #
  # }else{
  #
  #   xdf_y <- data.frame(y = ystar[uncens_inds], x = x.train[uncens_inds,])
  #   xdf_y_test <- data.frame(x = x.test)
  #
  #   sampler_y <- dbarts(y ~ .,
  #                       data = xdf_y,
  #                       test = xdf_y_test,
  #                       control = control_y,
  #                       tree.prior = dbarts:::cgm(power = tree_power_y, base = tree_base_y,  split.probs = rep(1 / p_y, p_y)),
  #                       node.prior = node.prior,
  #                       resid.prior = resid.prior,
  #                       proposal.probs = proposal.probs,
  #                       sigma = sigmadbarts
  #   )
  #
  #   # print("Line 425")
  #
  #
  #   xdf_z <- data.frame(y = z - offsetz, x = w.train)
  #   xdf_z_test <- data.frame(x = w.test)
  #
  #
  #   sampler_z <- dbarts(y ~ .,
  #                       data = xdf_z,
  #                       test = xdf_z_test,
  #                       # weights = weightstemp,
  #                       control = control_z,
  #                       tree.prior = dbarts:::cgm(power = tree_power_z, base = tree_base_z,  split.probs = rep(1 / p_z, p_z)),
  #                       node.prior = node.prior,
  #                       resid.prior = resid.prior,
  #                       proposal.probs = proposal.probs,
  #                       sigma = 1#sigmadbarts
  #   )
  #
  # }

  # print("Line 473")

  preds.train_ystar <- matrix(NA, n, 1)
  preds.train_z <- matrix(NA, n, 1)

  preds.test_ystar <- matrix(NA, ntest, 1)
  preds.test_z <- matrix(NA, ntest, 1)

  mutemp_y <- rep(mean(ystar[uncens_inds]), length(uncens_inds))

  mutemp_y_trees <- rep(0, length(uncens_inds))
  mutemp_z_trees <- rep(0, n)
  mutemp_y_lin <- rep(0, length(uncens_inds))
  mutemp_z_lin <- rep(0, n)
  mutemp_test_y_trees <- rep(0, ntest)
  mutemp_test_z_trees <- rep(0, ntest)

  if(linearterms){
    alpha_var <- solve(solve(Avar_p) +
                         crossprod(wmat_train[cens_inds,, drop = FALSE]) +
                         ((gamma1^2 + phi1)/phi1)*crossprod(wmat_train[uncens_inds,]))

    # print("line 795 = ")

    alpha0hat <- solve(crossprod(wmat_train[cens_inds,, drop = FALSE])) %*% crossprod(wmat_train[cens_inds,, drop = FALSE], z[cens_inds] - offsetz)
    # print("line 798 = ")
    alpha1hat <- solve(crossprod(wmat_train[uncens_inds,, drop = FALSE])) %*% crossprod(wmat_train[uncens_inds,, drop = FALSE],
                                                                                        z[uncens_inds] - offsetz -
                                                                                          (ystar[uncens_inds]  - mutemp_y)*gamma1/(phi1 + gamma1^2))
    # print("line 800 = ")

    alpha_mean <- alpha_var %*% (solve(Avar_p) %*% Amean_p +
                                   crossprod(wmat_train[cens_inds,]) %*%  alpha0hat+
                                   ((gamma1^2 + phi1)/phi1)*crossprod(wmat_train[uncens_inds,]) %*% alpha1hat  )


    alpha_vec <- mvrnorm(1, mu = alpha_mean, Sigma = alpha_var)
    # print("line 808 = ")

    mutemp_z_lin <- wmat_train %*% alpha_vec
    mutemp_test_z_lin <- wmat_test %*% alpha_vec
    z_epsilon <- z - offsetz - mutemp_z_lin


    mutemp_z <- mutemp_z_lin
    if(any(is.na(z_epsilon))){
      stop("801 any(is.na(z_epsilon))")
    }
    if(any(is.na(mutemp_z_lin))){
      stop("804 any(is.na(mutemp_z_lin))")
    }

    if(jointbetagamma){
      # print("line 815 = ")
      y_resids <- ystar[uncens_inds] - gamma1*(z[uncens_inds] - offsetz - mutemp_z_lin[uncens_inds])
      Xresmat <- cbind(xmat_train ,z[uncens_inds] - offsetz - mutemp_z_lin[uncens_inds] )

      gamma0res <- c(Bmean_p,gamma0)
      G0res <- rbind(cbind(Bvar_p, rep(0, nrow(Bvar_p))),
                     t(c(rep(0,ncol(Bvar_p)),  tau*phi1)))

      Gbar <- solve(solve(G0res) + (1/phi1)*crossprod(Xresmat)  )
      ghat <- solve(crossprod(Xresmat))%*% crossprod(Xresmat,  ystar[uncens_inds])
      g_bar <- Gbar %*% (solve(G0res)%*%gamma0res + (1/phi1)*crossprod(Xresmat)%*% ghat)

      # sample beta vector and gamma simulataneously
      betagamma_vec <- mvrnorm(1, mu = g_bar, Sigma = Gbar)

      gamma1 <- betagamma_vec[length(betagamma_vec)]
      beta_vec <- betagamma_vec[- length(betagamma_vec)]
      mutemp_y_lin <- xmat_train %*% beta_vec
      mutemp_test_y_lin <- xmat_test %*% beta_vec

    }else{

      # # sample as SUR as outlined by VH? requires data augmentation of y.
      # D0_mat <- rbind(cbind(Avar_p, matrix(0,nrow = nrow(Avar_p), ncol = ncol(Bvar_p))),
      #                 cbind( matrix(0,nrow = nrow(Bvar_p), ncol = ncol(Avar_p)), Bvar_p))

      # instead sample alpha and beta separately

      y_resids <- ystar[uncens_inds] - gamma1*(z[uncens_inds] - offsetz - mutemp_z_lin[uncens_inds])

      # print("line 849 = ")

      Xuncensmat <- xmat_train

      Bvar <- solve(solve(Bvar_p) + (1/phi1)*crossprod(Xuncensmat)  )
      Bhat <- solve((1/phi1)*crossprod(Xuncensmat))%*% ((1/phi1)*crossprod(Xuncensmat,  y_resids) )
      B_bar <- Bvar %*% (solve(Bvar_p)%*%Bmean_p + (1/phi1)*crossprod(Xuncensmat)%*% Bhat)

      # sample beta vector and gamma simulataneously
      beta_vec <- mvrnorm(1, mu = B_bar, Sigma = Bvar)

      mutemp_y_lin <- xmat_train %*% beta_vec
      mutemp_test_y_lin <- xmat_test %*% beta_vec
    }
    mutemp_y <- mutemp_y_lin
  }

  #initialize sum-of-tree sampler
  z_resids <- z - offsetz #z_epsilon
  z_resids[uncens_inds] <- z[uncens_inds] - offsetz - (ystar[uncens_inds]  - mutemp_y)*gamma1/(phi1 + gamma1^2)

  # sampler_z$setResponse(y = z_resids)
  #
  # sampler_z$setSigma(sigma = 1)
  # sampler_z$setWeights(weights = weightstemp)

  # if(sparse){
  #   tempmodel <- sampler_z$model
  #   tempmodel@tree.prior@splitProbabilities <- s_z
  #   sampler_z$setModel(newModel = tempmodel)
  # }

  # print("Line 509")

  # sampler_z$sampleTreesFromPrior()
  #
  # # priormean_z <- sampler_z$predict(xdf_z)[1,]
  #
  # sampler_z$sampleNodeParametersFromPrior()
  #
  # samplestemp_z <- sampler_z$run()
  #
  #
  # # mutemp_z <- rep(0,n) # samplestemp_z$train[,1]
  # # mutemp_test_z <- rep(0,ntest) #samplestemp_z$test[,1]
  #
  # mutemp_z <- samplestemp_z$train[,1]
  # mutemp_test_z <- samplestemp_z$test[,1]

  ###### new myBART initialization for z model ######################

  # maybe (max(as.vector(Z.mat))-min(as.vector(Z.mat)))
  # can be replaced by something else
  # sigma2_mu <- (max(as.vector(Z.mat))-min(as.vector(Z.mat)))/((2 * k * sqrt(n.trees_censoring))^2)
  # sigma2_mu_z <- ((max(z_resids)-min(z_resids))/(2 * k_z * sqrt(n.trees_censoring)))^2
  sigma2_mu_z <- 9/( k_z * sqrt(n.trees_censoring))^2

  if(is.na(sigma2_mu_z )){

    print("z = ")
    print(z)

    print("ystar[uncens_inds] = ")
    print(ystar[uncens_inds])

    print("gamma1 = ")
    print(gamma1)

    print("gamma1 = ")
    print(gamma1)

    print("z_resids[uncens_inds] = ")
    print(z_resids[uncens_inds])

    print("z_resids = ")
    print(z_resids)

    print("k_z = ")
    print(k_z)

    print("n.trees_censoring = ")
    print(n.trees_censoring)

    stop("Line 814. sigma2_mu_z  NA")
  }

  # sigma2_mu <- 1/n.trees_censoring



  # EDIT THIS TO ACCOUNT FOR WEIGHTS

  # Create a list of trees for the initial stump
  curr_trees_z = create_stump(num_trees = n.trees_censoring,
                              y = as.vector(z_resids),
                              X = w.train)
  # Initialise the new trees as current one
  new_trees_z = curr_trees_z

  # Initialise the predicted values to zero
  mutemp_z = get_predictions(curr_trees_z, w.train, single_tree = n.trees_censoring == 1)
  # z_hat <- mutemp_z
  if(linearterms){
    mutemp_z <- mutemp_z + mutemp_z_lin
  }


  mu_z <- mutemp_z


  if(marginalize){
    weightz <- (gamma1^2 + phi1)/phi1
    weightstemp <- rep(1,n)
    weightstemp[uncens_inds] <- weightz
    binmat_all_y <- matrix(1, nrow = n1, ncol = n.trees_outcome)
    binmat_all_z <- matrix(1, nrow = n, ncol = n.trees_censoring)
    binmat_all_z_u <- matrix(1, nrow = n1, ncol = n.trees_censoring)
    binmat_all_z_c <- matrix(1, nrow = n0, ncol = n.trees_censoring)
    BtB_z_u <- crossprod(binmat_all_z[uncens_inds,])
    BtB_z_c <- crossprod(binmat_all_z[cens_inds,])

    firstcolindtrees_y <- 1:n.trees_outcome
    firstcolindtrees_z <- 1:n.trees_censoring


    # if(  !is.symmetric(BtB_z_u)){
    if( any(abs(BtB_z_u - t(BtB_z_u))> 0.001)){
      stop("line 1087 !is.symmetric(BtB_z_u) ")
    }


    if( (one_chol == TRUE)| linearterms ) {


      if(linearterms){
        if(one_chol ==TRUE){

          reslisttemp = tree_full_conditional_z_marg_lin_savechol(curr_trees_z, #[[j]],
                                                                  z_resids,# sigma2,
                                                                  sigma2_mu_z,
                                                                  weightstemp,
                                                                  weightz,
                                                                  binmat_all_z, cens_inds, uncens_inds, BtB_z_u, BtB_z_c,
                                                                  wmat_train, Amean_p, invAvar_p)

          IR_old_z <- reslisttemp[[2]]
          S_j_old_z <- reslisttemp[[3]]
        }
      }else{
        if(one_chol ==TRUE){
          reslisttemp = tree_full_conditional_z_marg_savechol(curr_trees_z, #[[j]],
                                                              z_resids,# sigma2,
                                                              sigma2_mu_z,
                                                              weightstemp,
                                                              weightz,
                                                              binmat_all_z, cens_inds, uncens_inds, BtB_z_u, BtB_z_c)

          IR_old_z <- reslisttemp[[2]]
          S_j_old_z <- reslisttemp[[3]]
        }
      }


      if(linearterms){
        if(one_chol ==TRUE){

          mudrawlist_z = simulate_mu_weighted_all_z_fast_lin(curr_trees_z,
                                                             z_resids,
                                                             # sigma2,
                                                             sigma2_mu_z,
                                                             weightstemp,
                                                             weightz,
                                                             binmat_all_z, cens_inds, uncens_inds, BtB_z_u, BtB_z_c, firstcolindtrees_z,
                                                             IR_old_z, S_j_old_z, wmat_train, Amean_p, invAvar_p)
        }else{
          mudrawlist_z = simulate_mu_weighted_all_z_lin(curr_trees_z,
                                                        z_resids,
                                                        # sigma2,
                                                        sigma2_mu_z,
                                                        weightstemp,
                                                        weightz,
                                                        binmat_all_z, cens_inds, uncens_inds, BtB_z_u, BtB_z_c, firstcolindtrees_z,
                                                        wmat_train, Amean_p, invAvar_p)
        }
      }else{
        mudrawlist_z = simulate_mu_weighted_all_z_fast(curr_trees_z,
                                                       z_resids,
                                                       # sigma2,
                                                       sigma2_mu_z,
                                                       weightstemp,
                                                       weightz,
                                                       binmat_all_z, cens_inds, uncens_inds, BtB_z_u, BtB_z_c, firstcolindtrees_z,
                                                       IR_old_z, S_j_old_z)
      }
    }else{
      mudrawlist_z = simulate_mu_weighted_all_z(curr_trees_z,
                                                z_resids,
                                                # sigma2,
                                                sigma2_mu_z,
                                                weightstemp,
                                                weightz,
                                                binmat_all_z, cens_inds, uncens_inds, BtB_z_u, BtB_z_c, firstcolindtrees_z)
    }

    mutemp_test_z <- get_predictions(curr_trees_z,
                                     w.test,
                                     single_tree = length(curr_trees_z) == 1)

    if(linearterms){
      mutemp_z <- cbind(wmat_train, binmat_all_z) %*% mudrawlist_z[[5]]
    }else{
      mutemp_z <- binmat_all_z %*% mudrawlist_z[[3]]
    }

  }




  z_epsilon <- z - offsetz - mutemp_z

  z_resids <- z - offsetz #z_epsilon
  z_resids[uncens_inds] <- z[uncens_inds] - offsetz - (ystar[uncens_inds]  - mutemp_y)*gamma1/(phi1 + gamma1^2)










  # mutemp_test_z <- sampler_z$predict(xdf_z_test)[,1]#samplestemp_z$test[,1]


  # if(sparse){
  var_count_z <- rep(0, p_z)
  # }
  # if(sparse){
  #   tempcounts <- fcount(sampler_z$getTrees()$var)
  #   tempcounts <- tempcounts[tempcounts$x != -1, ]
  #   var_count_z[tempcounts$x] <- tempcounts$N
  # }





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

  y_resids <- ystar[uncens_inds] - gamma1*(z[uncens_inds] - offsetz - mutemp_z[uncens_inds])
  y_uncens <- ystar[uncens_inds]

  # sampler_y$setResponse(y = y_resids)
  # sampler_y$setSigma(sigma = sqrt(phi1) )

  # sampler_y$setSigma(sigma = sigest)


  # if(sparse){
  #   tempmodel <- sampler_y$model
  #   tempmodel@tree.prior@splitProbabilities <- s_y
  #   sampler_y$setModel(newModel = tempmodel)
  # }


  # sampler_y$sampleTreesFromPrior()
  #
  # # priormean_y <- sampler_y$predict(xdf_y)[1,]
  #
  # sampler_y$sampleNodeParametersFromPrior()
  #
  # samplestemp_y <- sampler_y$run()
  #
  # # mutemp_y <- rep(mean(y),n) #samplestemp_y$train[,1]
  # # mutemp_test_y <- rep(mean(y),ntest) # samplestemp_y$test[,1]
  #
  # mutemp_y <- samplestemp_y$train[,1]
  # mutemp_test_y <- samplestemp_y$test

  ###### new myBART initialization for y model ######################

  # maybe (max(as.vector(Z.mat))-min(as.vector(Z.mat)))
  # can be replaced by something else
  # sigma2_mu <- (max(as.vector(Z.mat))-min(as.vector(Z.mat)))/((2 * k * sqrt(n.trees_outcome))^2)


  sigma2_mu_y <- ((max(ystar)-min(ystar))/(2 * k_y * sqrt(n.trees_outcome)))^2
  # sigma2_mu_y <- (max(ystar)-min(ystar))/((2 * k_y * sqrt(n.trees_outcome))^2)
  # An alternative would be to scale with sigest, e.g. 4*sigest//(2 * k_y * sqrt(n.trees_outcome))^2
  # sigma2_mu_y <-( 6*sigest/(2 * k_y * sqrt(n.trees_outcome)))^2

  # if(marginalize){
  #   sigma2_mu_y <- (max(ystar)-min(ystar))/(2 * k_y * sqrt(n.trees_outcome))^2
  # }

  if(is.na(sigma2_mu_y )){
    stop("Line 1733 sigma2_mu_y  NA")
  }

  # sigma2_mu <- 1/n.trees_outcome

  # Create a list of trees for the initial stump
  curr_trees_y = create_stump(num_trees = n.trees_outcome,
                              y = as.vector(y_resids),
                              X = x.train[uncens_inds,])
  # Initialise the new trees as current one
  new_trees_y = curr_trees_y

  # Initialise the predicted values to zero
  mutemp_y = get_predictions(curr_trees_y, x.train[uncens_inds,], single_tree = n.trees_outcome == 1)
  # y_hat <- mutemp_y
  if(linearterms){
    mutemp_y <- mutemp_y + mutemp_y_lin
  }
  mu_y <- mutemp_y


  if(marginalize){

    binmat_all_y_z <- cbind(binmat_all_y, z_epsilon[uncens_inds])
    BztBz_y <- crossprod(binmat_all_y_z)
    BtB_y <- crossprod(binmat_all_y)


    if(cov_prior == "VH"){
      priorgammavar <- tau*phi1
    }else{
      if(cov_prior == "Omori"){
        # stop("currently code only allows VH cov_prior")
        priorgammavar <- G0
      }else{
        if(jointgammanodes){
          stop("currently code only allows VH cov_prior with jointgammanodes")
        }
      }
    }

    if(linearterms){
      if(jointgammanodes){
        if(one_chol ==TRUE){

          # CURRENT TREE: compute the log of the marginalised likelihood + log of the tree prior
          reslisttemp = tree_full_conditional_y_marg_savechol_lin(curr_trees_y,
                                                                  y_uncens, # current_partial_residuals,
                                                                  phi1,priorgammavar,
                                                                  sigma2_mu_y, binmat_all_y_z, BztBz_y,
                                                                  Bmean_p, invBvar_p, xmat_train, gamma0)
          IR_old_y <- reslisttemp[[2]]
          S_j_old_y <- reslisttemp[[3]]
        }
      }else{
        if(one_chol ==TRUE){

          # CURRENT TREE: compute the log of the marginalised likelihood + log of the tree prior
          reslisttemp = tree_full_conditional_y_marg_nogamma_savechol_lin(curr_trees_y,
                                                                          y_resids, # current_partial_residuals,
                                                                          phi1,
                                                                          sigma2_mu_y, binmat_all_y, BtB_y,
                                                                          Bmean_p, invBvar_p, xmat_train)
          IR_old_y <- reslisttemp[[2]]
          S_j_old_y <- reslisttemp[[3]]
        }
      }
    }else{
      if(jointgammanodes){
        if(one_chol ==TRUE){
          # CURRENT TREE: compute the log of the marginalised likelihood + log of the tree prior
          reslisttemp = tree_full_conditional_y_marg_savechol(curr_trees_y,
                                                              y_uncens, # current_partial_residuals,
                                                              phi1,priorgammavar,
                                                              sigma2_mu_y, binmat_all_y_z, BztBz_y, gamma0)
          IR_old_y <- reslisttemp[[2]]
          S_j_old_y <- reslisttemp[[3]]
        }
      }else{
        if(one_chol ==TRUE){
          # CURRENT TREE: compute the log of the marginalised likelihood + log of the tree prior
          reslisttemp = tree_full_conditional_y_marg_nogamma_savechol(curr_trees_y,
                                                                      y_resids, # current_partial_residuals,
                                                                      phi1,
                                                                      sigma2_mu_y, binmat_all_y, BtB_y)
          IR_old_y <- reslisttemp[[2]]
          S_j_old_y <- reslisttemp[[3]]
        }
      }
    }

    if(linearterms){
      if(jointgammanodes){
        if(one_chol ==TRUE){
          mudrawlist_y = simulate_mu_all_y_fast_lin(curr_trees_y,
                                                    y_uncens, # current_partial_residuals,
                                                    phi1,
                                                    priorgammavar,
                                                    sigma2_mu_y, binmat_all_y_z, BztBz_y, firstcolindtrees_y, IR_old_y, S_j_old_y,
                                                    xmat_train, Bmean_p, invBvar_p, gamma0)
        }else{
          mudrawlist_y = simulate_mu_all_y_lin(curr_trees_y,
                                               y_uncens, # current_partial_residuals,
                                               phi1,
                                               priorgammavar,
                                               sigma2_mu_y, binmat_all_y_z, BztBz_y, firstcolindtrees_y,
                                               xmat_train, Bmean_p, invBvar_p, gamma0)
        }

      }else{
        if(one_chol ==TRUE){
          mudrawlist_y = simulate_mu_all_y_nogamma_fast_lin(curr_trees_y,
                                                            y_resids, # current_partial_residuals,
                                                            phi1,
                                                            sigma2_mu_y, binmat_all_y, BtB_y, firstcolindtrees_y, IR_old_y, S_j_old_y,
                                                            xmat_train, Bmean_p, invBvar_p)
        }else{

          mudrawlist_y = simulate_mu_all_y_nogamma_lin(curr_trees_y,
                                                       y_resids, # current_partial_residuals,
                                                       phi1,
                                                       sigma2_mu_y, binmat_all_y, BtB_y, firstcolindtrees_y,
                                                       xmat_train, Bmean_p, invBvar_p)
        }
      }

    }else{
      if(jointgammanodes){
        if(one_chol ==TRUE){
          mudrawlist_y = simulate_mu_all_y_fast(curr_trees_y,
                                                y_uncens, # current_partial_residuals,
                                                phi1,
                                                priorgammavar,
                                                sigma2_mu_y, binmat_all_y_z, BztBz_y, firstcolindtrees_y, IR_old_y, S_j_old_y, gamma0)
        }else{
          mudrawlist_y = simulate_mu_all_y(curr_trees_y,
                                           y_uncens, # current_partial_residuals,
                                           phi1,
                                           priorgammavar,
                                           sigma2_mu_y, binmat_all_y_z, BztBz_y, firstcolindtrees_y, gamma0)
        }

      }else{
        if(one_chol ==TRUE){
          mudrawlist_y = simulate_mu_all_y_nogamma_fast(curr_trees_y,
                                                        y_resids, # current_partial_residuals,
                                                        phi1,
                                                        sigma2_mu_y, binmat_all_y, BtB_y, firstcolindtrees_y, IR_old_y, S_j_old_y)
        }else{
          mudrawlist_y = simulate_mu_all_y_nogamma(curr_trees_y,
                                                   y_resids, # current_partial_residuals,
                                                   phi1,
                                                   sigma2_mu_y, binmat_all_y, BtB_y, firstcolindtrees_y)
        }

      }
    }


    mutemp_y <- binmat_all_y %*% mudrawlist_y[[3]]
    if(linearterms){
      if(jointgammanodes){
        mutemp_y <- mutemp_y + xmat_train %*% mudrawlist_y[[6]]
      }else{
        mutemp_y <- mutemp_y + xmat_train %*% mudrawlist_y[[4]]
      }
    }
    if(jointgammanodes){
      gamma1 <- mudrawlist_y[[5]]
    }
  }


  # if(sparse){
  var_count_y <- rep(0, p_y)
  # }
  # if(sparse){
  #   tempcounts <- fcount(sampler_y$getTrees()$var)
  #   tempcounts <- tempcounts[tempcounts$x != -1, ]
  #   var_count_y[tempcounts$x] <- tempcounts$N
  # }



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

  # create matrices of indicator variables for terminal nodes

  # if all initialized as stumps then all just constant vectors

  # will store as list, but can be stored as matrices instead
  # because matrices will be binded anyway

  binmat_list_y <- list()
  binmat_list_z <- list()

  for (j in 1:n.trees_outcome) {
    binmat_list_y[[j]] <- rep(1, n1) # just for uncensored outcomes?
  }
  for (j in 1:n.trees_censoring) {
    binmat_list_z[[j]] <- rep(1, n)
  }

  # or

  if(marginalize){
    binmat_all_y <- matrix(1, nrow = n1, ncol = n.trees_outcome)
    binmat_all_z <- matrix(1, nrow = n, ncol = n.trees_censoring)
    binmat_all_z_u <- matrix(1, nrow = n1, ncol = n.trees_censoring)
    binmat_all_z_c <- matrix(1, nrow = n0, ncol = n.trees_censoring)

    z_epsilon <- z - offsetz - mutemp_z

    BtB_y <- matrix(n1, nrow = n.trees_outcome, ncol = n.trees_outcome)
    BtB_y <- crossprod(binmat_all_y)

    Btz <- crossprod(binmat_all_y, z_epsilon[uncens_inds])
    # BztBz_y <- cbind(BtB_y, Btz)
    # BztBz_y <- rbind(BztBz_y, c(t(BtB_y), crossprod(z_epsilon[uncens_inds])))

    # BtB_z <- matrix(n, n.trees_censoring, n.trees_censoring)
    BtB_z_u <- matrix(n1, nrow = n.trees_censoring, ncol = n.trees_censoring)
    BtB_z_c <- matrix(n0, nrow = n.trees_censoring, ncol = n.trees_censoring)
    BtB_z_u <- crossprod(binmat_all_z_u)
    BtB_z_c <- crossprod(binmat_all_z_c)


    # if(  !is.symmetric(BtB_z_u)){
    if( any(abs(BtB_z_u - t(BtB_z_u))> 0.001)){

      stop("line 1577 !is.symmetric(BtB_z_u) ")
    }

    # then must save indices of column in matrix
    firstcolindtrees_y <- 1:n.trees_outcome
    firstcolindtrees_z <- 1:n.trees_censoring
  }











  #########  Begin Gibbs sampler ######################################################

  # pb <- progress_bar$new(total = n.iter+n.burnin)
  # pb <- progress_bar$new(
  #   format = " [:bar] :percent eta: :eta",
  #   total = n.iter+n.burnin, clear = FALSE, width= 60)




  # type_y_prev <- "none"
  # type_z_prev <- "none"

  #loop through the Gibbs sampler iterations
  for(iter in 1:(n.iter+n.burnin)){

    # print("iter = ")
    # print(iter)
    if(iter>1 &  (!marginalize) ){
      if(max(abs(mutemp_y - mutemp_y_lin - rowSums(tree_fits_store_y)))>0.0001){

        print("iter = ")
        print(iter)
        print("cbind(mutemp_y, mutemp_y_lin + rowSums(tree_fits_store_y)) = ")
        print(cbind(mutemp_y, mutemp_y_lin + rowSums(tree_fits_store_y)))


        stop(" print mutemp_y != mutemp_y_lin + rowSums(tree_fits_store_y)")

      }


      if(max(abs(mutemp_z - mutemp_z_lin - rowSums(tree_fits_store_z)))>0.0001){
        print("iter = ")
        print(iter)
        print("cbind(mutemp_z, mutemp_z_lin + rowSums(tree_fits_store_z)) = ")
        print(cbind(mutemp_z, mutemp_z_lin + rowSums(tree_fits_store_z)))

        stop(" print mutemp_z != mutemp_z_lin +  rowSums(tree_fits_store_z)")

      }
    }

    # if(eq_by_eq){
    #   var_zdraw <- 1
    #   # sig_ydraw <- phi1
    #
    # }else{
    #   var_zdraw <- phi1/(gamma1^2+phi1)
    #   # sig_ydraw <- phi1
    #
    # }

    temp_sd_z <- sqrt( phi1/(phi1+gamma1^2)   )

    ######### #draw the latent outcome z ##########################
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
      temp_zmean_cens <- offsetz + mutemp_z[cens_inds] #+ (ystar[cens_inds]  - mutemp_y[cens_inds])*gamma1/(phi1 + gamma1^2)

      z[cens_inds] <- rtruncnorm(n0, a= -Inf, b = 0, mean = temp_zmean_cens, sd = 1)
    }

    # temp_zmean_uncens <- offsetz + mutemp_z[uncens_inds] + (ystar[uncens_inds]  - mutemp_y[uncens_inds])*gamma1/(phi1 + gamma1^2)
    temp_zmean_uncens <- offsetz + mutemp_z[uncens_inds] + (ystar[uncens_inds]  - mutemp_y)*gamma1/(phi1 + gamma1^2)

    z[uncens_inds] <- rtruncnorm(n1, a= 0, b = Inf, mean = temp_zmean_uncens,
                                 sd = temp_sd_z)



    # z_epsilon <- z - offsetz - mutemp_z
    # y_epsilon <- ystar - mutemp_y

    z_epsilon <- z - offsetz - mutemp_z

    y_epsilon <- rep(0, n)
    y_epsilon[uncens_inds] <- ystar[uncens_inds] - mutemp_y

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

    ############ draw linear terms if not marginalizing #################

    if(linearterms & !marginalize){

      alpha_var <- solve(solve(Avar_p) +
                           crossprod(wmat_train[cens_inds, , drop = FALSE]) +
                           ((gamma1^2 + phi1)/phi1)*crossprod(wmat_train[uncens_inds,]))


      alpha0hat <- solve(crossprod(wmat_train[cens_inds,, drop = FALSE])) %*% crossprod(wmat_train[cens_inds,, drop = FALSE], z[cens_inds] - offsetz - mutemp_z_trees[cens_inds])
      alpha1hat <- solve(crossprod(wmat_train[uncens_inds,, drop = FALSE])) %*% crossprod(wmat_train[uncens_inds,, drop = FALSE],
                                                                                          z[uncens_inds] - offsetz - mutemp_z_trees[uncens_inds] -
                                                                                            (ystar[uncens_inds]  - mutemp_y_lin - mutemp_y_trees)*gamma1/(phi1 + gamma1^2))
      # print("line 1070 = ")

      alpha_mean <- alpha_var %*% (solve(Avar_p) %*% Amean_p +
                                     crossprod(wmat_train[cens_inds,, drop = FALSE]) %*%  alpha0hat+
                                     ((gamma1^2 + phi1)/phi1)*crossprod(wmat_train[uncens_inds,, drop = FALSE]) %*% alpha1hat  )

      # print("line 1076 = ")

      alpha_vec <- mvrnorm(1, mu = alpha_mean, Sigma = alpha_var)

      mutemp_z_lin <- wmat_train %*% alpha_vec
      mutemp_test_z_lin <- wmat_test %*% alpha_vec

      mutemp_z <- mutemp_z_lin + mutemp_z_trees
      mutemp_test_z <- mutemp_test_z_lin + mutemp_test_z_trees

      z_epsilon <- z - offsetz - mutemp_z

      # if(any(is.na(mutemp_z_lin))){
      #   stop("1213 any(is.na(mutemp_z))")
      # }
      # print("line 1081 = ")
      if(jointbetagamma){

        # print("line 1084 = ")

        Xresmat <- cbind(xmat_train ,z[uncens_inds] - mutemp_z_lin[uncens_inds] - mutemp_z_trees[uncens_inds] )

        gamma0res <- c(Bmean_p,gamma0)
        G0res <- rbind(cbind(Bvar_p, rep(0, nrow(Bvar_p))),
                       t(c(rep(0,ncol(Bvar_p)), tau*phi1)))

        Gbar <- solve(solve(G0res) + (1/phi1)*crossprod(Xresmat)  )
        ghat <- solve(crossprod(Xresmat))%*% crossprod(Xresmat,  ystar[uncens_inds] - mutemp_y_trees)
        g_bar <- Gbar %*% (solve(G0res)%*%gamma0res + (1/phi1)*crossprod(Xresmat)%*% ghat)

        # sample beta vector and gamma simulataneously
        betagamma_vec <- mvrnorm(1, mu = g_bar, Sigma = Gbar)

        gamma1 <- betagamma_vec[length(betagamma_vec)]
        beta_vec <- betagamma_vec[- length(betagamma_vec)]
        mutemp_y_lin <- xmat_train %*% beta_vec
        mutemp_test_y_lin <- xmat_test %*% beta_vec

        mutemp_y <- mutemp_y_lin + mutemp_y_trees
        mutemp_test_y <- mutemp_test_y_lin + mutemp_test_y_trees

      }else{
        # print("line 1105 = ")

        # # sample as SUR as outlined by VH? requires data augmentation of y.
        # D0_mat <- rbind(cbind(Avar_p, matrix(0,nrow = nrow(Avar_p), ncol = ncol(Bvar_p))),
        #                 cbind( matrix(0,nrow = nrow(Bvar_p), ncol = ncol(Avar_p)), Bvar_p))

        # instead sample alpha and beta separately

        y_resids <- ystar[uncens_inds] - mutemp_y_trees - gamma1*(z[uncens_inds] - offsetz - mutemp_z_lin[uncens_inds] - mutemp_z_trees[uncens_inds])

        Xuncensmat <- xmat_train

        Bvar <- solve(solve(Bvar_p) + (1/phi1)*crossprod(Xuncensmat)  )
        Bhat <- solve((1/phi1)*crossprod(Xuncensmat))%*% ((1/phi1)*crossprod(Xuncensmat,  y_resids) )
        B_bar <- Bvar %*% (solve(Bvar_p)%*%Bmean_p + (1/phi1)*crossprod(Xuncensmat)%*% Bhat)

        # sample beta vector and gamma simulataneously
        beta_vec <- mvrnorm(1, mu = B_bar, Sigma = Bvar)
        mutemp_y_lin <- xmat_train %*% beta_vec
        mutemp_test_y_lin <- xmat_test %*% beta_vec

        mutemp_y <- mutemp_y_lin + mutemp_y_trees
        mutemp_test <- mutemp_test_y_lin + mutemp_test_y_trees
      }
      y_epsilon[uncens_inds] <- ystar[uncens_inds] - mutemp_y

    } # end linear samples !marginalize




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

    z_resids <- z - offsetz #z_epsilon
    z_resids[uncens_inds] <- z[uncens_inds] - offsetz - (ystar[uncens_inds]  - mutemp_y)*gamma1/(phi1 + gamma1^2)

    y_resids <- ystar[uncens_inds] - gamma1*(z[uncens_inds] - offsetz - mutemp_z[uncens_inds])

    # #set the response for draws of z trees
    # sampler_z$setResponse(y = z_resids)
    # #set the standard deivation
    # sampler_z$setSigma(sigma = 1)


    weightz <- (gamma1^2 + phi1)/phi1
    weightstemp <- rep(1,n)
    weightstemp[uncens_inds] <- weightz

    # print("Draw z trees. iter = ")
    # print(iter)

    if(marginalize){


      for (j in 1:n.trees_censoring) {
        current_partial_residuals = z_resids #- mutemp_z + tree_fits_store_z[,j]


        # We need the new and old trees for the likelihoods
        new_trees_z <- curr_trees_z

        type_z = sample_move(curr_trees_z[[j]], i, 0, #n_burn
                             trans_prob)

        # Generate a new tree based on the current
        new_trees_z[[j]] <- update_tree(
          y = z_resids,
          X = w.train,
          type = type_z,
          curr_tree = curr_trees_z[[j]],
          node_min_size = node_min_size,
          s = s_z,
          max_bad_trees = max_bad_trees,
          splitting_rules = splitting_rules
        )

        # (c) Obtain the Metropolis-Hastings probability
        curr_tree_z <- curr_trees_z[[j]]
        new_tree_z <- new_trees_z[[j]]

        # must create tree matrices

        # very efficient code would just update relevant elements of indicator variable matrix
        # and elements of cross product matrix

        # For now, just convert node indices to binary variables


        # if( j > 1 | iter >1){
        #   if(ncol(BtB_z_u)!= ncol(binmat_all_z)){
        #     print("dim(BtB_z_u) =")
        #     print(dim(BtB_z_u))
        #     print("dim(binmat_all_z_new) =")
        #     print(dim(binmat_all_z_new))
        #
        #     print("j = ")
        #     print(j)
        #
        #     print("iter = ")
        #     print(iter)
        #
        #     stop("line 1241 ncol(BtB_z_u)!= ncol(binmat_all_z)")
        #   }
        # }

        # firstcolindtrees_z[j]

        # propsed model binary matrix
        # binmat_all_z

        # print("line 1882. iter = ")
        # print(iter)

        # require the node, not just the variable

        if (type_z == "grow") {
          # var_count_z[curr_trees_z[[j]]$var] <- var_count_z[curr_trees_z[[j]]$var] + 1

          # split node is just parent of last rows
          # new_tree_z$tree_matrix[nrow(new_tree_z$tree_matrix)-1,'parent']

          terminal_nodes_old = which(as.numeric(curr_tree_z$tree_matrix[,'terminal']) == 1)
          terminal_nodes_new = which(as.numeric(new_tree_z$tree_matrix[,'terminal']) == 1)
          removednode <- which(!( terminal_nodes_old %in% terminal_nodes_new ) )
          addednodes <- which(!( terminal_nodes_new %in% terminal_nodes_old ) )

          removednode_rowind <- terminal_nodes_old[removednode]
          addednodes_rowind <- terminal_nodes_new[addednodes]



          firstcolindtrees_z_new <- firstcolindtrees_z

          if(length(addednodes)==0){
            binmat_all_z_new <- binmat_all_z
            # BtB_z_new <- BtB_z
            BtB_z_new_u <- BtB_z_u
            BtB_z_new_c <- BtB_z_c
            # do not edit firstcolindtrees_z_new
          }else{


            # obtain new splitting variable and splitting point to calculate gating function
            #
            split_node_ind <- new_tree_z$tree_matrix[addednodes_rowind[1],'parent']
            split_var <- new_tree_z$tree_matrix[split_node_ind, 'split_variable']
            split_value <- new_tree_z$tree_matrix[split_node_ind, 'split_value']
            # calculate the gating function for all observations
            # gat_func_psi <- 1/(1 + exp(- (w.train[,split_var] - split_value)/tau_vec_censoring[j] ) )
            # gat_func_psi <- gating_func_logistic((w.train[,split_var] - split_value)/tau_vec_censoring[j] )
            gat_func_psi <- plogis((w.train[,split_var] - split_value)/tau_vec_censoring[j] )

            if(any(is.na(gat_func_psi))){
              print("gat_func_psi = ")
              print(gat_func_psi)

              print("w.train[,split_var] = ")
              print(w.train[,split_var])

              print("tau_vec_censoring[j] = ")
              print(tau_vec_censoring[j])

              print("split_var = ")
              print(split_var)

              print("split_value = ")
              print(split_value)

              print("addednodes_rowind = ")
              print(addednodes_rowind)

              print("new_tree_z$tree_matrix = ")
              print(new_tree_z$tree_matrix)

              stop("Line 1897 any(is.na(gat_func_psi))")
            }
            # create variables for new nodes
            # can either do this within a new grow_tree function or here
            # can just use node indices

            newnodesbin <- matrix(0, nrow(binmat_all_z),2)

            newnodesbin[,1] <- binmat_all_z[, (firstcolindtrees_z[j]-1+ removednode) , drop = FALSE]*gat_func_psi
            newnodesbin[,2] <- binmat_all_z[, (firstcolindtrees_z[j]-1+ removednode) , drop = FALSE]*(1-gat_func_psi)

            # newnodesbin[new_tree_z$node_indices == addednodes_rowind[1],1] <- rep(1, sum(new_tree_z$node_indices == addednodes_rowind[1]))
            # newnodesbin[new_tree_z$node_indices == addednodes_rowind[2],2] <- rep(1, sum(new_tree_z$node_indices == addednodes_rowind[2]))

            if(any(is.na(newnodesbin))){
              print("newnodesbin = ")
              print(newnodesbin)
              stop("Line 1921 any(is.na(newnodesbin))")
            }


            binmat_all_z_new <- binmat_all_z[, -(firstcolindtrees_z[j]-1+ removednode) , drop = FALSE]

            # BtB_z_new <- BtB_z[-(firstcolindtrees_z[j]-1+ removednode), -(firstcolindtrees_z[j]-1+ removednode) ]
            BtB_z_new_u <- BtB_z_u[-(firstcolindtrees_z[j]-1+ removednode), -(firstcolindtrees_z[j]-1+ removednode) , drop = FALSE]
            BtB_z_new_c <- BtB_z_c[-(firstcolindtrees_z[j]-1+ removednode), -(firstcolindtrees_z[j]-1+ removednode) , drop = FALSE]

            # if(  !is.symmetric(BtB_z_new_u)){
            if( any(abs(BtB_z_new_u - t(BtB_z_new_u))> 0.001)){

              stop("line 1966 !is.symmetric(BtB_z_new_u) ")
            }

            if(firstcolindtrees_z[j]-1 + addednodes[1]-1 == 0 ){
              binmat_all_z_new <- cbind(#binmat_all_z_new[, 1:(firstcolindtrees_z[j]-1 + addednodes-1 ) ],
                newnodesbin,
                binmat_all_z_new[, (firstcolindtrees_z[j]-1 + addednodes[1] ):ncol(binmat_all_z_new)  ])

              if(any(is.na(binmat_all_z_new))){
                print("binmat_all_z_new = ")
                print(binmat_all_z_new)
                stop("Line 1933 any(is.na(binmat_all_z_new))")
              }

              # if(any((binmat_all_z_new) ==0 )){
              #   print("which(binmat_all_z_new ==0, arr.ind = TRUE) = ")
              #   print(which(binmat_all_z_new ==0, arr.ind = TRUE))
              #
              #   print("binmat_all_z_new = ")
              #   print(binmat_all_z_new)
              #   stop("Line 1991 any((binmat_all_z_new) ==0 )))")
              # }

              # if((ncol(binmat_all_z_new) != ncol(binmat_all_z)+1  )){
              #
              #
              #   print("dim(BtB_z_new_u) = ")
              #   print(dim(BtB_z_new_u))
              #
              #   print("dim(BtB_z_u) = ")
              #   print(dim(BtB_z_u))
              #
              #   print("dim(binmat_all_z_new) = ")
              #   print(dim(binmat_all_z_new))
              #   print("dim(binmat_all_z) = ")
              #   print(dim(binmat_all_z))
              #   print("type_z = ")
              #   print(type_z)
              #
              #   stop("(Line 1302. ncol(binmat_all_z_new) != ncol(binmat_all_z)+1 )")
              # }


              # BtB_z_new <- cbind(rep(0, nrow(BtB_z_new)), rep(0, nrow(BtB_z_new)), BtB_z_new[, (firstcolindtrees_z[j]-1 + addednodes[1] ):ncol(BtB_z_new) ])
              # BtB_z_new <- rbind(rep(0, ncol(BtB_z_new)), rep(0, ncol(BtB_z_new)), BtB_z_new[, (firstcolindtrees_z[j]-1 + addednodes[1] ):nrow(BtB_z_new) ])
              #
              # BtB_z_new[1,1] <- sum(new_tree_z$node_indices == addednodes[1])
              # BtB_z_new[2,2] <- sum(new_tree_z$node_indices == addednodes[2])
              #
              # BtB_z_new[ (1:2) ,-c( (1:2) )] <- crossprod((binmat_all_z_new[ ,(1:2) ]) , binmat_all_z_new[ , - (1:2) ])
              # BtB_z_new[-c( (1:2)),(1:2)] <- t(BtB_z_new[(1:2),-c( (1:2))])


              # print("dim(binmat_all_z_new) = ")
              # print(dim(binmat_all_z_new))
              # print("dim(BtB_z_new_u) = ")
              # print(dim(BtB_z_new_u))
              #
              # print("addednodes = ")
              # print(addednodes)
              #
              # print("(firstcolindtrees_z[j]-1 + addednodes[1] ):ncol(BtB_z_new_u) = ")
              # print((firstcolindtrees_z[j]-1 + addednodes[1] ):ncol(BtB_z_new_u))
              #
              # print("line 1311")

              BtB_z_new_u <- cbind(rep(0, nrow(BtB_z_new_u)), rep(0, nrow(BtB_z_new_u)),
                                   BtB_z_new_u[, (firstcolindtrees_z[j]-1 + addednodes[1] ):ncol(BtB_z_new_u), drop = FALSE ])

              # print("line 1314")
              #
              # print("dim(BtB_z_new_u) = ")
              # print(dim(BtB_z_new_u))


              BtB_z_new_u <- rbind(rep(0, ncol(BtB_z_new_u)), rep(0, ncol(BtB_z_new_u)),
                                   BtB_z_new_u[(firstcolindtrees_z[j]-1 + addednodes[1] ):nrow(BtB_z_new_u), , drop = FALSE ])

              # print("dim(BtB_z_new_u) = ")
              # print(dim(BtB_z_new_u))

              # print("dim(BtB_z_new_u) = ")
              # print(dim(BtB_z_new_u))

              BtB_z_new_u[1:2,1:2] <-  t(newnodesbin[uncens_inds,]) %*% newnodesbin[uncens_inds,]

              # BtB_z_new_u[1,1] <- sum(newnodesbin[uncens_inds,1]^2) # sum(new_tree_z$node_indices[uncens_inds] == addednodes_rowind[1])
              # BtB_z_new_u[2,2] <- sum(newnodesbin[uncens_inds,2]^2) # sum(new_tree_z$node_indices[uncens_inds] == addednodes_rowind[2])

              BtB_z_new_u[ (1:2) ,-c( (1:2) )] <- crossprod((binmat_all_z_new[ uncens_inds,(1:2), drop = FALSE ]) ,
                                                            binmat_all_z_new[ uncens_inds, - (1:2), drop = FALSE ])
              BtB_z_new_u[-c( (1:2)),(1:2)] <- t(BtB_z_new_u[(1:2),-c( (1:2)), drop = FALSE ])

              # print("line 1333 dim(BtB_z_new_u) = ")
              # print(dim(BtB_z_new_u))


              if(any(is.na(BtB_z_new_u))){
                print("BtB_z_new_u = ")
                print(BtB_z_new_u)
                stop("Line 2009 any(is.na(BtB_z_new_u))")
              }

              BtB_z_new_c <- cbind(rep(0, nrow(BtB_z_new_c)), rep(0, nrow(BtB_z_new_c)),
                                   BtB_z_new_c[, (firstcolindtrees_z[j]-1 + addednodes[1] ):ncol(BtB_z_new_c), drop = FALSE ])
              BtB_z_new_c <- rbind(rep(0, ncol(BtB_z_new_c)), rep(0, ncol(BtB_z_new_c)),
                                   BtB_z_new_c[(firstcolindtrees_z[j]-1 + addednodes[1] ):nrow(BtB_z_new_c), , drop = FALSE])

              BtB_z_new_c[1:2,1:2] <-  t(newnodesbin[cens_inds,]) %*% newnodesbin[cens_inds,]

              # BtB_z_new_c[1,1] <-  sum(newnodesbin[cens_inds,1]^2) # sum(new_tree_z$node_indices[cens_inds] == addednodes_rowind[1])
              # BtB_z_new_c[2,2] <-  sum(newnodesbin[cens_inds,2]^2) # sum(new_tree_z$node_indices[cens_inds] == addednodes_rowind[2])

              BtB_z_new_c[ (1:2) ,-c( (1:2) )] <- crossprod((binmat_all_z_new[ cens_inds,(1:2), drop = FALSE  ]) ,
                                                            binmat_all_z_new[ cens_inds, - (1:2), drop = FALSE  ])
              BtB_z_new_c[-c( (1:2)),(1:2)] <- t(BtB_z_new_c[(1:2),-c( (1:2)), drop = FALSE ])


              # if(  !is.symmetric(BtB_z_new_u)){
              if( any(abs(BtB_z_new_u - t(BtB_z_new_u))> 0.001)){

                stop("line 2064 !is.symmetric(BtB_z_new_u) ")
              }


              # # new_tree_z$node_indices gives node indices of all observations
              #
              # # create row for first of 2 new nodes
              # # for each pair node in other trees,
              # # count the number of node obs that are also in addednodes[1]
              # for(j2 in setdiff(1:n.trees_censoring,j) ){
              #   temptree_z <- curr_trees_z[[j2]]
              #
              #   terminal_nodes_j = which(as.numeric(temptree_z$tree_matrix[,'terminal']) == 1)
              #
              #   # can this loop be vectorized?
              #   for(col_ind in 1:length(terminal_nodes_j)){
              #     # +1 for 1 more node added
              #     # + firstcolindtrees_z[j]-1 to go to just before columns of j2^th tree
              #     # + col_ind to go to the relevant column for the particular node
              #
              #     BtB_z_new[1,1 +  firstcolindtrees_z[j2]-1 + col_ind] <-   sum( (new_tree_z$node_indices == addednodes[1])&
              #                                                                      ( temptree_z$node_indices== terminal_nodes_j[col_ind] )  )
              #
              #     BtB_z_new[1 +  firstcolindtrees_z[j2]-1 + col_ind,1] <- BtB_z_new[1,1 +  firstcolindtrees_z[j2]-1 + col_ind]
              #
              #     BtB_z_new[2,1 +  firstcolindtrees_z[j2]-1 + col_ind] <-   sum( (new_tree_z$node_indices == addednodes[2])&
              #                                                                      ( temptree_z$node_indices== terminal_nodes_j[col_ind] )  )
              #
              #     BtB_z_new[1 +  firstcolindtrees_z[j2]-1 + col_ind,2] <- BtB_z_new[2,1 +  firstcolindtrees_z[j2]-1 + col_ind]
              #
              #   }
              # }
              # firstcolindtrees_z_new[2:length(firstcolindtrees_z_new)] <- 1 + firstcolindtrees_z_new[2:length(firstcolindtrees_z_new)]

            }else{
              if(firstcolindtrees_z[j]-1 + addednodes[2]-1 == ncol(binmat_all_z_new) +1 ){
                binmat_all_z_new <- cbind(binmat_all_z_new[, 1:(firstcolindtrees_z[j]-1 + addednodes[1]-1 ), drop = FALSE  ],
                                          newnodesbin#,
                                          #binmat_all_z_new[, (firstcolindtrees_z[j]-1 + addednodes ):ncol(binmat_all_z_new) ]
                )

                if(any(is.na(binmat_all_z_new))){
                  print("binmat_all_z_new = ")
                  print(binmat_all_z_new)
                  stop("Line 2059 any(is.na(binmat_all_z_new))")
                }

                # if(any((binmat_all_z_new) ==0 )){
                #   print("which(binmat_all_z_new ==0, arr.ind = TRUE) = ")
                #   print(which(binmat_all_z_new ==0, arr.ind = TRUE))
                #
                #   print("binmat_all_z_new = ")
                #   print(binmat_all_z_new)
                #   stop("Line 1991 any((binmat_all_z_new) ==0 )))")
                # }

                # BtB_z_new <- cbind( BtB_z_new[, 1:(firstcolindtrees_z[j]-1 + addednodes[1]-1 ) ],
                #                     rep(0, nrow(BtB_z_new)), rep(0, nrow(BtB_z_new)))
                # BtB_z_new <- rbind(BtB_z_new[, 1:(firstcolindtrees_z[j]-1 + addednodes[1]-1 ) ],
                #                    rep(0, ncol(BtB_z_new)), rep(0, ncol(BtB_z_new)))
                #
                # BtB_z_new[ncol(BtB_z_new)-1,ncol(BtB_z_new)-1] <- sum(new_tree_z$node_indices == addednodes[1])
                # BtB_z_new[ncol(BtB_z_new),ncol(BtB_z_new)] <- sum(new_tree_z$node_indices == addednodes[2])
                #
                # BtB_z_new[c(ncol(BtB_z_new)-1,ncol(BtB_z_new) ),
                #           -c(c(ncol(BtB_z_new)-1,ncol(BtB_z_new) ))] <-
                #   crossprod((binmat_all_z_new[ ,c(ncol(BtB_z_new)-1,ncol(BtB_z_new)) ]) , binmat_all_z_new[ , - c(ncol(BtB_z_new)-1,ncol(BtB_z_new)) ])
                #
                # BtB_z_new[-c(c(ncol(BtB_z_new)-1,ncol(BtB_z_new) )),
                #           c(ncol(BtB_z_new)-1,ncol(BtB_z_new) )] <- t(BtB_z_new[c(ncol(BtB_z_new)-1,ncol(BtB_z_new) ),
                #                                                                 -c(c(ncol(BtB_z_new)-1,ncol(BtB_z_new) ))])


                BtB_z_new_u <- cbind( BtB_z_new_u[, 1:(firstcolindtrees_z[j]-1 + addednodes[1]-1  ), drop = FALSE  ],
                                      rep(0, nrow(BtB_z_new_u)), rep(0, nrow(BtB_z_new_u)))
                BtB_z_new_u <- rbind(BtB_z_new_u[1:(firstcolindtrees_z[j]-1 + addednodes[1]-1  ), , drop = FALSE ],
                                     rep(0, ncol(BtB_z_new_u)), rep(0, ncol(BtB_z_new_u)))

                # print("line 1410")
                BtB_z_new_u[(ncol(BtB_z_new_u)-1):ncol(BtB_z_new_u) ,
                            (ncol(BtB_z_new_u)-1):ncol(BtB_z_new_u)] <- t(newnodesbin[uncens_inds,]) %*% newnodesbin[uncens_inds,]

                # BtB_z_new_u[ncol(BtB_z_new_u)-1,ncol(BtB_z_new_u)-1] <- sum(newnodesbin[uncens_inds,1]^2) # sum(new_tree_z$node_indices[uncens_inds] == addednodes_rowind[1])
                # BtB_z_new_u[ncol(BtB_z_new_u),ncol(BtB_z_new_u)] <- sum(newnodesbin[uncens_inds,2]^2) # sum(new_tree_z$node_indices[uncens_inds] == addednodes_rowind[2])

                BtB_z_new_u[c(ncol(BtB_z_new_u)-1,ncol(BtB_z_new_u) ),
                            -c(c(ncol(BtB_z_new_u)-1,ncol(BtB_z_new_u) ))] <-
                  crossprod((binmat_all_z_new[uncens_inds ,c(ncol(BtB_z_new_u)-1,ncol(BtB_z_new_u)), drop = FALSE  ]) ,
                            binmat_all_z_new[uncens_inds , - c(ncol(BtB_z_new_u)-1,ncol(BtB_z_new_u)), drop = FALSE  ])

                BtB_z_new_u[-c(c(ncol(BtB_z_new_u)-1,ncol(BtB_z_new_u) )),
                            c(ncol(BtB_z_new_u)-1,ncol(BtB_z_new_u) )] <- t(BtB_z_new_u[c(ncol(BtB_z_new_u)-1,ncol(BtB_z_new_u) ),
                                                                                        -c(c(ncol(BtB_z_new_u)-1,ncol(BtB_z_new_u) )),
                                                                                        drop = FALSE ])

                if(any(is.na(BtB_z_new_u))){
                  print("BtB_z_new_u = ")
                  print(BtB_z_new_u)
                  stop("Line 2098 any(is.na(BtB_z_new_u))")
                }


                # if(any((BtB_z_new_u) ==0 )){
                #   print("which(BtB_z_new_u ==0, arr.ind = TRUE) = ")
                #   print(which(BtB_z_new_u ==0, arr.ind = TRUE))
                #
                #   print("BtB_z_new_u = ")
                #   print(BtB_z_new_u)
                #   stop("Line 1991 any((BtB_z_new_u) ==0 )))")
                # }


                BtB_z_new_c <- cbind( BtB_z_new_c[, 1:(firstcolindtrees_z[j]-1 + addednodes[1]-1 ), drop = FALSE  ],
                                      rep(0, nrow(BtB_z_new_c)), rep(0, nrow(BtB_z_new_c)))
                BtB_z_new_c <- rbind(BtB_z_new_c[1:(firstcolindtrees_z[j]-1 + addednodes[1]-1 ), ,drop = FALSE],
                                     rep(0, ncol(BtB_z_new_c)), rep(0, ncol(BtB_z_new_c)))

                # BtB_z_new_c[ncol(BtB_z_new_c)-1,ncol(BtB_z_new_c)-1] <-  sum(newnodesbin[cens_inds,1]^2) # sum(new_tree_z$node_indices[cens_inds] == addednodes_rowind[1])
                # BtB_z_new_c[ncol(BtB_z_new_c),ncol(BtB_z_new_c)] <- sum(newnodesbin[cens_inds,2]^2) # ssum(new_tree_z$node_indices[cens_inds] == addednodes_rowind[2])

                BtB_z_new_c[(ncol(BtB_z_new_c)-1):ncol(BtB_z_new_c) ,
                            (ncol(BtB_z_new_c)-1):ncol(BtB_z_new_c)] <- t(newnodesbin[cens_inds,]) %*% newnodesbin[cens_inds,]


                BtB_z_new_c[c(ncol(BtB_z_new_c)-1,ncol(BtB_z_new_c) ),
                            -c(c(ncol(BtB_z_new_c)-1,ncol(BtB_z_new_c) ))] <-
                  crossprod((binmat_all_z_new[cens_inds ,c(ncol(BtB_z_new_c)-1,ncol(BtB_z_new_c)) ]) ,
                            binmat_all_z_new[cens_inds , - c(ncol(BtB_z_new_c)-1,ncol(BtB_z_new_c)), drop = FALSE  ])

                BtB_z_new_c[-c(c(ncol(BtB_z_new_c)-1,ncol(BtB_z_new_c) )),
                            c(ncol(BtB_z_new_c)-1,ncol(BtB_z_new_c) )] <- t(BtB_z_new_c[c(ncol(BtB_z_new_c)-1,ncol(BtB_z_new_c) ),
                                                                                        -c(c(ncol(BtB_z_new_c)-1,ncol(BtB_z_new_c) )), drop = FALSE ])

                # if(  !is.symmetric(BtB_z_new_u)){
                if( any(abs(BtB_z_new_u - t(BtB_z_new_u))> 0.001)){

                  stop("line 2171 !is.symmetric(BtB_z_new_u) ")
                }

                # # new_tree_z$node_indices gives node indices of all observations
                #
                # # create row for first of 2 new nodes
                # # for each pair node in other trees,
                # # count the number of node obs that are also in addednodes[1]
                # for(j2 in setdiff(1:n.trees_censoring,j) ){
                #   temptree_z <- curr_trees_z[[j2]]
                #
                #   terminal_nodes_j = which(as.numeric(temptree_z$tree_matrix[,'terminal']) == 1)
                #
                #   # can this loop be vectorized?
                #   for(col_ind in 1:length(terminal_nodes_j)){
                #     # + firstcolindtrees_z[j]-1 to go to just before columns of j2^th tree
                #     # + col_ind to go to the relevant column for the particular node
                #
                #     BtB_z_new[ncol(BtB_z_new)-1,firstcolindtrees_z[j2]-1 + col_ind] <-   sum( (new_tree_z$node_indices == addednodes[1])&
                #                                                                      ( temptree_z$node_indices== terminal_nodes_j[col_ind] )  )
                #
                #     BtB_z_new[  firstcolindtrees_z[j2]-1 + col_ind,ncol(BtB_z_new)-1] <- BtB_z_new[1,1 +  firstcolindtrees_z[j2]-1 + col_ind]
                #
                #     BtB_z_new[ncol(BtB_z_new), firstcolindtrees_z[j2]-1 + col_ind] <-   sum( (new_tree_z$node_indices == addednodes[2])&
                #                                                                      ( temptree_z$node_indices== terminal_nodes_j[col_ind] )  )
                #
                #     BtB_z_new[  firstcolindtrees_z[j2]-1 + col_ind,ncol(BtB_z_new)] <- BtB_z_new[2,1 +  firstcolindtrees_z[j2]-1 + col_ind]
                #
                #   }
                # }

                # last tree grown. First column index unchanged

              }else{
                binmat_all_z_new <- cbind(binmat_all_z_new[, 1:(firstcolindtrees_z[j]-1 + addednodes[1]-1 ) , drop = FALSE ],
                                          newnodesbin,
                                          binmat_all_z_new[, (firstcolindtrees_z[j]-1 + addednodes[1] ):ncol(BtB_z_new_u), drop = FALSE  ])

                if(any(is.na(newnodesbin))){
                  print("newnodesbin = ")
                  print(newnodesbin)
                  stop("Line 2155 any(is.na(newnodesbin))")
                }
                if(any(is.na(binmat_all_z_new))){
                  print("binmat_all_z_new = ")
                  print(binmat_all_z_new)
                  stop("Line 2160 any(is.na(binmat_all_z_new))")
                }
                # BtB_z_new <- cbind( BtB_z_new[, 1:(firstcolindtrees_z[j]-1 + addednodes[1]-1 ) ],
                #                     rep(0, nrow(BtB_z_new)), rep(0, nrow(BtB_z_new)), BtB_z_new[, (firstcolindtrees_z[j]-1 + addednodes[1] ):ncol(binmat_all_z_new)  ])
                # BtB_z_new <- rbind(BtB_z_new[ 1:(firstcolindtrees_z[j]-1 + addednodes[1]-1 ), ],
                #                    rep(0, ncol(BtB_z_new)), rep(0, ncol(BtB_z_new)),
                #                    BtB_z_new[, (firstcolindtrees_z[j]-1 + addednodes[1] ):nrow(binmat_all_z_new)  ])
                #
                # BtB_z_new[firstcolindtrees_z[j]-1 + addednodes[1],
                #           firstcolindtrees_z[j]-1 + addednodes[1]] <- sum(new_tree_z$node_indices == addednodes[1])
                # BtB_z_new[firstcolindtrees_z[j]-1 + addednodes[2],
                #           firstcolindtrees_z[j]-1 + addednodes[2]] <- sum(new_tree_z$node_indices == addednodes[2])
                #
                # BtB_z_new[firstcolindtrees_z[j]-1 + addednodes[1:2],
                #           -c( firstcolindtrees_z[j]-1 + addednodes[1:2])] <-
                #   crossprod((binmat_all_z_new[ ,firstcolindtrees_z[j]-1 + addednodes[1:2] ]) , binmat_all_z_new[ , - (firstcolindtrees_z[j]-1 + addednodes[1:2]) ])
                #
                # BtB_z_new[-c(firstcolindtrees_z[j]-1 + addednodes[1:2]),
                #           firstcolindtrees_z[j]-1 + addednodes[1:2]] <- t(BtB_z_new[firstcolindtrees_z[j]-1 + addednodes[1:2],
                #                                                                   -c( firstcolindtrees_z[j]-1 + addednodes[1:2])])


                BtB_z_new_u <- cbind( BtB_z_new_u[, 1:(firstcolindtrees_z[j]-1 + addednodes[1]-1 ), drop = FALSE  ],
                                      rep(0, nrow(BtB_z_new_u)), rep(0, nrow(BtB_z_new_u)),
                                      BtB_z_new_u[, (firstcolindtrees_z[j]-1 + addednodes[1] ):ncol(BtB_z_new_u) , drop = FALSE  ])
                BtB_z_new_u <- rbind(BtB_z_new_u[ 1:(firstcolindtrees_z[j]-1 + addednodes[1]-1 ), , drop = FALSE  ],
                                     rep(0, ncol(BtB_z_new_u)), rep(0, ncol(BtB_z_new_u)),
                                     BtB_z_new_u[(firstcolindtrees_z[j]-1 + addednodes[1] ):nrow(BtB_z_new_u), , drop = FALSE])


                # print("line 1515")
                BtB_z_new_u[firstcolindtrees_z[j]-1 + addednodes[1:2],
                            firstcolindtrees_z[j]-1 + addednodes[1:2]] <- t(newnodesbin[uncens_inds,]) %*% newnodesbin[uncens_inds,]

                # BtB_z_new_u[firstcolindtrees_z[j]-1 + addednodes[1],
                #             firstcolindtrees_z[j]-1 + addednodes[1]] <- sum(newnodesbin[uncens_inds,1]^2) # sum(new_tree_z$node_indices[uncens_inds] == addednodes_rowind[1])
                # BtB_z_new_u[firstcolindtrees_z[j]-1 + addednodes[2],
                #             firstcolindtrees_z[j]-1 + addednodes[2]] <- sum(newnodesbin[uncens_inds,2]^2) # sum(new_tree_z$node_indices[uncens_inds] == addednodes_rowind[2])

                BtB_z_new_u[firstcolindtrees_z[j]-1 + addednodes[1:2],
                            -c( firstcolindtrees_z[j]-1 + addednodes[1:2])] <-
                  crossprod((binmat_all_z_new[ uncens_inds,firstcolindtrees_z[j]-1 + addednodes[1:2] , drop = FALSE ]) ,
                            binmat_all_z_new[ uncens_inds, - (firstcolindtrees_z[j]-1 + addednodes[1:2]) , drop = FALSE ])

                BtB_z_new_u[-c(firstcolindtrees_z[j]-1 + addednodes[1:2]),
                            firstcolindtrees_z[j]-1 + addednodes[1:2]] <- t(BtB_z_new_u[firstcolindtrees_z[j]-1 + addednodes[1:2],
                                                                                        -c( firstcolindtrees_z[j]-1 + addednodes[1:2]),
                                                                                        drop = FALSE ])

                if(any(is.na(BtB_z_new_u))){
                  print("BtB_z_new_u = ")
                  print(BtB_z_new_u)
                  stop("Line 2202 any(is.na(BtB_z_new_u))")
                }


                # if(any((binmat_all_z_new) ==0 )){
                #   print("which(binmat_all_z_new ==0, arr.ind = TRUE) = ")
                #   print(which(binmat_all_z_new ==0, arr.ind = TRUE))
                #
                #   print("binmat_all_z_new = ")
                #   print(binmat_all_z_new)
                #   stop("Line 2339 any((binmat_all_z_new) ==0 )))")
                # }
                #
                # if(any((BtB_z_new_u) ==0 )){
                #   print("which(BtB_z_new_u ==0, arr.ind = TRUE) = ")
                #   print(which(BtB_z_new_u ==0, arr.ind = TRUE))
                #
                #   print("BtB_z_new_u = ")
                #   print(BtB_z_new_u)
                #   stop("Line 2348 any((BtB_z_new_u) ==0 )))")
                # }

                BtB_z_new_c <- cbind( BtB_z_new_c[, 1:(firstcolindtrees_z[j]-1 + addednodes[1]-1 ) , drop = FALSE ],
                                      rep(0, nrow(BtB_z_new_c)), rep(0, nrow(BtB_z_new_c)),
                                      BtB_z_new_c[, (firstcolindtrees_z[j]-1 + addednodes[1] ):ncol(BtB_z_new_c) , drop = FALSE  ])
                BtB_z_new_c <- rbind(BtB_z_new_c[1:(firstcolindtrees_z[j]-1 + addednodes[1]-1 ), , drop = FALSE],
                                     rep(0, ncol(BtB_z_new_c)), rep(0, ncol(BtB_z_new_c)),
                                     BtB_z_new_c[(firstcolindtrees_z[j]-1 + addednodes[1] ):nrow(BtB_z_new_c), , drop = FALSE])

                BtB_z_new_c[firstcolindtrees_z[j]-1 + addednodes[1:2],
                            firstcolindtrees_z[j]-1 + addednodes[1:2]] <- t(newnodesbin[cens_inds,]) %*% newnodesbin[cens_inds,]

                # BtB_z_new_c[firstcolindtrees_z[j]-1 + addednodes[1],
                #             firstcolindtrees_z[j]-1 + addednodes[1]] <- sum(newnodesbin[cens_inds,1]^2) # sum(new_tree_z$node_indices[cens_inds] == addednodes_rowind[1])
                # BtB_z_new_c[firstcolindtrees_z[j]-1 + addednodes[2],
                #             firstcolindtrees_z[j]-1 + addednodes[2]] <- sum(newnodesbin[cens_inds,2]^2) # sum(new_tree_z$node_indices[cens_inds] == addednodes_rowind[2])

                BtB_z_new_c[firstcolindtrees_z[j]-1 + addednodes[1:2],
                            -c( firstcolindtrees_z[j]-1 + addednodes[1:2])] <-
                  crossprod((binmat_all_z_new[ cens_inds,firstcolindtrees_z[j]-1 + addednodes[1:2] , drop = FALSE ]) ,
                            binmat_all_z_new[ cens_inds, - (firstcolindtrees_z[j]-1 + addednodes[1:2]), drop = FALSE  ])

                BtB_z_new_c[-c(firstcolindtrees_z[j]-1 + addednodes[1:2]),
                            firstcolindtrees_z[j]-1 + addednodes[1:2]] <- t(BtB_z_new_c[firstcolindtrees_z[j]-1 + addednodes[1:2],
                                                                                        -c( firstcolindtrees_z[j]-1 + addednodes[1:2]), drop = FALSE ])


                # if(  !is.symmetric(BtB_z_new_u)){
                if( any(abs(BtB_z_new_u - t(BtB_z_new_u))> 0.001)){

                  stop("line 2294 !is.symmetric(BtB_z_new_u) ")
                }
                # new_tree_z$node_indices gives node indices of all observations

                # create row for first of 2 new nodes
                # for each pair node in other trees,
                # count the number of node obs that are also in addednodes[1]
                # for(j2 in setdiff(1:n.trees_censoring,j) ){
                #   temptree_z <- curr_trees_z[[j2]]
                #   terminal_nodes_j = which(as.numeric(temptree_z$tree_matrix[,'terminal']) == 1)
                #
                #   # can this loop be vectorized?
                #   for(col_ind in 1:length(terminal_nodes_j)){
                #     # + firstcolindtrees_z[j]-1 to go to just before columns of j2^th tree
                #     # + col_ind to go to the relevant column for the particular node
                #
                #     # if new tree before j2, add 1 to the index because there is a new column
                #     if(j2 < j){
                #       add_ind <- 0
                #     }else{ #j2 > j
                #       add_ind <- 1
                #     }
                #
                #     BtB_z_new[firstcolindtrees_z[j]-1 + addednodes[1],
                #               firstcolindtrees_z[j2]-1 + col_ind + add_ind] <-   sum( (new_tree_z$node_indices == addednodes[1])&
                #                                                                                 ( temptree_z$node_indices== terminal_nodes_j[col_ind] )  )
                #
                #     BtB_z_new[  firstcolindtrees_z[j2]-1 + col_ind + add_ind,
                #                 firstcolindtrees_z[j]-1 + addednodes[1]] <- BtB_z_new[1,1 +  firstcolindtrees_z[j2]-1 + col_ind]
                #
                #     BtB_z_new[firstcolindtrees_z[j]-1 + addednodes[2],
                #               firstcolindtrees_z[j2]-1 + col_ind + add_ind] <-   sum( (new_tree_z$node_indices == addednodes[2])&
                #                                                                                ( temptree_z$node_indices== terminal_nodes_j[col_ind] )  )
                #
                #     BtB_z_new[  firstcolindtrees_z[j2]-1 + col_ind + add_ind,
                #                 firstcolindtrees_z[j]-1 + addednodes[2]] <- BtB_z_new[2,1 +  firstcolindtrees_z[j2]-1 + col_ind]
                #
                #   }
                # }

                # firstcolindtrees_z_new[(j+1):n.trees_censoring]<- firstcolindtrees_z_new[(j+1):n.trees_censoring] + 1

              }
            }

            if(j < n.trees_censoring){
              firstcolindtrees_z_new[(j+1):n.trees_censoring]<- firstcolindtrees_z_new[(j+1):n.trees_censoring] + 1
            }
          }

          # if(any(dim(BtB_z_new_c) != dim(BtB_z_c)+1 )){
          #   stop("any(dim(BtB_z_new_c) != dim(BtB_z_c) -1 )")
          # }
          #
          # if(any(dim(BtB_z_new_u) != dim(BtB_z_u)+1 )){
          #   stop("any(dim(BtB_z_new_u) != dim(BtB_z_u) -1)")
          # }
          #
          # if((ncol(binmat_all_z_new) != ncol(binmat_all_z)+1  )){
          #
          #
          #   print("dim(BtB_z_new_u) = ")
          #   print(dim(BtB_z_new_u))
          #
          #   print("dim(BtB_z_u) = ")
          #   print(dim(BtB_z_u))
          #
          #   print("dim(binmat_all_z_new) = ")
          #   print(dim(binmat_all_z_new))
          #   print("dim(binmat_all_z) = ")
          #   print(dim(binmat_all_z))
          #   print("type_z = ")
          #   print(type_z)
          #
          #   stop("(ncol(binmat_all_z_new) != ncol(binmat_all_z)+1 )")
          # }

        } else{
          if (type_z == "prune") {
          # var_count_z[curr_trees_z[[j]]$var] <- var_count_z[curr_trees_z[[j]]$var] - 1
          terminal_nodes_old = which(as.numeric(curr_tree_z$tree_matrix[,'terminal']) == 1)
          terminal_nodes_new = which(as.numeric(new_tree_z$tree_matrix[,'terminal']) == 1)


          firstcolindtrees_z_new <- firstcolindtrees_z
          # if(   new_tree_z$pruned_parent ==-1   ){
          if(   length(terminal_nodes_old) == length(terminal_nodes_new)   ){
            binmat_all_z_new <- binmat_all_z
            # BtB_z_new <- BtB_z
            BtB_z_new_u <- BtB_z_u
            BtB_z_new_c <- BtB_z_c
            # no pruning, do not edit firstcolindtrees_z_new
          }else{
            # delete column corresponding to pruned nodes and create column corresponding to parent node (does ordering matter?)
            # yes, ordering matters because otherwise would not be able to find columns to delete or grow in future steps
            # new_tree_z$pruned_parent

            # # perhaps more efficient to just consider the difference
            # removednodes <- terminal_nodes_old[which(!( terminal_nodes_old %in% terminal_nodes_new ) )]
            # addednode <- terminal_nodes_new[which(!( terminal_nodes_new %in% terminal_nodes_old ) )]

            addednode <- which(terminal_nodes_new == new_tree_z$pruned_parent) # new terminal node is parent of removed nodes

            # if(length(addednode) == 0){
            #   print("terminal_nodes_new = ")
            #   print(terminal_nodes_new)
            #   print("new_tree_z = ")
            #   print(new_tree_z)
            #   stop("line 1813 length(addednode) == 0")
            # }

            # removed nodes were children of parent node
            removednodes <- which(terminal_nodes_old %in%  which(curr_tree_z$tree_matrix[ , 'parent'] == new_tree_z$pruned_parent))

            # removednodes_rowind <- terminal_nodes_old[removednodes]
            # addednode_rowind <- terminal_nodes_new[addednode]

            newparentbin <- binmat_all_z[,firstcolindtrees_z[j]-1+ removednodes[1], drop = FALSE ] +
              binmat_all_z[,firstcolindtrees_z[j]-1+ removednodes[2], drop = FALSE ]

            if(any(is.na(newparentbin))){
              print("newparentbin = ")
              print(newparentbin)
              stop("Line 2345 any(is.na(newparentbin))")
            }


            # if(  !is.symmetric(BtB_z_u)){
            if( any(abs(BtB_z_u - t(BtB_z_u))> 0.001)){

              stop("line 2433 !is.symmetric(BtB_z_u) ")
            }

            binmat_all_z_new <- binmat_all_z[, -(firstcolindtrees_z[j]-1+ removednodes) , drop = FALSE ]

            # BtB_z_new <- BtB_z[-(firstcolindtrees_z[j]-1+ removednode), -(firstcolindtrees_z[j]-1+ removednode) ]

            BtB_z_new_u <- BtB_z_u[-(firstcolindtrees_z[j]-1+ removednodes), -(firstcolindtrees_z[j]-1+ removednodes), drop = FALSE  ]
            BtB_z_new_c <- BtB_z_c[-(firstcolindtrees_z[j]-1+ removednodes), -(firstcolindtrees_z[j]-1+ removednodes) , drop = FALSE ]


            # if(  !is.symmetric(BtB_z_new_u)){
            if( any(abs(BtB_z_new_u - t(BtB_z_new_u))> 0.001)){

              stop("line 2439 !is.symmetric(BtB_z_new_u) ")
            }

            if(firstcolindtrees_z[j]-1 + addednode-1 == 0 ){

              # print("line 1839")
              binmat_all_z_new <- cbind(#binmat_all_z_new[, 1:(firstcolindtrees_z[j]-1 + addednode-1 ) ],
                newparentbin,
                binmat_all_z_new[, (firstcolindtrees_z[j]-1 + addednode ):ncol(binmat_all_z_new) , drop = FALSE ])


              # BtB_z_new <- cbind(rep(0, nrow(BtB_z_new)),  BtB_z_new[, (firstcolindtrees_z[j]-1 + addednode ):ncol(BtB_z_new) ])
              # BtB_z_new <- rbind(rep(0, ncol(BtB_z_new)), BtB_z_new[, (firstcolindtrees_z[j]-1 + addednode ):nrow(BtB_z_new) ])
              #
              # BtB_z_new[1,1] <- sum(new_tree_z$node_indices == addednode)
              #
              # BtB_z_new[ 1 ,-c( 1 )] <- crossprod((binmat_all_z_new[ , 1 ]) , binmat_all_z_new[ , - c(1) ])
              #
              # BtB_z_new[-c( 1),1] <- t(BtB_z_new[1,-c( 1 )])


              BtB_z_new_u <- cbind(rep(0, nrow(BtB_z_new_u)),  BtB_z_new_u[, (firstcolindtrees_z[j]-1 + addednode ):ncol(BtB_z_new_u), drop = FALSE  ])
              BtB_z_new_u <- rbind(rep(0, ncol(BtB_z_new_u)),
                                   BtB_z_new_u[(firstcolindtrees_z[j]-1 + addednode ):nrow(BtB_z_new_u), , drop = FALSE ])

              BtB_z_new_u[1,1] <- sum(newparentbin[uncens_inds]^2) # sum(new_tree_z$node_indices[uncens_inds] == addednode_rowind)

              BtB_z_new_u[ 1 ,-c( 1 )] <- crossprod((binmat_all_z_new[uncens_inds , 1 , drop = FALSE ]) ,
                                                    binmat_all_z_new[uncens_inds , - c(1) , drop = FALSE ])

              BtB_z_new_u[-c( 1),1] <- t(BtB_z_new_u[1,-c( 1 ), drop = FALSE])



              BtB_z_new_c <- cbind(rep(0, nrow(BtB_z_new_c)),  BtB_z_new_c[, (firstcolindtrees_z[j]-1 + addednode ):ncol(BtB_z_new_c), drop = FALSE  ])
              BtB_z_new_c <- rbind(rep(0, ncol(BtB_z_new_c)),
                                   BtB_z_new_c[(firstcolindtrees_z[j]-1 + addednode ):nrow(BtB_z_new_c), , drop = FALSE])

              BtB_z_new_c[1,1] <- sum(newparentbin[cens_inds]^2)  # sum(new_tree_z$node_indices[cens_inds] == addednode_rowind)

              BtB_z_new_c[ 1 ,-c( 1 )] <- crossprod((binmat_all_z_new[cens_inds , 1 , drop = FALSE ]) ,
                                                    binmat_all_z_new[cens_inds , - c(1) , drop = FALSE ])

              BtB_z_new_c[-c( 1),1] <- t(BtB_z_new_c[1,-c( 1 ), drop = FALSE])

              # firstcolindtrees_z_new[(j+1):n.trees_censoring]<- firstcolindtrees_z_new[(j+1):n.trees_censoring] - 1


              # if(any((binmat_all_z_new) ==0 )){
              #   print("which(binmat_all_z_new ==0, arr.ind = TRUE) = ")
              #   print(which(binmat_all_z_new ==0, arr.ind = TRUE))
              #
              #   print("binmat_all_z_new = ")
              #   print(binmat_all_z_new)
              #   stop("Line 2576 any((binmat_all_z_new) ==0 )))")
              # }
              #
              # if(any((BtB_z_new_u) ==0 )){
              #   print("which(BtB_z_new_u ==0, arr.ind = TRUE) = ")
              #   print(which(BtB_z_new_u ==0, arr.ind = TRUE))
              #
              #   print("BtB_z_new_u = ")
              #   print(BtB_z_new_u)
              #   stop("Line 2585 any((BtB_z_new_u) ==0 )))")
              # }

              # if(  !is.symmetric(BtB_z_new_u)){
              if( any(abs(BtB_z_new_u - t(BtB_z_new_u))> 0.001)){

                stop("line 2469 !is.symmetric(BtB_z_new_u) ")
              }

            }else{
              if(firstcolindtrees_z[j]-1 + addednode-1 == ncol(binmat_all_z_new) ){
                binmat_all_z_new <- cbind(binmat_all_z_new[, 1:(firstcolindtrees_z[j]-1 + addednode-1 ) , drop = FALSE ],
                                          newparentbin#,
                                          #binmat_all_z_new[, (firstcolindtrees_z[j]-1 + addednode ):ncol(binmat_all_z_new) ]
                )

                # BtB_z_new <- cbind( BtB_z_new[, 1:(firstcolindtrees_z[j]-1 + addednode-1 ) ],
                #                     rep(0, nrow(BtB_z_new)))
                # BtB_z_new <- rbind(BtB_z_new[, 1:(firstcolindtrees_z[j]-1 + addednodes[1]-1 ) ],
                #                    rep(0, ncol(BtB_z_new)))
                #
                # BtB_z_new[ncol(BtB_z_new),ncol(BtB_z_new)] <- sum(new_tree_z$node_indices == addednode)
                #
                # BtB_z_new[c(ncol(BtB_z_new) ),-c(ncol(BtB_z_new) )] <-
                #   crossprod((binmat_all_z_new[ ,c(ncol(BtB_z_new) ) ]) , binmat_all_z_new[ , - c(ncol(BtB_z_new) ) ])
                #
                # BtB_z_new[-c(ncol(BtB_z_new) ),c(ncol(BtB_z_new) )] <- t(BtB_z_new[c(ncol(BtB_z_new) ), - c(ncol(BtB_z_new) )])


                BtB_z_new_u <- cbind( BtB_z_new_u[, 1:(firstcolindtrees_z[j]-1 + addednode-1 ), drop = FALSE  ],
                                      rep(0, nrow(BtB_z_new_u)))
                BtB_z_new_u <- rbind(BtB_z_new_u[1:(firstcolindtrees_z[j]-1 + addednode-1 ), , drop = FALSE],
                                     rep(0, ncol(BtB_z_new_u)))

                BtB_z_new_u[ncol(BtB_z_new_u),ncol(BtB_z_new_u)] <- sum(newparentbin[uncens_inds]^2) # sum(new_tree_z$node_indices[uncens_inds] == addednode_rowind)

                BtB_z_new_u[c(ncol(BtB_z_new_u) ),-c(ncol(BtB_z_new_u) )] <-
                  crossprod((binmat_all_z_new[uncens_inds , c(ncol(BtB_z_new_u) ), drop = FALSE  ]) ,
                            binmat_all_z_new[uncens_inds , - c(ncol(BtB_z_new_u) ), drop = FALSE  ])

                BtB_z_new_u[-c(ncol(BtB_z_new_u) ),c(ncol(BtB_z_new_u) )] <- t(BtB_z_new_u[c(ncol(BtB_z_new_u) ),
                                                                                           - c(ncol(BtB_z_new_u) ),
                                                                                           drop = FALSE ])


                BtB_z_new_c <- cbind( BtB_z_new_c[, 1:(firstcolindtrees_z[j]-1 + addednode-1 ), drop = FALSE  ],
                                      rep(0, nrow(BtB_z_new_c)))
                BtB_z_new_c <- rbind(BtB_z_new_c[ 1:(firstcolindtrees_z[j]-1 + addednode-1 ), , drop = FALSE],
                                     rep(0, ncol(BtB_z_new_c)))

                BtB_z_new_c[ncol(BtB_z_new_c),ncol(BtB_z_new_c)] <- sum(newparentbin[cens_inds]^2) # sum(new_tree_z$node_indices[cens_inds] == addednode_rowind)

                BtB_z_new_c[c(ncol(BtB_z_new_c) ),-c(ncol(BtB_z_new_c) )] <-
                  crossprod((binmat_all_z_new[cens_inds ,c(ncol(BtB_z_new_c) ), drop = FALSE  ]) ,
                            binmat_all_z_new[cens_inds , - c(ncol(BtB_z_new_c) ), drop = FALSE  ])

                BtB_z_new_c[-c(ncol(BtB_z_new_c) ),c(ncol(BtB_z_new_c) )] <- t(BtB_z_new_c[c(ncol(BtB_z_new_c) ),
                                                                                           - c(ncol(BtB_z_new_c) ),
                                                                                           drop = FALSE ])



                # if(any((binmat_all_z_new) ==0 )){
                #   print("which(binmat_all_z_new ==0, arr.ind = TRUE) = ")
                #   print(which(binmat_all_z_new ==0, arr.ind = TRUE))
                #
                #   print("binmat_all_z_new = ")
                #   print(binmat_all_z_new)
                #   stop("Line 2653 any((binmat_all_z_new) ==0 )))")
                # }
                #
                # if(any((BtB_z_new_u) ==0 )){
                #   print("which(BtB_z_new_u ==0, arr.ind = TRUE) = ")
                #   print(which(BtB_z_new_u ==0, arr.ind = TRUE))
                #
                #   print("BtB_z_new_u = ")
                #   print(BtB_z_new_u)
                #   stop("Line 2662 any((BtB_z_new_u) ==0 )))")
                # }

                # if(  !is.symmetric(BtB_z_new_u)){
                if( any(abs(BtB_z_new_u - t(BtB_z_new_u))> 0.001)){

                  stop("line 2519 !is.symmetric(BtB_z_new_u) ")
                }
                # firstcolindtrees_z_new unchanged
              }else{

                binmat_all_z_new <- cbind(binmat_all_z_new[, 1:(firstcolindtrees_z[j]-1 + addednode-1 ), drop = FALSE  ],
                                          newparentbin,
                                          binmat_all_z_new[, (firstcolindtrees_z[j]-1 + addednode ):ncol(binmat_all_z_new), drop = FALSE  ])

                # BtB_z_new <- cbind( BtB_z_new[, 1:(firstcolindtrees_z[j]-1 + addednode-1 ) ],
                #                     rep(0, nrow(BtB_z_new)),
                #                     BtB_z_new[, (firstcolindtrees_z[j]-1 + addednode ):ncol(binmat_all_z_new)  ])
                # BtB_z_new <- rbind(BtB_z_new[ 1:(firstcolindtrees_z[j]-1 + addednode-1 ), ],
                #                    rep(0, ncol(BtB_z_new)),
                #                    BtB_z_new[, (firstcolindtrees_z[j]-1 + addednode ):nrow(binmat_all_z_new)  ])
                #
                # BtB_z_new[firstcolindtrees_z[j]-1 + addednode,
                #           firstcolindtrees_z[j]-1 + addednode] <- sum(new_tree_z$node_indices == addednode)
                #
                # BtB_z_new[firstcolindtrees_z[j]-1 + addednode,
                #           - c(firstcolindtrees_z[j]-1 + addednode)] <-
                #   crossprod((binmat_all_z_new[ ,firstcolindtrees_z[j]-1 + addednode ]) , binmat_all_z_new[ , - (firstcolindtrees_z[j]-1 + addednode) ])
                #
                # BtB_z_new[-c( firstcolindtrees_z[j]-1 + addednode),
                #           firstcolindtrees_z[j]-1 + addednode] <- t(BtB_z_new[firstcolindtrees_z[j]-1 + addednode,
                #                                                                     - c( firstcolindtrees_z[j]-1 + addednode)])

                BtB_z_new_u <- cbind( BtB_z_new_u[, 1:(firstcolindtrees_z[j]-1 + addednode-1 ), drop = FALSE  ],
                                      rep(0, nrow(BtB_z_new_u)),
                                      BtB_z_new_u[, (firstcolindtrees_z[j]-1 + addednode ):ncol(BtB_z_new_u), drop = FALSE   ])
                BtB_z_new_u <- rbind(BtB_z_new_u[1:(firstcolindtrees_z[j]-1 + addednode-1 ), , drop = FALSE],
                                     rep(0, ncol(BtB_z_new_u)),
                                     BtB_z_new_u[ (firstcolindtrees_z[j]-1 + addednode ):nrow(BtB_z_new_u), , drop = FALSE ])


                # if(  !is.symmetric(BtB_z_new_u)){
                if( any(abs(BtB_z_new_u - t(BtB_z_new_u))> 0.001)){

                  stop("line 2579 !is.symmetric(BtB_z_new_u) ")
                }

                BtB_z_new_u[firstcolindtrees_z[j]-1 + addednode,
                            firstcolindtrees_z[j]-1 + addednode] <- sum(newparentbin[uncens_inds]^2) # sum(new_tree_z$node_indices[uncens_inds] == addednode_rowind)

                BtB_z_new_u[firstcolindtrees_z[j]-1 + addednode,
                            - c(firstcolindtrees_z[j]-1 + addednode)] <-
                  crossprod((binmat_all_z_new[ uncens_inds,firstcolindtrees_z[j]-1 + addednode, drop = FALSE  ]) ,
                            binmat_all_z_new[ uncens_inds, - (firstcolindtrees_z[j]-1 + addednode), drop = FALSE  ])

                BtB_z_new_u[-c( firstcolindtrees_z[j]-1 + addednode),
                            firstcolindtrees_z[j]-1 + addednode] <- t(BtB_z_new_u[firstcolindtrees_z[j]-1 + addednode,
                                                                                  - c( firstcolindtrees_z[j]-1 + addednode), drop = FALSE ])


                BtB_z_new_c <- cbind( BtB_z_new_c[, 1:(firstcolindtrees_z[j]-1 + addednode-1 ) , drop = FALSE ],
                                      rep(0, nrow(BtB_z_new_c)),
                                      BtB_z_new_c[, (firstcolindtrees_z[j]-1 + addednode ):ncol(BtB_z_new_c) , drop = FALSE  ])
                BtB_z_new_c <- rbind(BtB_z_new_c[1:(firstcolindtrees_z[j]-1 + addednode-1 ) , , drop = FALSE],
                                     rep(0, ncol(BtB_z_new_c)),
                                     BtB_z_new_c[(firstcolindtrees_z[j]-1 + addednode ):nrow(BtB_z_new_c) , , drop = FALSE ])

                BtB_z_new_c[firstcolindtrees_z[j]-1 + addednode,
                            firstcolindtrees_z[j]-1 + addednode] <- sum(newparentbin[cens_inds]^2) # sum(new_tree_z$node_indices[cens_inds] == addednode_rowind)

                BtB_z_new_c[firstcolindtrees_z[j]-1 + addednode,
                            - c(firstcolindtrees_z[j]-1 + addednode)] <-
                  crossprod((binmat_all_z_new[ cens_inds,firstcolindtrees_z[j]-1 + addednode , drop = FALSE ]) ,
                            binmat_all_z_new[ cens_inds, - (firstcolindtrees_z[j]-1 + addednode), drop = FALSE  ])

                BtB_z_new_c[-c( firstcolindtrees_z[j]-1 + addednode),
                            firstcolindtrees_z[j]-1 + addednode] <- t(BtB_z_new_c[firstcolindtrees_z[j]-1 + addednode,
                                                                                  - c( firstcolindtrees_z[j]-1 + addednode), drop = FALSE ])

                # if(any((binmat_all_z_new) ==0 )){
                #   print("which(binmat_all_z_new ==0, arr.ind = TRUE) = ")
                #   print(which(binmat_all_z_new ==0, arr.ind = TRUE))
                #
                #   print("binmat_all_z_new = ")
                #   print(binmat_all_z_new)
                #   stop("Line 2747 any((binmat_all_z_new) ==0 )))")
                # }
                #
                # if(any((BtB_z_new_u) ==0 )){
                #   print("which(BtB_z_new_u ==0, arr.ind = TRUE) = ")
                #   print(which(BtB_z_new_u ==0, arr.ind = TRUE))
                #
                #   print("BtB_z_new_u = ")
                #   print(BtB_z_new_u)
                #   stop("Line 2756 any((BtB_z_new_u) ==0 )))")
                # }

                # if(  !is.symmetric(BtB_z_new_u)){
                if( any(abs(BtB_z_new_u - t(BtB_z_new_u))> 0.001)){

                  stop("line 2589 !is.symmetric(BtB_z_new_u) ")
                }
                # firstcolindtrees_z_new[(j+1):n.trees_censoring]<- firstcolindtrees_z_new[(j+1):n.trees_censoring] - 1

              }
            }
            # child_left = as.numeric(old_tree_z$tree_matrix[new_tree_z$pruned_parent, 'child_left'])
            # child_right = as.numeric(old_tree_z$tree_matrix[new_tree_z$pruned_parent, 'child_right'])

            if(j < n.trees_censoring){
              firstcolindtrees_z_new[(j+1):n.trees_censoring]<- firstcolindtrees_z_new[(j+1):n.trees_censoring] - 1
            }

          }

        } else { # change step
          firstcolindtrees_z_new <- firstcolindtrees_z

          if(new_tree_z$var[1] != 0){ # What if change step returned $var equal to c(0,0) ????
            # var_count_z[curr_trees_z[[j]]$var[1]] <- var_count_z[curr_trees_z[[j]]$var[1]] - 1
            # var_count_z[curr_trees_z[[j]]$var[2]] <- var_count_z[curr_trees_z[[j]]$var[2]] + 1

            binmat_all_z_new <- binmat_all_z
            # BtB_z_new <- BtB_z
            BtB_z_new_u <- BtB_z_u
            BtB_z_new_c <- BtB_z_c

            # find column of changed node
            # there is probably a more efficient way of doing this
            # changednodes <- sort(unique(new_tree_z$node_indices[which(new_tree_z$node_indices !=  old_tree_z$node_indices)]))
            changednodes_rowind <- sort(get_children(new_tree_z$tree_matrix, new_tree_z$node_to_change))

            terminal_nodes_new = which(as.numeric(new_tree_z$tree_matrix[,'terminal']) == 1)
            changednodes <- sort(which(terminal_nodes_new %in% changednodes_rowind))

            # terminal_nodes_old = which(as.numeric(curr_tree_z$tree_matrix[,'terminal']) == 1)

            # obtain old splitting variable and splitting point to calculate old gating function
            split_node_ind <- curr_tree_z$tree_matrix[changednodes_rowind[1],'parent']
            split_var_old <- curr_tree_z$tree_matrix[split_node_ind, 'split_variable']
            split_value_old <- curr_tree_z$tree_matrix[split_node_ind, 'split_value']
            # calculate the gating function for all observations
            # gat_func_psi_old <- 1/(1 + exp(- (w.train[,split_var] - split_value)/tau_vec_censoring[j] ) )
            # gat_func_psi_old <- gating_func_logistic((w.train[,split_var_old] - split_value_old)/tau_vec_censoring[j] )
            gat_func_psi_old <- plogis((w.train[,split_var_old] - split_value_old)/tau_vec_censoring[j] )

            # obtain new splitting variable and splitting point to calculate new gating function
            #
            split_node_ind <- new_tree_z$tree_matrix[changednodes_rowind[1],'parent']
            split_var <- new_tree_z$tree_matrix[split_node_ind, 'split_variable']
            split_value <- new_tree_z$tree_matrix[split_node_ind, 'split_value']
            # calculate the gating function for all observations
            # gat_func_psi_new <- 1/(1 + exp(- (w.train[,split_var] - split_value)/tau_vec_censoring[j] ) )
            # gat_func_psi_new <- gating_func_logistic((w.train[,split_var] - split_value)/tau_vec_censoring[j] )
            gat_func_psi_new <- plogis((w.train[,split_var] - split_value)/tau_vec_censoring[j] )




            if(any(is.na(gat_func_psi_new))){
              print("gat_func_psi_new = ")
              print(gat_func_psi_new)

              print("w.train[,split_var] = ")
              print(w.train[,split_var])

              print("tau_vec_censoring[j] = ")
              print(tau_vec_censoring[j])

              print("split_var = ")
              print(split_var)

              print("split_value = ")
              print(split_value)

              print("changednodes = ")
              print(changednodes)

              print("new_tree_z$tree_matrix = ")
              print(new_tree_z$tree_matrix)


              stop("Line 2561. any(is.na(gat_func_psi_new))")
            }
            if(any(is.na(gat_func_psi_old))){
              print("gat_func_psi_old = ")
              print(gat_func_psi_old)
              stop("Line 2569. any(is.na(gat_func_psi_old))")
            }
            # just replace the two columns for the relevant terminal nodes
            # newnodesbin <- matrix(0, nrow(binmat_all_z),2)
            # newnodesbin[new_tree_z$node_indices == changednodes_rowind[1],1] <- rep(1, sum(new_tree_z$node_indices == changednodes_rowind[1]))
            # newnodesbin[new_tree_z$node_indices == changednodes_rowind[2],2] <- rep(1, sum(new_tree_z$node_indices == changednodes_rowind[2]))

            newnodesbin <- matrix(0, nrow(binmat_all_z),length(changednodes))
            # newnodesbin[,1] <- binmat_all_z_new[,firstcolindtrees_z[j]-1 + changednodes[1]]*
            #   gating_func_logistic_plogis_ratio((w.train[,split_var] - split_value)/tau_vec_censoring[j],
            #                                     (w.train[,split_var_old] - split_value_old)/tau_vec_censoring[j])
            #   # gat_func_psi_new/gat_func_psi_old
            # newnodesbin[,2] <- binmat_all_z_new[,firstcolindtrees_z[j]-1 + changednodes[2]]*
            #   gating_func_logistic_plogis_ratio(-1*(w.train[,split_var] - split_value)/tau_vec_censoring[j],
            #                                     -1*(w.train[,split_var_old] - split_value_old)/tau_vec_censoring[j])
              # (1 - gat_func_psi_new)/(1-gat_func_psi_old)

            # if(any(is.na(newnodesbin))){
              newnodesbin <- matrix(0, nrow(binmat_all_z),length(changednodes))
              newnodesbin[,1] <- exp(log(binmat_all_z_new[,firstcolindtrees_z[j]-1 + changednodes[1]]) +
                gating_func_logistic_plogis_ratio_logdiff((w.train[,split_var] - split_value)/tau_vec_censoring[j],
                                                  (w.train[,split_var_old] - split_value_old)/tau_vec_censoring[j]))
              # gat_func_psi_new/gat_func_psi_old
              newnodesbin[,2] <- exp(log(binmat_all_z_new[,firstcolindtrees_z[j]-1 + changednodes[2]])+
                                       gating_func_logistic_plogis_ratio_logdiff(-1*(w.train[,split_var] - split_value)/tau_vec_censoring[j],
                                                  -1*(w.train[,split_var_old] - split_value_old)/tau_vec_censoring[j]))
            # }


            if(any(is.na(newnodesbin))){
              print("newnodesbin = ")
              print(newnodesbin)
              stop("Line 2586. any(is.na(newnodesbin))")
            }

            if(any(is.infinite(newnodesbin))){
              # anc_new = get_branch(new_tree_z[[j]])
              #
              # if(is.null(anc_new)){
              #   phi_matrix_new <- matrix(1, nrow = length(uncens_inds), ncol = 1)
              # }else{
              #   phi_matrix_new = phi_app(as.matrix(w.train), as.matrix(anc_new), tau_vec_censoring[j])
              # }
              #
              # first_col <- firstcolindtrees_z[j]
              # last_col <- 0
              # if(j==n.trees_censoring){
              #   last_col <- ncol(binmat_all_y)
              # }else{
              #   last_col <- firstcolindtrees_z[j+1]-1
              # }
              # #new bin mat
              # binmat_all_z_new <- binmat_all_z
              # binmat_all_z_new[,first_col:last_col] <- phi_matrix_new
              #
              # newnodesbin <- phi_matrix_new[,changednodes]

              stop("Line 2906 infinite in newnodesbin")
            }



            # if(  !is.symmetric(BtB_z_new_u)){
            if( any(abs(BtB_z_new_u - t(BtB_z_new_u))> 0.001)){
              print("max(BtB_z_new_u - t(BtB_z_new_u)) = ")
              print(max(BtB_z_new_u - t(BtB_z_new_u)))
              stop("line 2731 !is.symmetric(BtB_z_new_u) ")
            }

            if(length(changednodes)!=2){
              print("changednodes = ")
              print(changednodes)
             stop("length(changednodes)!=2")
            }

            BtB_z_new_u[firstcolindtrees_z[j]-1 + changednodes,
                        firstcolindtrees_z[j]-1 + changednodes] <- t(newnodesbin[uncens_inds,]) %*% newnodesbin[uncens_inds,]

            BtB_z_new_c[firstcolindtrees_z[j]-1 + changednodes,
                        firstcolindtrees_z[j]-1 + changednodes] <- t(newnodesbin[cens_inds,]) %*% newnodesbin[cens_inds,]

            # for(change_ind in 1:length(changednodes)){
            #   # newnodesbin[new_tree_z$node_indices == changednodes_rowind[change_ind],change_ind] <- rep(1, sum(new_tree_z$node_indices == changednodes_rowind[change_ind]))
            #
            #   BtB_z_new_u[firstcolindtrees_z[j]-1 + changednodes[change_ind],
            #               firstcolindtrees_z[j]-1 + changednodes[change_ind]] <- sum(newnodesbin[uncens_inds,change_ind]^2) # sum(new_tree_z$node_indices[uncens_inds] == changednodes_rowind[change_ind])
            #
            #   BtB_z_new_c[firstcolindtrees_z[j]-1 + changednodes[change_ind],
            #               firstcolindtrees_z[j]-1 + changednodes[change_ind]] <-  sum(newnodesbin[cens_inds,change_ind]^2) # sum(new_tree_z$node_indices[cens_inds] == changednodes_rowind[change_ind])
            #
            # }


            # if(  !is.symmetric(BtB_z_new_u)){
            if( any(abs(BtB_z_new_u - t(BtB_z_new_u))> 0.001)){
              print("which(BtB_z_new_u != t(BtB_z_new_u)) = ")
              print(which(BtB_z_new_u != t(BtB_z_new_u), arr.ind = TRUE))

              print("max(BtB_z_new_u - t(BtB_z_new_u)) = ")
              print(max(BtB_z_new_u - t(BtB_z_new_u)))
              print("firstcolindtrees_z[j]-1 + changednodes = ")
              print(firstcolindtrees_z[j]-1 + changednodes)
              stop("line 2749 !is.symmetric(BtB_z_new_u) ")
            }


            binmat_all_z_new[,firstcolindtrees_z[j]-1 + changednodes] <- newnodesbin

            # BtB_z_new[firstcolindtrees_z[j]-1 + changednodes[1],
            #           firstcolindtrees_z[j]-1 + changednodes[1]] <- sum(new_tree_z$node_indices == addednode[1])
            # BtB_z_new[firstcolindtrees_z[j]-1 + changednodes[2],
            #           firstcolindtrees_z[j]-1 + changednodes[2]] <- sum(new_tree_z$node_indices == addednode[2])
            #
            # BtB_z_new[firstcolindtrees_z[j]-1 + changednodes,
            #           - c(firstcolindtrees_z[j]-1 + changednodes)] <-
            #   crossprod((binmat_all_z_new[ ,firstcolindtrees_z[j]-1 + changednodes ]) , binmat_all_z_new[ , - (firstcolindtrees_z[j]-1 + changednodes) ])
            #
            # BtB_z_new[-c( firstcolindtrees_z[j]-1 + changednodes),
            #           firstcolindtrees_z[j]-1 + changednodes] <- t(BtB_z_new[firstcolindtrees_z[j]-1 + changednodes,
            #                                                               - c( firstcolindtrees_z[j]-1 + changednodes)])


            # BtB_z_new_u[firstcolindtrees_z[j]-1 + changednodes[1],
            #           firstcolindtrees_z[j]-1 + changednodes[1]] <- sum(new_tree_z$node_indices[uncens_inds] == changednodes_rowind[1])
            # BtB_z_new_u[firstcolindtrees_z[j]-1 + changednodes[2],
            #           firstcolindtrees_z[j]-1 + changednodes[2]] <- sum(new_tree_z$node_indices[uncens_inds] == changednodes_rowind[2])

            BtB_z_new_u[firstcolindtrees_z[j]-1 + changednodes,
                        - c(firstcolindtrees_z[j]-1 + changednodes)] <-
              crossprod((binmat_all_z_new[ uncens_inds ,firstcolindtrees_z[j]-1 + changednodes, drop = FALSE  ]) ,
                        binmat_all_z_new[ uncens_inds , - (firstcolindtrees_z[j]-1 + changednodes), drop = FALSE  ])

            BtB_z_new_u[-c( firstcolindtrees_z[j]-1 + changednodes),
                        firstcolindtrees_z[j]-1 + changednodes] <- t(BtB_z_new_u[firstcolindtrees_z[j]-1 + changednodes,
                                                                                 - c( firstcolindtrees_z[j]-1 + changednodes),
                                                                                 drop = FALSE ])

            # if(  !is.symmetric(BtB_z_new_u)){
            if( any(abs(BtB_z_new_u - t(BtB_z_new_u))> 0.001)){
              stop("line 2812 !is.symmetric(BtB_z_new_u) ")
            }


            # if(any((binmat_all_z_new) ==0 )){
            #   print("which(binmat_all_z_new ==0, arr.ind = TRUE) = ")
            #   print(which(binmat_all_z_new ==0, arr.ind = TRUE))
            #
            #   print("binmat_all_z_new = ")
            #   print(binmat_all_z_new)
            #   stop("Line 2971 any((binmat_all_z_new) ==0 )))")
            # }
            #
            # if(any((BtB_z_new_u) ==0 )){
            #   print("which(BtB_z_new_u ==0, arr.ind = TRUE) = ")
            #   print(which(BtB_z_new_u ==0, arr.ind = TRUE))
            #
            #   print("BtB_z_new_u = ")
            #   print(BtB_z_new_u)
            #   stop("Line 2980 any((BtB_z_new_u) ==0 )))")
            # }

            # BtB_z_new_c[firstcolindtrees_z[j]-1 + changednodes[1],
            #             firstcolindtrees_z[j]-1 + changednodes[1]] <- sum(new_tree_z$node_indices[cens_inds] == changednodes_rowind[1])
            # BtB_z_new_c[firstcolindtrees_z[j]-1 + changednodes[2],
            #             firstcolindtrees_z[j]-1 + changednodes[2]] <- sum(new_tree_z$node_indices[cens_inds] == changednodes_rowind[2])

            BtB_z_new_c[firstcolindtrees_z[j]-1 + changednodes,
                        - c(firstcolindtrees_z[j]-1 + changednodes)] <-
              crossprod((binmat_all_z_new[ cens_inds ,firstcolindtrees_z[j]-1 + changednodes , drop = FALSE ]) ,
                        binmat_all_z_new[ cens_inds , - (firstcolindtrees_z[j]-1 + changednodes) , drop = FALSE ])

            BtB_z_new_c[-c( firstcolindtrees_z[j]-1 + changednodes),
                        firstcolindtrees_z[j]-1 + changednodes] <- t(BtB_z_new_c[firstcolindtrees_z[j]-1 + changednodes,
                                                                                 - c( firstcolindtrees_z[j]-1 + changednodes), drop = FALSE ])


            # if(  !is.symmetric(BtB_z_new_u)){
            if( any(abs(BtB_z_new_u - t(BtB_z_new_u))> 0.001)){
              stop("line 2801 !is.symmetric(BtB_z_new_u) ")
            }

          }else{
            binmat_all_z_new <- binmat_all_z
            # BtB_z_new <- BtB_z
            BtB_z_new_u <- BtB_z_u
            BtB_z_new_c <- BtB_z_c
          }
          # if(any(dim(BtB_z_new_c) != dim(BtB_z_c) )){
          #   print("dim(BtB_z_new_c) = ")
          #   print(dim(BtB_z_new_c))
          #   print("dim(BtB_z_c) = ")
          #   print(dim(BtB_z_c))
          #
          #   stop("any(dim(BtB_z_new_c) != dim(BtB_z_c) )")
          # }
          #
          # if(any(dim(BtB_z_new_u) != dim(BtB_z_u) )){
          #   stop("any(dim(BtB_z_new_u) != dim(BtB_z_u) )")
          # }
          #
          # if(any(dim(binmat_all_z_new) != dim(binmat_all_z) )){
          #   stop("any(dim(binmat_all_z_new) != dim(binmat_all_z) )")
          # }
        }
      }


        # BtB_z_c <- crossprod(binmat_all_z[cens_inds, ])
        # BtB_z_u <- crossprod(binmat_all_z[uncens_inds, ])
        # BtB_z_new_c <- crossprod(binmat_all_z_new[cens_inds, ])
        # BtB_z_new_u <- crossprod(binmat_all_z_new[uncens_inds, ])

        #
        #   # check that binmat_all_z_new defined properly
        #
        #   for(j2 in 1:n.trees_censoring){
        #
        #     tempnodes <- new_trees_z[[j2]]$node_indices
        #     sorteduniqnodes <- sort(unique(tempnodes))
        #     for(node_ind in 1:length(unique(tempnodes))){
        #       nodeval <- sorteduniqnodes[node_ind]
        #       if(any(binmat_all_z_new[,firstcolindtrees_z_new[j2]-1 + node_ind] !=   1*(new_trees_z[[j2]]$node_indices == nodeval) )){
        #         print("j2 = ")
        #         print(j2)
        #         print("nodeval = ")
        #         print(nodeval)
        #         print("firstcolindtrees_z_new[j2] = ")
        #         print(firstcolindtrees_z_new[j2])
        #         print("tempnodes = ")
        #         print(tempnodes)
        #         print("node_ind = ")
        #         print(node_ind)
        #         print("sorteduniqnodes = ")
        #         print(sorteduniqnodes)
        #
        #         print("binmat_all_z_new[,firstcolindtrees_z_new[j2]-1 + node_ind] = ")
        #         print(binmat_all_z_new[,firstcolindtrees_z_new[j2]-1 + node_ind])
        #         print("1*(new_trees_z[[j2]]$node_indices == nodeval) = ")
        #         print(1*(new_trees_z[[j2]]$node_indices == nodeval))
        #
        #         stop("line 2254. any(binmat_all_z_new[,firstcolindtrees_z_new[j2]-1 + node_ind] !=   1*(new_trees_z[[j2]]$node_indices == nodeval) )")
        #       }
        #     }
        #   }
        #
        #
        #   stop("line 2217 max(abs((BtB_z_new_u + BtB_z_new_c) - crossprod(binmat_all_z_new))) > 0.0001 ")
        # }


        # print("line 2927. iter = ")
        # print(iter)

        # CURRENT TREE: compute the log of the marginalized likelihood + log of the tree prior
        z_resids <- z - offsetz #z_epsilon
        z_resids[uncens_inds] <- z[uncens_inds] - offsetz - (ystar[uncens_inds]  - mutemp_y)*gamma1/(phi1 + gamma1^2)

        z_uncens <- z_resids[uncens_inds]
        z_cens <- z_resids[cens_inds]

        current_partial_residuals <- z_resids

        weightz <- (gamma1^2 + phi1)/phi1
        weightstemp <- rep(1,n)
        weightstemp[uncens_inds] <- weightz


        # most efficient code would update cross-product subsetted by censored and uncensored observations here
        # (it is probably possible to calculate more quickly than in a matrix calculation)
        # also can apply kernel trick


        # if(j==1){
        # it should be possible to avoid duplication of this calculation

        if(linearterms){
          if(one_chol ==TRUE){
            reslisttemp = tree_full_conditional_z_marg_lin_savechol(curr_trees_z, #[[j]],
                                                                    current_partial_residuals,# sigma2,
                                                                    sigma2_mu_z,
                                                                    weightstemp,
                                                                    weightz,
                                                                    binmat_all_z, cens_inds, uncens_inds, BtB_z_u, BtB_z_c,
                                                                    wmat_train, Amean_p, invAvar_p)
            l_old_z = reslisttemp[[1]] + get_tree_prior(curr_trees_z[[j]], alpha_z, beta_z) # the priors for all unchanged trees will cancel out

            IR_old_z <- reslisttemp[[2]]
            S_j_old_z <- reslisttemp[[3]]
          }else{
            l_old_z = tree_full_conditional_z_marg_lin(curr_trees_z, #[[j]],
                                                       current_partial_residuals,# sigma2,
                                                       sigma2_mu_z,
                                                       weightstemp,
                                                       weightz,
                                                       binmat_all_z, cens_inds, uncens_inds, BtB_z_u, BtB_z_c,
                                                       wmat_train, Amean_p, invAvar_p) +
              get_tree_prior(curr_trees_z[[j]], alpha_z, beta_z) # the priors for all unchanged trees will cancel out
          }
        }else{
          if(one_chol ==TRUE){
            reslisttemp = tree_full_conditional_z_marg_savechol(curr_trees_z, #[[j]],
                                                                current_partial_residuals,# sigma2,
                                                                sigma2_mu_z,
                                                                weightstemp,
                                                                weightz,
                                                                binmat_all_z, cens_inds, uncens_inds, BtB_z_u, BtB_z_c)
            l_old_z = reslisttemp[[1]] + get_tree_prior(curr_trees_z[[j]], alpha_z, beta_z) # the priors for all unchanged trees will cancel out

            IR_old_z <- reslisttemp[[2]]
            S_j_old_z <- reslisttemp[[3]]
          }else{

            if(any(is.na(BtB_z_u))){

              print( "which(is.na(BtB_z_u),arr.ind = TRUE) = " )
              print( which(is.na(BtB_z_u),arr.ind = TRUE) )
              stop("any(is.na(BtB_z_u))")
            }

            # if(!is.symmetric(BtB_z_new_u)){
            if( any(abs(BtB_z_new_u - t(BtB_z_new_u))> 0.001)){
                print("!is.symmetric(BtB_z_new_u)")
            }
            # if(!is.symmetric(BtB_z_new_c)){
            if( any(abs(BtB_z_new_c - t(BtB_z_new_c))> 0.001)){
                print("!is.symmetric(BtB_z_new_c)")
            }

            l_old_z = tree_full_conditional_z_marg(curr_trees_z, #[[j]],
                                                   current_partial_residuals,# sigma2,
                                                   sigma2_mu_z,
                                                   weightstemp,
                                                   weightz,
                                                   binmat_all_z, cens_inds, uncens_inds, BtB_z_u, BtB_z_c) +
              get_tree_prior(curr_trees_z[[j]], alpha_z, beta_z) # the priors for all unchanged trees will cancel out
          }
        }
        # }



        # print("line 3022 iter = ")
        # print(iter)

        if((nrow(new_tree_z$tree_matrix) == nrow(curr_tree_z$tree_matrix) ) & (type_z != "change" )){
          alpha_MH <- 0
          # print("no good trees")
          # if(j==1){
          # it should be possible to avoid duplication of this calculation




        }else{



          if(linearterms){
            if(one_chol ==TRUE){
              # NEW TREE: compute the log of the marginalized likelihood + log of the tree prior
              reslisttemp = tree_full_conditional_z_marg_lin_savechol(new_trees_z, #[[j]],
                                                                      current_partial_residuals,# sigma2,
                                                                      sigma2_mu_z,
                                                                      weightstemp,
                                                                      weightz,
                                                                      binmat_all_z_new, cens_inds, uncens_inds, BtB_z_new_u, BtB_z_new_c,
                                                                      wmat_train, Amean_p, invAvar_p)
              l_new_z = reslisttemp[[1]] + get_tree_prior(new_trees_z[[j]], alpha_z, beta_z) # the priors for all unchanged trees will cancel out

              IR_new_z <- reslisttemp[[2]]
              S_j_new_z <- reslisttemp[[3]]
            }else{
              # NEW TREE: compute the log of the marginalized likelihood + log of the tree prior
              l_new_z = tree_full_conditional_z_marg_lin(new_trees_z, #[[j]],
                                                         current_partial_residuals,# sigma2,
                                                         sigma2_mu_z,
                                                         weightstemp,
                                                         weightz,
                                                         binmat_all_z_new, cens_inds, uncens_inds, BtB_z_new_u, BtB_z_new_c,
                                                         wmat_train, Amean_p, invAvar_p) + get_tree_prior(new_trees_z[[j]], alpha_z, beta_z) # the priors for all unchanged trees will cancel out

            }
          }else{
            if(one_chol ==TRUE){
              # NEW TREE: compute the log of the marginalized likelihood + log of the tree prior
              reslisttemp = tree_full_conditional_z_marg_savechol(new_trees_z, #[[j]],
                                                                  current_partial_residuals,# sigma2,
                                                                  sigma2_mu_z,
                                                                  weightstemp,
                                                                  weightz,
                                                                  binmat_all_z_new, cens_inds, uncens_inds, BtB_z_new_u, BtB_z_new_c)
              l_new_z = reslisttemp[[1]] + get_tree_prior(new_trees_z[[j]], alpha_z, beta_z) # the priors for all unchanged trees will cancel out

              IR_new_z <- reslisttemp[[2]]
              S_j_new_z <- reslisttemp[[3]]

            }else{

              if(any(is.na(BtB_z_new_u))){
                print("Line 2835")
                print( "which(is.na(BtB_z_new_u),arr.ind = TRUE) = " )
                print( which(is.na(BtB_z_new_u),arr.ind = TRUE) )
                stop("any(is.na(BtB_z_new_u))")
              }
              # if(!is.symmetric(BtB_z_new_u)){
              if( any(abs(BtB_z_new_u - t(BtB_z_new_u))> 0.001)){

                print("max(BtB_z_new_u - t(BtB_z_new_u)) = ")
                print(max(BtB_z_new_u - t(BtB_z_new_u)))
                stop("!is.symmetric(BtB_z_new_u)")
              }
              # if(!is.symmetric(BtB_z_new_c)){
              if( any(abs(BtB_z_new_c - t(BtB_z_new_c))> 0.001)){

                print("max(BtB_z_new_c - t(BtB_z_new_c)) = ")
                print(max(BtB_z_new_c - t(BtB_z_new_c)))
                stop("!is.symmetric(BtB_z_new_c)")
              }
              # NEW TREE: compute the log of the marginalized likelihood + log of the tree prior
              l_new_z = tree_full_conditional_z_marg(new_trees_z, #[[j]],
                                                     current_partial_residuals,# sigma2,
                                                     sigma2_mu_z,
                                                     weightstemp,
                                                     weightz,
                                                     binmat_all_z_new, cens_inds, uncens_inds,
                                                     BtB_z_new_u, BtB_z_new_c) +
                get_tree_prior(new_trees_z[[j]], alpha_z, beta_z) # the priors for all unchanged trees will cancel out
            }
          }

          alpha_MH = alpha_mh(l_new_z,l_old_z, curr_trees_z[[j]],new_trees_z[[j]], type_z)

        }


        # print("line 3116 iter = ")
        # print(iter)

        if(is.na(alpha_MH)){
          print("l_old_z = ")
          print(l_old_z)
          print("l_new_z = ")
          print(l_new_z)
          print("tree_full_conditional_z_marg(new_trees_z, #[[j]],
                                                     current_partial_residuals,# sigma2,
                                                     sigma2_mu_z,
                                                     weightstemp,
                                                     weightz,
                                                     binmat_all_z_new, cens_inds, uncens_inds, BtB_z_new_u, BtB_z_new_c) = ")


          print(tree_full_conditional_z_marg(new_trees_z, #[[j]],
                                             current_partial_residuals,# sigma2,
                                             sigma2_mu_z,
                                             weightstemp,
                                             weightz,
                                             binmat_all_z_new, cens_inds, uncens_inds, BtB_z_new_u, BtB_z_new_c))
          print("get_tree_prior(new_trees_z[[j]], alpha_z, beta_z) = ")
          print(get_tree_prior(new_trees_z[[j]], alpha_z, beta_z))
          print("curr_tree_z = ")
          print(curr_tree_z)
          print("new_tree_z = ")
          print(new_tree_z)
          print("get_tree_prior(curr_trees_z[[j]], alpha_z, beta_z) = ")
          print(get_tree_prior(curr_trees_z[[j]], alpha_z, beta_z))
          print(" get_tree_prior(new_trees_z[[j]], alpha_z, beta_z) = ")
          print( get_tree_prior(new_trees_z[[j]], alpha_z, beta_z))
          stop("line 2771 alpha_MH NA")
        }

        if(alpha_MH > runif(1)) {
          curr_trees_z[[j]] = new_trees_z[[j]]
          binmat_all_z <- binmat_all_z_new
          firstcolindtrees_z <- firstcolindtrees_z_new
          BtB_z_u <- BtB_z_new_u
          BtB_z_c <- BtB_z_new_c


          # if(  !is.symmetric(BtB_z_u)){
          if( any(abs(BtB_z_u - t(BtB_z_u))> 0.001)){
            stop("line 3103 !is.symmetric(BtB_z_u) ")
          }

          l_old_z <- l_new_z

          if( (one_chol == TRUE) ) {
            IR_old_z <- IR_new_z
            S_j_old_z <- S_j_new_z
          }

          # if(sparse){
          if (type_z == "grow") {
            var_count_z[curr_trees_z[[j]]$var] <- var_count_z[curr_trees_z[[j]]$var] + 1
          } else if (type_z == "prune") {
            var_count_z[curr_trees_z[[j]]$var] <- var_count_z[curr_trees_z[[j]]$var] - 1
          } else {
            if(curr_trees_z[[j]]$var[1]!=0){ # What if change step returned $var equal to c(0,0) ????
              var_count_z[curr_trees_z[[j]]$var[1]] <- var_count_z[curr_trees_z[[j]]$var[1]] - 1
              var_count_z[curr_trees_z[[j]]$var[2]] <- var_count_z[curr_trees_z[[j]]$var[2]] + 1
            }
          }
          # }
        }
        # type_z_prev <- type_z
      } # end loop over z trees


      BtB_z_c <- crossprod(binmat_all_z[cens_inds, ])
      BtB_z_u <- crossprod(binmat_all_z[uncens_inds, ])


      # print("line 3192 iter = ")
      # print(iter)

      # if(  !is.symmetric(BtB_z_u)){
      if( any(abs(BtB_z_u - t(BtB_z_u))> 0.001)){
        stop("line 3134 !is.symmetric(BtB_z_u) ")
      }

      if(mh_tau_bandwidth){
        # Compute the log of the marginalized likelihood and the log of the tau prior for the current tree

        # print("line 3203 iter = ")
        # print(iter)

        l_old_z2 = l_old_z  +
          log_tau_prior(tau_vec_censoring[j], tau_rate) + log(tau_vec_censoring[j])

        # print("line 3209 iter = ")
        # print(iter)

        # Calculate the new bandwidth using Random Walk
        # tau_new[[j]] = tau[[j]]*exp(runif(n = 1,min = -1,max = 1))
        tau_new = tau_vec_censoring[j]*(5^(runif(n = 1,min = -1,max = 1)))

        # print("line 3216 iter = ")
        # print(iter)

        anc_new = get_branch(curr_trees_z[[j]])
        # if(ncol(as.matrix(anc_new))==1){
        #   stop("Line 3169 (ncol(as.matrix(anc_new))==1")
        # }

        # print("line 3215 iter = ")
        # print(iter)

        if(is.null(anc_new)){
          phi_matrix_new <- matrix(1, nrow = nrow(w.train), ncol = 1)
        }else{
          phi_matrix_new = phi_app(as.matrix(w.train), as.matrix(anc_new), tau_new)
        }


        # print("line 3221 iter = ")
        # print(iter)

        first_col <- firstcolindtrees_z[j]
        last_col <- 0
        if(j==n.trees_censoring){
          last_col <- ncol(binmat_all_z)
        }else{
          last_col <- firstcolindtrees_z[j+1]-1
        }
        #new bin mat
        binmat_all_z_new <- binmat_all_z
        binmat_all_z_new[, first_col:last_col] <- phi_matrix_new
        # calculate new  B BtB etc
        BtB_y_new <- BtB_y

        BtB_z_new_u <- BtB_z_u
        BtB_z_new_c <- BtB_z_c

        # if(  !is.symmetric(BtB_z_new_u)){
        if( any(abs(BtB_z_new_u - t(BtB_z_new_u))> 0.001)){
          stop("line 3187 !is.symmetric(BtB_z_new_u) ")
        }


        # print("line 3246 iter = ")
        # print(iter)

        BtB_z_new_u[first_col:last_col, ] <- t(phi_matrix_new[uncens_inds,]) %*% binmat_all_z_new[uncens_inds,]
        BtB_z_new_u[ , first_col:last_col ] <- t(BtB_z_new_u[first_col:last_col, ])
        BtB_z_new_c[first_col:last_col, ] <- t(phi_matrix_new[cens_inds,]) %*% binmat_all_z_new[cens_inds,]
        BtB_z_new_c[ , first_col:last_col ] <- t(BtB_z_new_c[first_col:last_col, ])

        # print("line 3242 iter = ")
        # print(iter)

        # if(any((binmat_all_z_new) ==0 )){
        #   print("which(binmat_all_z_new ==0, arr.ind = TRUE) = ")
        #   print(which(binmat_all_z_new ==0, arr.ind = TRUE))
        #
        #   print("binmat_all_z_new = ")
        #   print(binmat_all_z_new)
        #   stop("Line 3416 any((binmat_all_z_new) ==0 )))")
        # }
        #
        # if(any((BtB_z_new_u) ==0 )){
        #   print("which(BtB_z_new_u ==0, arr.ind = TRUE) = ")
        #   print(which(BtB_z_new_u ==0, arr.ind = TRUE))
        #
        #   print("BtB_z_new_u = ")
        #   print(BtB_z_new_u)
        #   stop("Line 3425 any((BtB_z_new_u) ==0 )))")
        # }



        # if(  !is.symmetric(BtB_z_new_u)){
        if( any(abs(BtB_z_new_u - t(BtB_z_new_u))> 0.001)){

          print("which(BtB_z_new_u != t(BtB_z_new_u), arr.ind = TRUE) = ")
          print(which(BtB_z_new_u != t(BtB_z_new_u), arr.ind = TRUE))
          print("first_col:last_col = ")
          print(first_col:last_col)

          print("dim(phi_matrix_new[uncens_inds,]) = ")
          print(dim(phi_matrix_new[uncens_inds,]))

          print("dim(binmat_all_z_new[uncens_inds,]) = ")
          print(dim(binmat_all_z_new[uncens_inds,]))


          print("dim(BtB_z_new_u) = ")
          print(dim(BtB_z_new_u))

          print("t(phi_matrix_new[uncens_inds,]) %*% binmat_all_z_new[uncens_inds,] = ")
          print(t(phi_matrix_new[uncens_inds,]) %*% binmat_all_z_new[uncens_inds,])

          print("BtB_z_new_u[first_col:last_col, first_col:last_col] = ")
          print(BtB_z_new_u[first_col:last_col, first_col:last_col])

          stop("line 3199 !is.symmetric(BtB_z_new_u) ")
        }

        if(linearterms){
            # NEW TREE: compute the log of the marginalized likelihood + log of the tree prior
            l_new_z = tree_full_conditional_z_marg_lin(curr_trees_z, #[[j]],
                                                       current_partial_residuals,# sigma2,
                                                       sigma2_mu_z,
                                                       weightstemp,
                                                       weightz,
                                                       binmat_all_z_new, cens_inds, uncens_inds, BtB_z_new_u, BtB_z_new_c,
                                                       wmat_train, Amean_p, invAvar_p) +
              get_tree_prior(curr_trees_z[[j]], alpha_z, beta_z) # the priors for all unchanged trees will cancel out
        }else{
            # NEW TREE: compute the log of the marginalized likelihood + log of the tree prior
          if(any(is.na(BtB_z_new_u))){
            print("Line 2972")

            print( "which(is.na(BtB_z_new_u),arr.ind = TRUE) = " )
            print( which(is.na(BtB_z_new_u),arr.ind = TRUE) )
            stop("any(is.na(BtB_z_new_u))")
          }

          # if(!is.symmetric(BtB_z_new_u)){
          if( any(abs(BtB_z_new_u - t(BtB_z_new_u))> 0.001)){
              print("!is.symmetric(BtB_z_new_u)")
          }
          # if(!is.symmetric(BtB_z_new_c)){
          if( any(abs(BtB_z_new_c - t(BtB_z_new_c))> 0.001)){
              print("!is.symmetric(BtB_z_new_c)")
          }

          l_new_z = tree_full_conditional_z_marg(curr_trees_z, #[[j]],
                                                   current_partial_residuals,# sigma2,
                                                   sigma2_mu_z,
                                                   weightstemp,
                                                   weightz,
                                                   binmat_all_z_new, cens_inds, uncens_inds,
                                                 BtB_z_new_u, BtB_z_new_c) +
              get_tree_prior(curr_trees_z[[j]], alpha_z, beta_z) # the priors for all unchanged trees will cancel out

        }



        l_new_z2 = l_new_z + log_tau_prior(tau_new, tau_rate) + log(tau_new)


        # Here, the calculation of alpha doesn't depend on any transition probabilities
        a = exp(l_new_z2 - l_old_z2)

        if(is.na(a)| is.null(a)){
          print("l_new_z = ")
          print(l_new_z)
          print("l_new_z2 = ")
          print(l_new_z2)
          print("tau_new = ")
          print(tau_new)
          print("log_tau_prior(tau_new, tau_rate) = ")
          print(log_tau_prior(tau_new, tau_rate))
          print("log(tau_new) = ")
          print(log(tau_new))
          print("l_old_z2 = ")
          print(l_old_z2)
          print("log_tau_prior(tau_vec_censoring[j], tau_rate) = ")
          print(log_tau_prior(tau_vec_censoring[j], tau_rate))
          print("log(tau_vec_censoring[j]) = ")
          print(log(tau_vec_censoring[j]))

        }

        if(a > runif(1)) { # In case the alpha is bigger than a uniformly sampled value between zero and one
          tau_vec_censoring[j] = tau_new # The current bandwidth "becomes" the new bandwidth, if the latter is better
          binmat_all_z <- binmat_all_z_new
          # calculate new  B BtB etc
          BtB_y <- BtB_y_new

          BtB_z_u <- BtB_z_new_u
          BtB_z_c <- BtB_z_new_c


          # if(  !is.symmetric(BtB_z_u)){
          if( any(abs(BtB_z_u - t(BtB_z_u))> 0.001)){

            stop("line 3251 !is.symmetric(BtB_z_u) ")
          }
          # UPDATE B, BTB
          # UPDATE PARTIAL RESIDUALS (IF ANY?)
          # UPDATE YHAT
        }

      }



      # print("line 3365 iter = ")
      # print(iter)


      if( (one_chol == TRUE)| linearterms ) {
        if(linearterms){
          if(one_chol ==TRUE){

            mudrawlist_z = simulate_mu_weighted_all_z_fast_lin(curr_trees_z,
                                                               current_partial_residuals,
                                                               sigma2_mu_z,
                                                               weightstemp,
                                                               weightz,
                                                               binmat_all_z, cens_inds, uncens_inds, BtB_z_u, BtB_z_c, firstcolindtrees_z,
                                                               IR_old_z, S_j_old_z, wmat_train, Amean_p, invAvar_p)
          }else{
            mudrawlist_z = simulate_mu_weighted_all_z_lin(curr_trees_z,
                                                          current_partial_residuals,
                                                          sigma2_mu_z,
                                                          weightstemp,
                                                          weightz,
                                                          binmat_all_z, cens_inds, uncens_inds, BtB_z_u, BtB_z_c, firstcolindtrees_z, wmat_train, Amean_p, invAvar_p)
          }
        }else{
          mudrawlist_z = simulate_mu_weighted_all_z_fast(curr_trees_z,
                                                         current_partial_residuals,
                                                         sigma2_mu_z,
                                                         weightstemp,
                                                         weightz,
                                                         binmat_all_z, cens_inds, uncens_inds, BtB_z_u, BtB_z_c, firstcolindtrees_z,
                                                         IR_old_z, S_j_old_z)
        }
      }else{
        mudrawlist_z = simulate_mu_weighted_all_z(curr_trees_z,
                                                  current_partial_residuals,
                                                  sigma2_mu_z,
                                                  weightstemp,
                                                  weightz,
                                                  binmat_all_z, cens_inds, uncens_inds, BtB_z_u, BtB_z_c, firstcolindtrees_z)
      }

      curr_trees_z <- mudrawlist_z[[1]]
      new_trees_z <- curr_trees_z

      if(linearterms){
        mutemp_z <- cbind(wmat_train, binmat_all_z) %*% mudrawlist_z[[5]]
      }else{
        mutemp_z <- binmat_all_z %*% mudrawlist_z[[3]]
      }
      # # Updating BART predictions
      # current_fit = get_predictions(curr_trees_z[j], w.train, single_tree = TRUE)
      # mutemp_z = mutemp_z - tree_fits_store_z[,j] # subtract the old fit
      # mutemp_z = mutemp_z + current_fit # add the new fit
      # tree_fits_store_z[,j] = current_fit # update the new fit
      #


      # print("line 3420 iter = ")
      # print(iter)

    }else{  # z sample not marginalized code


      for (j in 1:n.trees_censoring) {
        current_partial_residuals = z_resids - mutemp_z + tree_fits_store_z[,j]


        # We need the new and old trees for the likelihoods
        new_trees_z <- curr_trees_z

        type_z = sample_move(curr_trees_z[[j]], i, 0, #n_burn
                             trans_prob)

        # Generate a new tree based on the current
        new_trees_z[[j]] <- update_tree(
          y = z_resids,
          X = w.train,
          type = type_z,
          curr_tree = curr_trees_z[[j]],
          node_min_size = node_min_size,
          s = s_z,
          max_bad_trees = max_bad_trees,
          splitting_rules = splitting_rules
        )



        # (c) Obtain the Metropolis-Hastings probability
        curr_tree_z <- curr_trees_z[[j]]
        new_tree_z <- new_trees_z[[j]]


        if((nrow(new_tree_z$tree_matrix) == nrow(curr_tree_z$tree_matrix) ) & (type_z != "change" )){
          alpha_MH <- 0
          # print("no good trees")
        }else{

          # if(j==1){
          # CURRENT TREE: compute the log of the marginalised likelihood + log of the tree prior
          l_old_z = tree_full_conditional_weighted(curr_trees_z[[j]],
                                                   current_partial_residuals,# sigma2,
                                                   sigma2_mu_z,
                                                   weightstemp) +
            get_tree_prior(curr_trees_z[[j]], alpha_z, beta_z)
          # }

          # NEW TREE: compute the log of the marginalised likelihood + log of the tree prior
          l_new_z = tree_full_conditional_weighted(new_trees_z[[j]],
                                                   current_partial_residuals,# sigma2,
                                                   sigma2_mu_z,
                                                   weightstemp) +
            get_tree_prior(new_trees_z[[j]], alpha_z, beta_z)

          alpha_MH = alpha_mh(l_new_z,l_old_z, curr_trees_z[[j]],new_trees_z[[j]], type_z)

        }


        if(is.na(alpha_MH)){
          print("l_old_z = ")
          print(l_old_z)
          print("l_new_z = ")
          print(l_new_z)

          print("curr_tree_z = ")
          print(curr_tree_z)

          print("new_tree_z = ")
          print(new_tree_z)

          print("get_tree_prior(curr_trees_z[[j]], alpha_z, beta_z) = ")
          print(get_tree_prior(curr_trees_z[[j]], alpha_z, beta_z))

          print(" get_tree_prior(new_trees_z[[j]], alpha_z, beta_z) = ")
          print( get_tree_prior(new_trees_z[[j]], alpha_z, beta_z))

          stop("line 3123. alpha_MH NA")

        }

        if(alpha_MH > runif(1)) {
          curr_trees_z[[j]] = new_trees_z[[j]]
          l_old_z <- l_new_z
          # if(sparse){

          if (type_z == "grow") {
            var_count_z[curr_trees_z[[j]]$var] <- var_count_z[curr_trees_z[[j]]$var] + 1
          } else if (type_z == "prune") {
            var_count_z[curr_trees_z[[j]]$var] <- var_count_z[curr_trees_z[[j]]$var] - 1
          } else {
            if(curr_trees_z[[j]]$var[1]!=0){ # What if change step returned $var equal to c(0,0) ????
              var_count_z[curr_trees_z[[j]]$var[1]] <- var_count_z[curr_trees_z[[j]]$var[1]] - 1
              var_count_z[curr_trees_z[[j]]$var[2]] <- var_count_z[curr_trees_z[[j]]$var[2]] + 1
            }
          }
          # }

        }

        curr_trees_z[[j]] = simulate_mu_weighted(curr_trees_z[[j]],
                                                 current_partial_residuals,
                                                 # sigma2,
                                                 sigma2_mu_z,
                                                 weightstemp)

        # Updating BART predictions
        current_fit = get_predictions(curr_trees_z[j], w.train, single_tree = TRUE)
        mutemp_z = mutemp_z - tree_fits_store_z[,j] # subtract the old fit
        mutemp_z = mutemp_z + current_fit # add the new fit

        mutemp_z_trees = mutemp_z_trees - tree_fits_store_z[,j] # subtract the old fit
        mutemp_z_trees = mutemp_z_trees + current_fit # add the new fit


        tree_fits_store_z[,j] = current_fit # update the new fit

      } # end loop over z trees

    } # end else (not marginalizing)



    # print("weightstemp = ")
    # print(weightstemp)

    # print("Line 737")
    # print("weightstemp = ")
    # print(weightstemp)
    #
    # print("gamma1 = ")
    # print(gamma1)
    #
    # print("phi1 = ")
    # print(phi1)

    # sampler_z$setWeights(weights = weightstemp)
    #
    # if(sparse){
    #   tempmodel <- sampler_z$model
    #   tempmodel@tree.prior@splitProbabilities <- s_z
    #   sampler_z$setModel(newModel = tempmodel)
    # }
    #
    # # print("Line 741")
    #
    # samplestemp_z <- sampler_z$run()
    #
    # mutemp_z <- samplestemp_z$train[,1]
    # mutemp_test_z <- samplestemp_z$test[,1]
    # # mutemp_test_z <- sampler_z$test[,1]#samplestemp_z$test[,1]
    # # mutemp_test_z <- sampler_z$predict(xdf_z_test)[,1]#samplestemp_z$test[,1]


    # if(sparse){
    #   tempcounts <- fcount(sampler_z$getTrees()$var)
    #   tempcounts <- tempcounts[tempcounts$x != -1, ]
    #   var_count_z <- rep(0, p_z)
    #   var_count_z[tempcounts$x] <- tempcounts$N
    # }

    # print("length(mutemp_test_z) = ")
    # print(length(mutemp_test_z))
    #
    # print("nrow(xdf_z_test) = ")
    # print(nrow(xdf_z_test))

    #update z_epsilon
    z_epsilon <- z - offsetz - mutemp_z

    mutemp_test_z <- get_predictions(curr_trees_z,
                                     w.test,
                                     single_tree = length(curr_trees_z) == 1)

    if(linearterms & !marginalize){
      mutemp_test_z_trees <- mutemp_test_z
      mutemp_test_z <- mutemp_test_z_trees + mutemp_test_z_lin
    }

    if(linearterms &  marginalize){
      mutemp_test_z <- mutemp_test_z + wmat_test %*% mudrawlist_z[[4]]
    }

    ####### draw sums of trees for y #######################################################

    # print("Draw y trees. iter = ")
    # print(iter)

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

    y_resids <- ystar[uncens_inds] - gamma1*(z[uncens_inds] - offsetz - mutemp_z[uncens_inds])
    sd_ydraw <- sqrt(phi1)

    # print("y_resids = ")
    #
    # print(y_resids)

    #set the response for draws of z trees
    # sampler_y$setResponse(y = y_resids)
    # #set the standard deviation
    # sampler_y$setSigma(sigma = sd_ydraw)
    #
    # if(sparse){
    #   tempmodel <- sampler_y$model
    #   tempmodel@tree.prior@splitProbabilities <- s_y
    #   sampler_y$setModel(newModel = tempmodel)
    # }
    #
    # samplestemp_y <- sampler_y$run()
    #
    # mutemp_y <- samplestemp_y$train[,1]
    # mutemp_test_y <- samplestemp_y$test[,1]


    if(marginalize){
      Btz <- crossprod(binmat_all_y, z_epsilon[uncens_inds])
      Btz_new <- Btz
      ztz <- crossprod(z_epsilon[uncens_inds])
      # print("2467. dim(ztz) = ")
      # print(dim(ztz))
      for (j in 1:n.trees_outcome) {
        # current_partial_residuals = y_resids - mutemp_y + tree_fits_store_y[,j]

        # We need the new and old trees for the likelihoods
        new_trees_y <- curr_trees_y

        type_y = sample_move(curr_trees_y[[j]], i, 0, #n_burn
                             trans_prob)

        # Generate a new tree based on the current
        new_trees_y[[j]] <- update_tree(
          y = y_uncens,
          X = x.train[uncens_inds,],
          type = type_y,
          curr_tree = curr_trees_y[[j]],
          node_min_size = node_min_size,
          s = s_y,
          max_bad_trees = max_bad_trees,
          splitting_rules = splitting_rules
        )

        # (c) Obtain the Metropolis-Hastings probability
        curr_tree_y <- curr_trees_y[[j]]
        new_tree_y <- new_trees_y[[j]]



        # if(ncol(binmat_all_y)!= ncol(BtB_y)){
        #
        #   print("iter = ")
        #   print(iter)
        #
        #   print("j = ")
        #   print(j)
        #
        #   print("dim(binmat_all_y) = ")
        #   print(dim(binmat_all_y))
        #
        #   print("dim(BtB_y) = ")
        #   print(dim(BtB_y))
        #
        #   stop("line 2500. ncol(binmat_all_y)!= ncol(BtB_y)")
        # }

        if (type_y == "grow") {
          # var_count_y[curr_trees_y[[j]]$var] <- var_count_y[curr_trees_y[[j]]$var] + 1

          # split node is just parent of last rows
          # new_tree_y$tree_matrix[nrow(new_tree_y$tree_matrix)-1,'parent']

          terminal_nodes_old = which(as.numeric(curr_tree_y$tree_matrix[,'terminal']) == 1)
          terminal_nodes_new = which(as.numeric(new_tree_y$tree_matrix[,'terminal']) == 1)
          removednode <- which(!( terminal_nodes_old %in% terminal_nodes_new ) )
          addednodes <- which(!( terminal_nodes_new %in% terminal_nodes_old ) )

          removednode_rowind <- terminal_nodes_old[removednode]
          addednodes_rowind <- terminal_nodes_new[addednodes]




          firstcolindtrees_y_new <- firstcolindtrees_y

          if(length(addednodes)==0){
            binmat_all_y_new <- binmat_all_y
            BtB_y_new <- BtB_y
            Btz_new <- Btz
            # BztBz_y_new <- BztBz_y
            # do not edit firstcolindtrees_y_new
          }else{

            # obtain new splitting variable and splitting point to calculate gating function
            #
            split_node_ind <- new_tree_y$tree_matrix[addednodes_rowind[1],'parent']
            split_var <- new_tree_y$tree_matrix[split_node_ind, 'split_variable']
            split_value <- new_tree_y$tree_matrix[split_node_ind, 'split_value']
            # calculate the gating function for all observations
            # gat_func_psi <- 1/(1 + exp(- (x.train[uncens_inds,split_var] - split_value)/tau_vec_outcome[j] ) )
            # gat_func_psi <- gating_func_logistic((x.train[uncens_inds,split_var] - split_value)/tau_vec_outcome[j] )
            gat_func_psi <- plogis((x.train[uncens_inds,split_var] - split_value)/tau_vec_outcome[j] )

            # create binary variables for new nodes
            # can either do this within a new grow_tree function or here
            # can just use node indices

            # # check right node removed
            # if( terminal_nodes_old[removednode] != (  new_tree_y$tree_matrix[nrow(new_tree_y$tree_matrix)-1,'parent']    )  ){
            #   stop(" terminal_nodes_old[removednode] != (  new_tree_y$tree_matrix[nrow(new_tree_y$tree_matrix)-1,'parent']    ) ")
            # }
            # # check right node added
            # if( any( terminal_nodes_new[addednodes] != (nrow(new_tree_y$tree_matrix)-1):(nrow(new_tree_y$tree_matrix)) ) ){
            #   stop("terminal_nodes_new[addednodes] != (nrow(new_tree_y$tree_matrix)-1):(nrow(new_tree_y$tree_matrix)) ")
            # }


            newnodesbin <- matrix(0, nrow(binmat_all_y),2)

            # print("length(gat_func_psi) = ")
            # print(length(gat_func_psi))
            # print("length((1-gat_func_psi)) = ")
            # print(length((1-gat_func_psi)))
            # print("dim(binmat_all_y[, (firstcolindtrees_y[j]-1+ removednode) , drop = FALSE]) = ")
            # print(dim(binmat_all_y[, (firstcolindtrees_y[j]-1+ removednode) , drop = FALSE]))
            # print("dim(newnodesbin) = ")
            # print(dim(newnodesbin))
            # print("dim(x.train) = ")
            # print(dim(x.train))

            newnodesbin[,1] <- binmat_all_y[, (firstcolindtrees_y[j]-1+ removednode) , drop = FALSE]*gat_func_psi
            newnodesbin[,2] <- binmat_all_y[, (firstcolindtrees_y[j]-1+ removednode) , drop = FALSE]*(1-gat_func_psi)

            # print("Line 3392 ")

            # newnodesbin[new_tree_y$node_indices == addednodes_rowind[1],1] <- rep(1, sum(new_tree_y$node_indices == addednodes_rowind[1]))
            # newnodesbin[new_tree_y$node_indices == addednodes_rowind[2],2] <- rep(1, sum(new_tree_y$node_indices == addednodes_rowind[2]))

            binmat_all_y_new <- binmat_all_y[, -(firstcolindtrees_y[j]-1+ removednode) , drop = FALSE ]
            BtB_y_new <- BtB_y[-(firstcolindtrees_y[j]-1+ removednode), -(firstcolindtrees_y[j]-1+ removednode) , drop = FALSE]
            BtB_y_new <- BtB_y[-(firstcolindtrees_y[j]-1+ removednode), -(firstcolindtrees_y[j]-1+ removednode), drop = FALSE ]
            Btz_new <- Btz[-(firstcolindtrees_y[j]-1+ removednode)  ]

            if(firstcolindtrees_y[j]-1 + addednodes[1]-1 == 0 ){
              binmat_all_y_new <- cbind(#binmat_all_y_new[, 1:(firstcolindtrees_y[j]-1 + addednodes-1 ) ],
                newnodesbin,
                binmat_all_y_new[, (firstcolindtrees_y[j]-1 + addednodes[1] ):ncol(binmat_all_y_new), drop = FALSE ])

              BtB_y_new <- cbind(rep(0, nrow(BtB_y_new)), rep(0, nrow(BtB_y_new)),
                                 BtB_y_new[, (firstcolindtrees_y[j]-1 + addednodes[1] ):ncol(BtB_y_new), drop = FALSE ])
              BtB_y_new <- rbind(rep(0, ncol(BtB_y_new)), rep(0, ncol(BtB_y_new)),
                                 BtB_y_new[(firstcolindtrees_y[j]-1 + addednodes[1] ):nrow(BtB_y_new), , drop = FALSE])

              # BtB_y_new[1,1] <- sum(newnodesbin[,1]^2) # sum(new_tree_y$node_indices == addednodes_rowind[1])
              # BtB_y_new[2,2] <- sum(newnodesbin[,2]^2) # sum(new_tree_y$node_indices == addednodes_rowind[2])

              BtB_y_new[1:2,1:2] <- t(newnodesbin) %*% newnodesbin

              BtB_y_new[ (1:2) ,-c( (1:2) )] <- crossprod((binmat_all_y_new[ ,(1:2), drop = FALSE ]) ,
                                                          binmat_all_y_new[ , - (1:2), drop = FALSE ])
              BtB_y_new[-c( (1:2)),(1:2)] <- t(BtB_y_new[(1:2),-c( (1:2)), drop = FALSE])

              Btz_new <- c(NA,NA,Btz_new)
              Btz_new[(1:2)] <- crossprod((binmat_all_y_new[ ,(1:2), drop = FALSE ]) , z_epsilon[uncens_inds])


              # if(any((binmat_all_y_new) ==0 )){
              #   print("which(binmat_all_y_new ==0, arr.ind = TRUE) = ")
              #   print(which(binmat_all_y_new ==0, arr.ind = TRUE))
              #
              #   print("binmat_all_y_new = ")
              #   print(binmat_all_y_new)
              #   stop("Line 3998 any((binmat_all_y_new) ==0 )))")
              # }
              #
              # if(any((BtB_y_new) ==0 )){
              #   print("which(BtB_y_new ==0, arr.ind = TRUE) = ")
              #   print(which(BtB_y_new ==0, arr.ind = TRUE))
              #
              #   print("BtB_y_new = ")
              #   print(BtB_y_new)
              #   stop("Line 4007 any((BtB_y_new) ==0 )))")
              # }
              #
              # if(any(is.na(binmat_all_y_new))){
              #   print("addednodes = ")
              #   print(addednodes)
              #   print("which(is.na(binmat_all_y_new), arr.ind = TRUE) = ")
              #   print(which(is.na(binmat_all_y_new), arr.ind = TRUE))
              #
              #   stop("Line 3825 NA in binmat_all_y_new")
              # }

            }else{
              if(firstcolindtrees_y[j]-1 + addednodes[2]-1 == ncol(binmat_all_y_new) +1 ){
                binmat_all_y_new <- cbind(binmat_all_y_new[, 1:(firstcolindtrees_y[j]-1 + addednodes[1]-1 ), drop = FALSE ],
                                          newnodesbin#,
                                          #binmat_all_y_new[, (firstcolindtrees_y[j]-1 + addednodes ):ncol(binmat_all_y_new) ]
                )

                BtB_y_new <- cbind( BtB_y_new[, 1:(firstcolindtrees_y[j]-1 + addednodes[1]-1 ), drop = FALSE ],
                                    rep(0, nrow(BtB_y_new)), rep(0, nrow(BtB_y_new)))
                BtB_y_new <- rbind(BtB_y_new[1:(firstcolindtrees_y[j]-1 + addednodes[1]-1 ), , drop = FALSE],
                                   rep(0, ncol(BtB_y_new)), rep(0, ncol(BtB_y_new)))

                # BtB_y_new[ncol(BtB_y_new)-1,ncol(BtB_y_new)-1] <- sum(newnodesbin[,1]^2) # sum(new_tree_y$node_indices == addednodes_rowind[1])
                # BtB_y_new[ncol(BtB_y_new),ncol(BtB_y_new)] <- sum(newnodesbin[,2]^2) # sum(new_tree_y$node_indices == addednodes_rowind[2])

                BtB_y_new[(ncol(BtB_y_new)-1):ncol(BtB_y_new),(ncol(BtB_y_new)-1):ncol(BtB_y_new) ] <- t(newnodesbin) %*% newnodesbin


                BtB_y_new[c(ncol(BtB_y_new)-1,ncol(BtB_y_new) ),
                          -c(c(ncol(BtB_y_new)-1,ncol(BtB_y_new) ))] <-
                  crossprod((binmat_all_y_new[ ,c(ncol(BtB_y_new)-1,ncol(BtB_y_new)), drop = FALSE ]) ,
                            binmat_all_y_new[ , - c(ncol(BtB_y_new)-1,ncol(BtB_y_new)), drop = FALSE ])

                BtB_y_new[-c(c(ncol(BtB_y_new)-1,ncol(BtB_y_new) )),
                          c(ncol(BtB_y_new)-1,ncol(BtB_y_new) )] <- t(BtB_y_new[c(ncol(BtB_y_new)-1,ncol(BtB_y_new) ),
                                                                                -c(c(ncol(BtB_y_new)-1,ncol(BtB_y_new) )), drop = FALSE])

                Btz_new <- c(Btz_new,NA,NA)

                Btz_new[c(length(Btz_new)-1,length(Btz_new))] <- crossprod((binmat_all_y_new[ ,c(ncol(BtB_y_new)-1,ncol(BtB_y_new)), drop = FALSE ]) ,
                                                                           z_epsilon[uncens_inds])

                # if(any((binmat_all_y_new) ==0 )){
                #   print("which(binmat_all_y_new ==0, arr.ind = TRUE) = ")
                #   print(which(binmat_all_y_new ==0, arr.ind = TRUE))
                #
                #   print("binmat_all_y_new = ")
                #   print(binmat_all_y_new)
                #   stop("Line 4057 any((binmat_all_y_new) ==0 )))")
                # }
                #
                # if(any((BtB_y_new) ==0 )){
                #   print("which(BtB_y_new ==0, arr.ind = TRUE) = ")
                #   print(which(BtB_y_new ==0, arr.ind = TRUE))
                #
                #   print("BtB_y_new = ")
                #   print(BtB_y_new)
                #   stop("Line 4066 any((BtB_y_new) ==0 )))")
                # }

                if(any(is.na(binmat_all_y_new))){
                  print("addednodes = ")
                  print(addednodes)
                  print("which(is.na(binmat_all_y_new), arr.ind = TRUE) = ")
                  print(which(is.na(binmat_all_y_new), arr.ind = TRUE))

                  stop("Line 3857 NA in binmat_all_y_new")
                }

              }else{
                binmat_all_y_new <- cbind(binmat_all_y_new[, 1:(firstcolindtrees_y[j]-1 + addednodes[1]-1 ), drop = FALSE ],
                                          newnodesbin,
                                          binmat_all_y_new[, (firstcolindtrees_y[j]-1 + addednodes[1] ):ncol(binmat_all_y_new), drop = FALSE ])

                BtB_y_new <- cbind( BtB_y_new[, 1:(firstcolindtrees_y[j]-1 + addednodes[1]-1 ), drop = FALSE ],
                                    rep(0, nrow(BtB_y_new)), rep(0, nrow(BtB_y_new)),
                                    BtB_y_new[, (firstcolindtrees_y[j]-1 + addednodes[1] ):ncol(BtB_y_new), drop = FALSE  ])
                BtB_y_new <- rbind(BtB_y_new[ 1:(firstcolindtrees_y[j]-1 + addednodes[1]-1 ), , drop = FALSE],
                                   rep(0, ncol(BtB_y_new)), rep(0, ncol(BtB_y_new)),
                                   BtB_y_new[(firstcolindtrees_y[j]-1 + addednodes[1] ):nrow(BtB_y_new), , drop = FALSE])

                # BtB_y_new[firstcolindtrees_y[j]-1 + addednodes[1],
                #           firstcolindtrees_y[j]-1 + addednodes[1]] <- sum(newnodesbin[,1]^2) # sum(new_tree_y$node_indices == addednodes_rowind[1])
                # BtB_y_new[firstcolindtrees_y[j]-1 + addednodes[2],
                #           firstcolindtrees_y[j]-1 + addednodes[2]] <- sum(newnodesbin[,2]^2) # sum(new_tree_y$node_indices == addednodes_rowind[2])

                BtB_y_new[firstcolindtrees_y[j]-1 + addednodes[1:2],
                          firstcolindtrees_y[j]-1 + addednodes[1:2]] <- t(newnodesbin) %*% newnodesbin



                BtB_y_new[firstcolindtrees_y[j]-1 + addednodes[1:2],
                          -c( firstcolindtrees_y[j]-1 + addednodes[1:2])] <-
                  crossprod((binmat_all_y_new[ ,firstcolindtrees_y[j]-1 + addednodes[1:2] , drop = FALSE]) ,
                            binmat_all_y_new[ , - (firstcolindtrees_y[j]-1 + addednodes[1:2]), drop = FALSE ])

                BtB_y_new[-c(firstcolindtrees_y[j]-1 + addednodes[1:2]),
                          firstcolindtrees_y[j]-1 + addednodes[1:2]] <- t(BtB_y_new[firstcolindtrees_y[j]-1 + addednodes[1:2],
                                                                                    -c( firstcolindtrees_y[j]-1 + addednodes[1:2]), drop = FALSE])

                Btz_new <- c(Btz_new[1:(firstcolindtrees_y[j]-1 + addednodes[1]-1)], NA, NA,
                             Btz_new[(firstcolindtrees_y[j]-1 + addednodes[1]):(length(Btz_new))])

                Btz_new[firstcolindtrees_y[j]-1 + addednodes[1:2]] <- crossprod((binmat_all_y_new[ ,firstcolindtrees_y[j]-1 + addednodes[1:2], drop = FALSE ]) ,
                                                                                z_epsilon[uncens_inds])

                # if(any((binmat_all_y_new) ==0 )){
                #   print("which(binmat_all_y_new ==0, arr.ind = TRUE) = ")
                #   print(which(binmat_all_y_new ==0, arr.ind = TRUE))
                #
                #   print("binmat_all_y_new = ")
                #   print(binmat_all_y_new)
                #   stop("Line 4121 any((binmat_all_y_new) ==0 )))")
                # }
                #
                # if(any((BtB_y_new) ==0 )){
                #   print("which(BtB_y_new ==0, arr.ind = TRUE) = ")
                #   print(which(BtB_y_new ==0, arr.ind = TRUE))
                #
                #   print("BtB_y_new = ")
                #   print(BtB_y_new)
                #   stop("Line 4130 any((BtB_y_new) ==0 )))")
                # }

                if(any(is.na(binmat_all_y_new))){
                  print("addednodes = ")
                  print(addednodes)
                  print("which(is.na(binmat_all_y_new), arr.ind = TRUE) = ")
                  print(which(is.na(binmat_all_y_new), arr.ind = TRUE))

                  stop("Line 3895 NA in binmat_all_y_new")
                }


              }
            }
            if(j < n.trees_censoring){
              firstcolindtrees_y_new[(j+1):n.trees_censoring] <- firstcolindtrees_y_new[(j+1):n.trees_censoring] + 1
            }
          }


        } else if (type_y == "prune") {
          # var_count_y[curr_trees_y[[j]]$var] <- var_count_y[curr_trees_y[[j]]$var] - 1
          terminal_nodes_old = which(as.numeric(curr_tree_y$tree_matrix[,'terminal']) == 1)
          terminal_nodes_new = which(as.numeric(new_tree_y$tree_matrix[,'terminal']) == 1)
          firstcolindtrees_y_new <- firstcolindtrees_y

          # if(   new_tree_y$pruned_parent ==-1   ){
          if(  length(terminal_nodes_old) == length(terminal_nodes_new)    ){
            binmat_all_y_new <- binmat_all_y
            Btz_new <- Btz
            BtB_y_new <- BtB_y
          }else{
            # delete column corresponding to pruned nodes and create column corresponding to parent node (does ordering matter?)
            # yes, ordering matters because otherwise would not be able to find columns to delete or grow in future steps
            # new_tree_y$pruned_parent

            # addednode <- new_tree_y$pruned_parent

            # # perhaps more efficient to just consider the difference
            # removednodes <- terminal_nodes_old[which(!( terminal_nodes_old %in% terminal_nodes_new ) )]
            # addednode <- terminal_nodes_new[which(!( terminal_nodes_new %in% terminal_nodes_old ) )]

            addednode <- which(terminal_nodes_new == new_tree_y$pruned_parent) # new terminal node is parent of removed nodes

            # if(length(addednode) != 1){
            #   print("terminal_nodes_new = ")
            #   print(terminal_nodes_new)
            #   print("new_tree_y = ")
            #   print(new_tree_y)
            #   print("addednode = ")
            #   print(addednode)
            #
            #   print("addednode = ")
            #   print(addednode)
            #   stop("line 2844 length(addednode) != 1")
            # }

            # removed nodes were children of parent node
            removednodes <- which(terminal_nodes_old %in%  which(curr_tree_y$tree_matrix[ , 'parent'] == new_tree_y$pruned_parent))

            # if(length(removednodes) != 2){
            #   print("terminal_nodes_old = ")
            #   print(terminal_nodes_old)
            #   print("new_tree_y = ")
            #   print(new_tree_y)
            #   stop("line 2857 length(removednodes) != 2")
            # }


            # removednodes_rowind <- terminal_nodes_old[removednodes]
            addednode_rowind <- terminal_nodes_new[addednode]

            newparentbin <- binmat_all_y[,firstcolindtrees_y[j]-1+ removednodes[1], drop = FALSE] +
              binmat_all_y[,firstcolindtrees_y[j]-1+ removednodes[2], drop = FALSE]


            binmat_all_y_new <- binmat_all_y[, -(firstcolindtrees_y[j]-1+ removednodes), drop = FALSE ]

            # print("line 2860. dim(binmat_all_y_new) = ")
            # print(dim(binmat_all_y_new))

            BtB_y_new <- BtB_y[-(firstcolindtrees_y[j]-1+ removednodes), -(firstcolindtrees_y[j]-1+ removednodes), drop = FALSE ]

            Btz_new <- Btz[-(firstcolindtrees_y[j]-1+ removednodes)]

            if(firstcolindtrees_y[j]-1 + addednode-1 == 0 ){
              binmat_all_y_new <- cbind(#binmat_all_y_new[, 1:(firstcolindtrees_y[j]-1 + addednode-1 ) ],
                newparentbin,
                binmat_all_y_new[, (firstcolindtrees_y[j]-1 + addednode ):ncol(binmat_all_y_new), drop = FALSE ])

              BtB_y_new <- cbind(rep(0, nrow(BtB_y_new)),  BtB_y_new[, (firstcolindtrees_y[j]-1 + addednode ):ncol(BtB_y_new), drop = FALSE ])
              BtB_y_new <- rbind(rep(0, ncol(BtB_y_new)), BtB_y_new[(firstcolindtrees_y[j]-1 + addednode ):nrow(BtB_y_new), , drop = FALSE ])

              # if(ncol(BtB_y_new)!=ncol(binmat_all_y_new)){
              #   print("firstcolindtrees_y[j]-1 + addednode = ")
              #   print(firstcolindtrees_y[j]-1 + addednode)
              #
              #   print("firstcolindtrees_y = ")
              #   print(firstcolindtrees_y)
              #
              #   print("removednodes = ")
              #   print(removednodes)
              #
              #
              #   print("dim(BtB_y) = ")
              #   print(dim(BtB_y))
              #
              #
              #   print("dim(BtB_y) = ")
              #   print(dim(BtB_y))
              #
              #   print("dim(binmat_all_y) = ")
              #   print(dim(binmat_all_y))
              #
              #   print("addednode = ")
              #   print(addednode)
              #
              #   print("dim(newparentbin) = ")
              #   print(dim(newparentbin))
              #
              #
              #   print("dim(BtB_y_new) = ")
              #   print(dim(BtB_y_new))
              #
              #   print("dim(binmat_all_y_new) = ")
              #   print(dim(binmat_all_y_new))
              #   stop("ncol(BtB_y_new)!=ncol(binmat_all_y_new)")
              # }

              BtB_y_new[1,1] <- sum(newparentbin^2) # sum(new_tree_y$node_indices == addednode_rowind)
              BtB_y_new[ 1 ,-c( 1 ) ] <- crossprod((binmat_all_y_new[ , 1, drop = FALSE ]) , binmat_all_y_new[ , - c(1), drop = FALSE ])
              BtB_y_new[-c( 1),1 ] <- t(BtB_y_new[1,-c( 1 ), drop = FALSE])

              Btz_new <- c(NA,Btz_new)

              Btz_new[1] <- crossprod((binmat_all_y_new[ ,1, drop = FALSE ]) , z_epsilon[uncens_inds])


              # if(any((binmat_all_y_new) ==0 )){
              #   print("which(binmat_all_y_new ==0, arr.ind = TRUE) = ")
              #   print(which(binmat_all_y_new ==0, arr.ind = TRUE))
              #
              #   print("binmat_all_y_new = ")
              #   print(binmat_all_y_new)
              #   stop("Line 4275 any((binmat_all_y_new) ==0 )))")
              # }
              #
              # if(any((BtB_y_new) ==0 )){
              #   print("which(BtB_y_new ==0, arr.ind = TRUE) = ")
              #   print(which(BtB_y_new ==0, arr.ind = TRUE))
              #
              #   print("BtB_y_new = ")
              #   print(BtB_y_new)
              #   stop("Line 4284 any((BtB_y_new) ==0 )))")
              # }

              if(any(is.na(binmat_all_y_new))){
                print("addednode = ")
                print(addednode)
                print("which(is.na(binmat_all_y_new), arr.ind = TRUE) = ")
                print(which(is.na(binmat_all_y_new), arr.ind = TRUE))

                stop("Line 4021 NA in binmat_all_y_new")
              }

            }else{
              if(firstcolindtrees_y[j]-1 + addednode-1 == ncol(binmat_all_y_new)  ){
                binmat_all_y_new <- cbind(binmat_all_y_new[, 1:(firstcolindtrees_y[j]-1 + addednode-1 ), drop = FALSE ],
                                          newparentbin#,
                                          #binmat_all_y_new[, (firstcolindtrees_y[j]-1 + addednode ):ncol(binmat_all_y_new) ]
                )

                BtB_y_new <- cbind( BtB_y_new[, 1:(firstcolindtrees_y[j]-1 + addednode-1 ), drop = FALSE ],
                                    rep(0, nrow(BtB_y_new)))
                BtB_y_new <- rbind(BtB_y_new[1:(firstcolindtrees_y[j]-1 + addednode-1 ), , drop = FALSE ],
                                   rep(0, ncol(BtB_y_new)))

                BtB_y_new[ncol(BtB_y_new),ncol(BtB_y_new)] <- sum(newparentbin^2) # sum(new_tree_y$node_indices == addednode_rowind)
                BtB_y_new[c(ncol(BtB_y_new) ),-c(ncol(BtB_y_new) )] <-
                  crossprod((binmat_all_y_new[ ,c(ncol(BtB_y_new) ) , drop = FALSE]) , binmat_all_y_new[ , - c(ncol(BtB_y_new) ), drop = FALSE ])
                BtB_y_new[-c(ncol(BtB_y_new) ),c(ncol(BtB_y_new) )] <- t(BtB_y_new[c(ncol(BtB_y_new) ), - c(ncol(BtB_y_new) ), drop = FALSE])

                Btz_new <- c(Btz_new,NA)

                Btz_new[c(ncol(BtB_y_new) )] <- crossprod((binmat_all_y_new[ ,c(ncol(BtB_y_new) ), drop = FALSE ]) , z_epsilon[uncens_inds])


                # if(any((binmat_all_y_new) ==0 )){
                #   print("which(binmat_all_y_new ==0, arr.ind = TRUE) = ")
                #   print(which(binmat_all_y_new ==0, arr.ind = TRUE))
                #
                #   print("binmat_all_y_new = ")
                #   print(binmat_all_y_new)
                #   stop("Line 4324 any((binmat_all_y_new) ==0 )))")
                # }
                #
                # if(any((BtB_y_new) ==0 )){
                #   print("which(BtB_y_new ==0, arr.ind = TRUE) = ")
                #   print(which(BtB_y_new ==0, arr.ind = TRUE))
                #
                #   print("BtB_y_new = ")
                #   print(BtB_y_new)
                #   stop("Line 4333 any((BtB_y_new) ==0 )))")
                # }

                if(any(is.na(binmat_all_y_new))){
                  print("addednode = ")
                  print(addednode)
                  print("which(is.na(binmat_all_y_new), arr.ind = TRUE) = ")
                  print(which(is.na(binmat_all_y_new), arr.ind = TRUE))

                  stop("Line 4041 NA in binmat_all_y_new")
                }

              }else{
                binmat_all_y_new <- cbind(binmat_all_y_new[, 1:(firstcolindtrees_y[j]-1 + addednode-1 ), drop = FALSE ],
                                          newparentbin,
                                          binmat_all_y_new[, (firstcolindtrees_y[j]-1 + addednode ):ncol(binmat_all_y_new), drop = FALSE ])


                BtB_y_new <- cbind( BtB_y_new[, 1:(firstcolindtrees_y[j]-1 + addednode-1 ), drop = FALSE ],
                                    rep(0, nrow(BtB_y_new)),
                                    BtB_y_new[, (firstcolindtrees_y[j]-1 + addednode ):ncol(BtB_y_new), drop = FALSE  ])
                BtB_y_new <- rbind(BtB_y_new[ 1:(firstcolindtrees_y[j]-1 + addednode-1 ), , drop = FALSE ],
                                   rep(0, ncol(BtB_y_new)),
                                   BtB_y_new[(firstcolindtrees_y[j]-1 + addednode ):nrow(BtB_y_new), , drop = FALSE ])

                BtB_y_new[firstcolindtrees_y[j]-1 + addednode,
                          firstcolindtrees_y[j]-1 + addednode] <- sum(newparentbin^2) # sum(new_tree_y$node_indices == addednode_rowind)
                BtB_y_new[firstcolindtrees_y[j]-1 + addednode,
                          - c(firstcolindtrees_y[j]-1 + addednode)] <-
                  crossprod((binmat_all_y_new[ ,firstcolindtrees_y[j]-1 + addednode, drop = FALSE ]) ,
                            binmat_all_y_new[ , - (firstcolindtrees_y[j]-1 + addednode), drop = FALSE ])
                BtB_y_new[-c( firstcolindtrees_y[j]-1 + addednode),
                          firstcolindtrees_y[j]-1 + addednode] <- t(BtB_y_new[firstcolindtrees_y[j]-1 + addednode,
                                                                              - c( firstcolindtrees_y[j]-1 + addednode),
                                                                              drop = FALSE])

                Btz_new <- c(Btz_new[1:(firstcolindtrees_y[j]-1 + addednode-1)], NA,
                             Btz_new[(firstcolindtrees_y[j]-1 + addednode):(length(Btz_new))])
                Btz_new[firstcolindtrees_y[j]-1 + addednode] <- crossprod((binmat_all_y_new[ , firstcolindtrees_y[j]-1 + addednode, drop = FALSE ]) ,
                                                                          z_epsilon[uncens_inds])

                # if(any((binmat_all_y_new) ==0 )){
                #   print("which(binmat_all_y_new ==0, arr.ind = TRUE) = ")
                #   print(which(binmat_all_y_new ==0, arr.ind = TRUE))
                #
                #   print("binmat_all_y_new = ")
                #   print(binmat_all_y_new)
                #   stop("Line 4380 any((binmat_all_y_new) ==0 )))")
                # }
                #
                # if(any((BtB_y_new) ==0 )){
                #   print("which(BtB_y_new ==0, arr.ind = TRUE) = ")
                #   print(which(BtB_y_new ==0, arr.ind = TRUE))
                #
                #   print("BtB_y_new = ")
                #   print(BtB_y_new)
                #   stop("Line 4389 any((BtB_y_new) ==0 )))")
                # }

                if(any(is.na(binmat_all_y_new))){
                  print("addednode = ")
                  print(addednode)
                  print("which(is.na(binmat_all_y_new), arr.ind = TRUE) = ")
                  print(which(is.na(binmat_all_y_new), arr.ind = TRUE))

                  stop("Line 4070 NA in binmat_all_y_new")
                }

              }
            }
            # child_left = as.numeric(old_tree_y$tree_matrix[new_tree_y$pruned_parent, 'child_left'])
            # child_right = as.numeric(old_tree_y$tree_matrix[new_tree_y$pruned_parent, 'child_right'])
            if(j < n.trees_censoring){
              firstcolindtrees_y_new[(j+1):n.trees_censoring]<- firstcolindtrees_y_new[(j+1):n.trees_censoring] - 1
            }
          }
        } else { # change step
          firstcolindtrees_y_new <- firstcolindtrees_y

          if(new_tree_y$var[1] != 0){ # What if change step returned $var equal to c(0,0) ????
            # var_count_y[curr_trees_y[[j]]$var[1]] <- var_count_y[curr_trees_y[[j]]$var[1]] - 1
            # var_count_y[curr_trees_y[[j]]$var[2]] <- var_count_y[curr_trees_y[[j]]$var[2]] + 1

            binmat_all_y_new <- binmat_all_y
            BtB_y_new <- BtB_y
            Btz_new <- Btz

            # find column of changed node
            # there is probably a more efficient way of doing this
            # changednodes <- sort(unique(new_tree_y$node_indices[which(new_tree_y$node_indices !=  old_tree_y$node_indices)]))
            changednodes_rowind <- sort(get_children(new_tree_y$tree_matrix, new_tree_y$node_to_change))

            terminal_nodes_new = which(as.numeric(new_tree_y$tree_matrix[,'terminal']) == 1)
            changednodes <- sort(which(terminal_nodes_new %in% changednodes_rowind))

            # obtain old splitting variable and splitting point to calculate old gating function
            split_node_ind <- curr_tree_y$tree_matrix[changednodes_rowind[1],'parent']
            split_var_old <- curr_tree_y$tree_matrix[split_node_ind, 'split_variable']
            split_value_old <- curr_tree_y$tree_matrix[split_node_ind, 'split_value']
            # calculate the gating function for all observations
            # gat_func_psi_old <- 1/(1 + exp(- (x.train[uncens_inds,split_var] - split_value)/tau_vec_outcome[j] ) )
            # gat_func_psi_old <- gating_func_logistic((x.train[uncens_inds,split_var] - split_value)/tau_vec_outcome[j] )
            gat_func_psi_old <- plogis((x.train[uncens_inds,split_var_old] - split_value_old)/tau_vec_outcome[j] )

            # obtain new splitting variable and splitting point to calculate new gating function
            #
            split_node_ind <- new_tree_y$tree_matrix[changednodes_rowind[1],'parent']
            split_var <- new_tree_y$tree_matrix[split_node_ind, 'split_variable']
            split_value <- new_tree_y$tree_matrix[split_node_ind, 'split_value']
            # calculate the gating function for all observations
            # gat_func_psi_new <- 1/(1 + exp(- (x.train[uncens_inds,split_var] - split_value)/tau_vec_outcome[j] ) )
            # gat_func_psi_new <- gating_func_logistic((x.train[uncens_inds,split_var] - split_value)/tau_vec_outcome[j] )
            gat_func_psi_new <- plogis((x.train[uncens_inds,split_var] - split_value)/tau_vec_outcome[j] )


            # just replace the two columns for the relevant terminal nodes
            # newnodesbin <- matrix(0, nrow(binmat_all_y),2)
            # newnodesbin[new_tree_y$node_indices == changednodes_rowind[1],1] <- rep(1, sum(new_tree_y$node_indices == changednodes_rowind[1]))
            # newnodesbin[new_tree_y$node_indices == changednodes_rowind[2],2] <- rep(1, sum(new_tree_y$node_indices == changednodes_rowind[2]))

            newnodesbin <- matrix(0, nrow(binmat_all_y),length(changednodes))

            # newnodesbin[,1] <- binmat_all_y_new[,firstcolindtrees_y[j]-1 + changednodes[1]]*
            #   gating_func_logistic_plogis_ratio((x.train[uncens_inds,split_var] - split_value)/tau_vec_outcome[j]   ,
            #                                     (x.train[uncens_inds,split_var_old] - split_value_old)/tau_vec_outcome[j])
            # # gat_func_psi_new/gat_func_psi_old
            # newnodesbin[,2] <- binmat_all_y_new[,firstcolindtrees_y[j]-1 + changednodes[2]]*
            #   gating_func_logistic_plogis_ratio(-(x.train[uncens_inds,split_var] - split_value)/tau_vec_outcome[j]   ,
            #                                     -(x.train[uncens_inds,split_var_old] - split_value_old)/tau_vec_outcome[j])
            # (1 - gat_func_psi_new)/(1-gat_func_psi_old)

            # if(any(is.na(newnodesbin))){
              newnodesbin <- matrix(0, nrow(binmat_all_y_new),length(changednodes))
              newnodesbin[,1] <- exp(log(binmat_all_y_new[,firstcolindtrees_y[j]-1 + changednodes[1]]) +
                                       gating_func_logistic_plogis_ratio_logdiff((x.train[uncens_inds,split_var] - split_value)/tau_vec_outcome[j],
                                                                                 (x.train[uncens_inds,split_var_old] - split_value_old)/tau_vec_outcome[j]))
              # gat_func_psi_new/gat_func_psi_old
              newnodesbin[,2] <- exp(log(binmat_all_y_new[,firstcolindtrees_y[j]-1 + changednodes[2]])+
                                       gating_func_logistic_plogis_ratio_logdiff(-1*(x.train[uncens_inds,split_var] - split_value)/tau_vec_outcome[j],
                                                                                 -1*(x.train[uncens_inds,split_var_old] - split_value_old)/tau_vec_outcome[j]))
            # }

            if(any(is.na(newnodesbin))){
             stop("Line 4476 NA in newnodesbin")
            }

            if(any(is.infinite(newnodesbin))){
              # anc_new = get_branch(new_tree_y[[j]])
              #
              # if(is.null(anc_new)){
              #   phi_matrix_new <- matrix(1, nrow = length(uncens_inds), ncol = 1)
              # }else{
              #   phi_matrix_new = phi_app(as.matrix(x.train[uncens_inds,]), as.matrix(anc_new), tau_vec_outcome[j])
              # }
              #
              # first_col <- firstcolindtrees_y[j]
              # last_col <- 0
              # if(j==n.trees_outcome){
              #   last_col <- ncol(binmat_all_y)
              # }else{
              #   last_col <- firstcolindtrees_y[j+1]-1
              # }
              # #new bin mat
              # binmat_all_y_new <- binmat_all_y
              # binmat_all_y_new[,first_col:last_col] <- phi_matrix_new
              #
              # newnodesbin <- phi_matrix_new[,changednodes]

              stop("Line 4499 infinite in newnodesbin")
            }


            if(length(changednodes)!=2){
              print("changednodes = ")
              print(changednodes)
              stop("length(changednodes)!=2")
            }

            BtB_y_new[firstcolindtrees_y[j]-1 + changednodes,
                      firstcolindtrees_y[j]-1 + changednodes] <- t(newnodesbin)%*%newnodesbin # sum(new_tree_y$node_indices[uncens_inds] == changednodes_rowind[change_ind])

            # for(change_ind in 1:length(changednodes)){
            #   # newnodesbin[new_tree_y$node_indices == changednodes_rowind[change_ind],change_ind] <- rep(1, sum(new_tree_y$node_indices == changednodes_rowind[change_ind]))
            #
            #   BtB_y_new[firstcolindtrees_y[j]-1 + changednodes[change_ind],
            #               firstcolindtrees_y[j]-1 + changednodes[change_ind]] <- sum(newnodesbin[,change_ind]^2) # sum(new_tree_y$node_indices[uncens_inds] == changednodes_rowind[change_ind])
            # }

            # for(change_ind in 1:length(changednodes)){
            #   newnodesbin[new_tree_y$node_indices == changednodes_rowind[change_ind],change_ind] <- rep(1, sum(new_tree_y$node_indices == changednodes_rowind[change_ind]))
            #
            #   BtB_y_new[firstcolindtrees_y[j]-1 + changednodes[change_ind],
            #             firstcolindtrees_y[j]-1 + changednodes[change_ind]] <- sum(new_tree_y$node_indices == changednodes_rowind[change_ind])
            #
            # }

            binmat_all_y_new[,firstcolindtrees_y[j]-1 + changednodes] <- newnodesbin

            # BtB_y_new[firstcolindtrees_y[j]-1 + changednodes[1],
            #           firstcolindtrees_y[j]-1 + changednodes[1]] <- sum(new_tree_y$node_indices == changednodes_rowind[1])
            # BtB_y_new[firstcolindtrees_y[j]-1 + changednodes[2],
            #           firstcolindtrees_y[j]-1 + changednodes[2]] <- sum(new_tree_y$node_indices == changednodes_rowind[2])

            BtB_y_new[firstcolindtrees_y[j]-1 + changednodes,
                      - c(firstcolindtrees_y[j]-1 + changednodes)] <-
              crossprod((binmat_all_y_new[ ,firstcolindtrees_y[j]-1 + changednodes, drop = FALSE ]) ,
                        binmat_all_y_new[ , - (firstcolindtrees_y[j]-1 + changednodes), drop = FALSE ])

            BtB_y_new[-c( firstcolindtrees_y[j]-1 + changednodes),
                      firstcolindtrees_y[j]-1 + changednodes] <- t(BtB_y_new[firstcolindtrees_y[j]-1 + changednodes,
                                                                             - c( firstcolindtrees_y[j]-1 + changednodes), drop = FALSE])
            Btz_new[firstcolindtrees_y[j]-1 + changednodes] <- crossprod((binmat_all_y_new[ , firstcolindtrees_y[j]-1 + changednodes, drop = FALSE ]) ,
                                                                         z_epsilon[uncens_inds])


            # if(any((binmat_all_y_new) ==0 )){
            #   print("which(binmat_all_y_new ==0, arr.ind = TRUE) = ")
            #   print(which(binmat_all_y_new ==0, arr.ind = TRUE))
            #
            #   print("binmat_all_y_new = ")
            #   print(binmat_all_y_new)
            #   stop("Line 4528 any((binmat_all_y_new) ==0 )))")
            # }
            #
            # if(any((BtB_y_new) ==0 )){
            #   print("which(BtB_y_new ==0, arr.ind = TRUE) = ")
            #   print(which(BtB_y_new ==0, arr.ind = TRUE))
            #
            #   print("BtB_y_new = ")
            #   print(BtB_y_new)
            #   stop("Line 4537 any((BtB_y_new) ==0 )))")
            # }

            if(any(is.infinite(binmat_all_y_new))){
              print("firstcolindtrees_y[j]-1 = ")
              print(firstcolindtrees_y[j]-1)
              print("changednodes = ")
              print(changednodes)
              print("which(is.infinite(binmat_all_y_new), arr.ind = TRUE) = ")
              print(which(is.infinite(binmat_all_y_new), arr.ind = TRUE))

              stop("Line 4548 infinite in binmat_all_y_new")
            }

            if(any(is.infinite(binmat_all_y_new))){
              print("firstcolindtrees_y[j]-1 = ")
              print(firstcolindtrees_y[j]-1)
              print("changednodes = ")
              print(changednodes)
              print("which(is.infinite(binmat_all_y_new), arr.ind = TRUE) = ")
              print(which(is.infinite(binmat_all_y_new), arr.ind = TRUE))

              stop("Line 4559 NA in binmat_all_y_new")
            }

            if(any(is.na(BtB_y_new))){
              print("firstcolindtrees_y[j]-1 = ")
              print(firstcolindtrees_y[j]-1)
              print("changednodes = ")
              print(changednodes)
              print("which(is.na(BtB_y_new), arr.ind = TRUE) = ")
              print(which(is.na(BtB_y_new), arr.ind = TRUE))

              stop("Line 4161 NA in binmat_all_y_new")
            }

            # if(any(BtB_y_new != crossprod(binmat_all_y_new) )){
            #   print("BtB_y_new =")
            #   print(BtB_y_new)
            #   print("binmat_all_y_new =")
            #   print(binmat_all_y_new)
            #   stop("line 3052. any(BtB_y_new != crossprod(binmat_all_y_new) )")
            # }

          }else{
            binmat_all_y_new <- binmat_all_y
            BtB_y_new <- BtB_y
            Btz_new <- Btz
          }
        }

        # print("Line 4145. iter = ")
        # print(iter)

        binmat_all_y_z <- cbind(binmat_all_y,z_epsilon[uncens_inds])
        binmat_all_y_new_z <- cbind(binmat_all_y_new,z_epsilon[uncens_inds])

        # if(max(abs(crossprod(binmat_all_y_new,z_epsilon[uncens_inds]) - Btz_new)) > 0.001){
        #   print("max(abs(crossprod(binmat_all_y_new,z_epsilon[uncens_inds]) - Btz_new)) = ")
        #   print(max(abs(crossprod(binmat_all_y_new,z_epsilon[uncens_inds]) - Btz_new)))
        #
        #   print("((crossprod(binmat_all_y_new,z_epsilon[uncens_inds]) - Btz_new)) = ")
        #   print(((crossprod(binmat_all_y_new,z_epsilon[uncens_inds]) - Btz_new)))
        #
        #   print("type_y = ")
        #   print(type_y)
        #
        #   stop("max(abs(crossprod(binmat_all_y_new,z_epsilon[uncens_inds]) - Btz_new)) > 0.001")
        # }

        # technically this is inefficient.
        # only need to update the elements of Btz corresponding to edited columns of B
        # Btz <- crossprod(binmat_all_y, z_epsilon[uncens_inds])
        # Btz_new <- crossprod(binmat_all_y_new, z_epsilon[uncens_inds])

        # print("dim(BtB_y) = ")
        # print(dim(BtB_y))
        # print("dim(Btz) = ")
        # print(dim(Btz))
        #
        # print("type_y = ")
        # print(type_y)

        # Btz <- crossprod(binmat_all_y,z_epsilon[uncens_inds])
        # Btz_new <- crossprod(binmat_all_y_new,z_epsilon[uncens_inds])

        # this must be updated because ztz has been updated
        BztBz_y <- cbind(BtB_y, Btz)
        BztBz_y <- rbind(BztBz_y, t(c(Btz, ztz)))
        # both Btz_new and ztz have been updated
        BztBz_y_new <- cbind(BtB_y_new, Btz_new)
        BztBz_y_new <- rbind(BztBz_y_new, t(c(Btz_new, ztz)))


        # for(j2 in 1:n.trees_outcome){
        #
        #   tempnodes <- new_trees_y[[j2]]$node_indices
        #   sorteduniqnodes <- sort(unique(tempnodes))
        #
        #   for(node_ind in 1:length(unique(tempnodes))){
        #     nodeval <- sorteduniqnodes[node_ind]
        #     if(any(binmat_all_y_new_z[,firstcolindtrees_y_new[j2]-1 + node_ind] !=   1*(new_trees_y[[j2]]$node_indices == nodeval) )){
        #       print("j2 = ")
        #       print(j2)
        #       print("nodeval = ")
        #       print(nodeval)
        #       print("firstcolindtrees_y_new[j2] = ")
        #       print(firstcolindtrees_y_new[j2])
        #       print("tempnodes = ")
        #       print(tempnodes)
        #       print("node_ind = ")
        #       print(node_ind)
        #       print("sorteduniqnodes = ")
        #       print(sorteduniqnodes)
        #
        #       print("binmat_all_y_new_z[,firstcolindtrees_y_new[j2]-1 + node_ind] = ")
        #       print(binmat_all_y_new_z[,firstcolindtrees_y_new[j2]-1 + node_ind])
        #       print("1*(new_trees_y[[j2]]$node_indices == nodeval) = ")
        #       print(1*(new_trees_y[[j2]]$node_indices == nodeval))
        #
        #       stop("line 2254. any(binmat_all_y_new_z[,firstcolindtrees_y_new[j2]-1 + node_ind] !=   1*(new_trees_y[[j2]]$node_indices == nodeval) )")
        #     }
        #   }
        # }
        #
        #   stop("line 3150. any(BztBz_y_new != crossprod(binmat_all_y_new_z) )")
        # }


        if(cov_prior == "VH"){
          priorgammavar <- tau*phi1
        }else{
          if(cov_prior == "Omori"){
            # stop("currently code only allows VH cov_prior")
            priorgammavar <- G0
          }else{
            if(jointgammanodes){
              stop("currently code only allows VH cov_prior with jointgammanodes")
            }
          }
        }

        # # BtB_y <- crossprod(binmat_all_y)
        # binmat_all_y_z <- cbind(binmat_all_y,z_epsilon[uncens_inds])
        # # binmat_all_y_new_z <- cbind(binmat_all_y_new,z_epsilon[uncens_inds])
        # # Btz <- crossprod(binmat_all_y,z_epsilon[uncens_inds])
        # # this must be updated because ztz has been updated
        # BztBz_y <- cbind(BtB_y, Btz)
        # BztBz_y <- rbind(BztBz_y, t(c(Btz, ztz)))
        # # both Btz_new and ztz have been updated
        # binmat_all_y_new_z <- cbind(binmat_all_y_new,z_epsilon[uncens_inds])
        # # BtB_y_new <- crossprod(binmat_all_y_new)
        # # Btz_new <- crossprod(binmat_all_y_new,z_epsilon[uncens_inds])
        # BztBz_y_new <- cbind(BtB_y_new, Btz_new)
        # BztBz_y_new <- rbind(BztBz_y_new, t(c(Btz_new, ztz)))
        # # BztBz_y_new <- crossprod(binmat_all_y_new_z)
        y_resids <- ystar[uncens_inds] - gamma1*(z[uncens_inds] - offsetz - mutemp_z[uncens_inds])

        # if(j==1){

        if(linearterms){
          if(jointgammanodes){
            if(one_chol ==TRUE){

              # CURRENT TREE: compute the log of the marginalised likelihood + log of the tree prior
              reslisttemp = tree_full_conditional_y_marg_savechol_lin(curr_trees_y,
                                                                      y_uncens, # current_partial_residuals,
                                                                      phi1,priorgammavar,
                                                                      sigma2_mu_y, binmat_all_y_z, BztBz_y,
                                                                      Bmean_p, invBvar_p, xmat_train, gamma0)
              l_old_y <- reslisttemp[[1]] + get_tree_prior(curr_trees_y[[j]], alpha_y, beta_y)
              IR_old_y <- reslisttemp[[2]]
              S_j_old_y <- reslisttemp[[3]]
            }else{
              # CURRENT TREE: compute the log of the marginalised likelihood + log of the tree prior
              l_old_y = tree_full_conditional_y_marg_lin(curr_trees_y,
                                                         y_uncens, # current_partial_residuals,
                                                         phi1,priorgammavar,
                                                         sigma2_mu_y, binmat_all_y_z, BztBz_y,
                                                         Bmean_p, invBvar_p, xmat_train, gamma0) + get_tree_prior(curr_trees_y[[j]], alpha_y, beta_y)
            }
          }else{
            if(one_chol ==TRUE){

              # CURRENT TREE: compute the log of the marginalised likelihood + log of the tree prior
              reslisttemp = tree_full_conditional_y_marg_nogamma_savechol_lin(curr_trees_y,
                                                                              y_resids, # current_partial_residuals,
                                                                              phi1,
                                                                              sigma2_mu_y, binmat_all_y, BtB_y,
                                                                              Bmean_p, invBvar_p, xmat_train)
              l_old_y <- reslisttemp[[1]] + get_tree_prior(curr_trees_y[[j]], alpha_y, beta_y)
              IR_old_y <- reslisttemp[[2]]
              S_j_old_y <- reslisttemp[[3]]
            }else{
              # CURRENT TREE: compute the log of the marginalised likelihood + log of the tree prior
              l_old_y = tree_full_conditional_y_marg_nogamma_lin(curr_trees_y,
                                                                 y_resids, # current_partial_residuals,
                                                                 phi1,
                                                                 sigma2_mu_y, binmat_all_y, BtB_y,
                                                                 Bmean_p, invBvar_p, xmat_train) + get_tree_prior(curr_trees_y[[j]], alpha_y, beta_y)
            }
          }
        }else{
          if(jointgammanodes){
            if(one_chol ==TRUE){
              # CURRENT TREE: compute the log of the marginalised likelihood + log of the tree prior
              reslisttemp = tree_full_conditional_y_marg_savechol(curr_trees_y,
                                                                  y_uncens, # current_partial_residuals,
                                                                  phi1,priorgammavar,
                                                                  sigma2_mu_y, binmat_all_y_z, BztBz_y, gamma0)
              l_old_y <- reslisttemp[[1]] + get_tree_prior(curr_trees_y[[j]], alpha_y, beta_y)
              IR_old_y <- reslisttemp[[2]]
              S_j_old_y <- reslisttemp[[3]]
            }else{

              # CURRENT TREE: compute the log of the marginalised likelihood + log of the tree prior
              l_old_y = tree_full_conditional_y_marg(curr_trees_y,
                                                     y_uncens, # current_partial_residuals,
                                                     phi1,priorgammavar,
                                                     sigma2_mu_y, binmat_all_y_z, BztBz_y, gamma0) +
                get_tree_prior(curr_trees_y[[j]], alpha_y, beta_y)
            }
          }else{
            if(one_chol ==TRUE){
              # CURRENT TREE: compute the log of the marginalised likelihood + log of the tree prior
              reslisttemp = tree_full_conditional_y_marg_nogamma_savechol(curr_trees_y,
                                                                          y_resids, # current_partial_residuals,
                                                                          phi1,
                                                                          sigma2_mu_y, binmat_all_y, BtB_y)
              l_old_y <- reslisttemp[[1]] + get_tree_prior(curr_trees_y[[j]], alpha_y, beta_y)
              IR_old_y <- reslisttemp[[2]]
              S_j_old_y <- reslisttemp[[3]]
            }else{
              # CURRENT TREE: compute the log of the marginalised likelihood + log of the tree prior
              l_old_y = tree_full_conditional_y_marg_nogamma(curr_trees_y,
                                                             y_resids, # current_partial_residuals,
                                                             phi1,
                                                             sigma2_mu_y, binmat_all_y, BtB_y) +
                get_tree_prior(curr_trees_y[[j]], alpha_y, beta_y)
            }
          }
        }

        # } # end j==1

        # print("line 3511")

        if((nrow(new_tree_y$tree_matrix) == nrow(curr_tree_y$tree_matrix) ) & (type_y != "change" )){
          alpha_MH <- 0
          # print("no good trees")

        }else{
          if(linearterms){
            if(jointgammanodes){
              if(one_chol == TRUE){

                # NEW TREE: compute the log of the marginalised likelihood + log of the tree prior
                reslisttemp = tree_full_conditional_y_marg_savechol_lin(new_trees_y,
                                                                        y_uncens, # current_partial_residuals,
                                                                        phi1,priorgammavar,
                                                                        sigma2_mu_y, binmat_all_y_new_z, BztBz_y_new,
                                                                        Bmean_p, invBvar_p, xmat_train, gamma0)
                l_new_y <- reslisttemp[[1]] + get_tree_prior(new_trees_y[[j]], alpha_y, beta_y)
                IR_new_y <- reslisttemp[[2]]
                S_j_new_y <- reslisttemp[[3]]
              }else{
                # NEW TREE: compute the log of the marginalised likelihood + log of the tree prior
                l_new_y = tree_full_conditional_y_marg_lin(new_trees_y,
                                                           y_uncens, # current_partial_residuals,
                                                           phi1,priorgammavar,
                                                           sigma2_mu_y, binmat_all_y_new_z, BztBz_y_new,
                                                           Bmean_p, invBvar_p, xmat_train, gamma0) + get_tree_prior(new_trees_y[[j]], alpha_y, beta_y)
              }
            }else{
              if(one_chol ==TRUE){
                # NEW TREE: compute the log of the marginalised likelihood + log of the tree prior
                reslisttemp = tree_full_conditional_y_marg_nogamma_savechol_lin(new_trees_y,
                                                                                y_resids, # current_partial_residuals,
                                                                                phi1,
                                                                                sigma2_mu_y, binmat_all_y_new, BtB_y_new,
                                                                                Bmean_p, invBvar_p, xmat_train)
                l_new_y <- reslisttemp[[1]] + get_tree_prior(new_trees_y[[j]], alpha_y, beta_y)
                IR_new_y <- reslisttemp[[2]]
                S_j_new_y <- reslisttemp[[3]]
              }else{
                # NEW TREE: compute the log of the marginalised likelihood + log of the tree prior
                l_new_y = tree_full_conditional_y_marg_nogamma_lin(new_trees_y,
                                                                   y_resids, # current_partial_residuals,
                                                                   phi1,
                                                                   sigma2_mu_y, binmat_all_y_new, BtB_y_new,
                                                                   Bmean_p, invBvar_p, xmat_train) + get_tree_prior(new_trees_y[[j]], alpha_y, beta_y)
              }
            }

          }else{
            if(jointgammanodes){
              if(one_chol ==TRUE){
                # NEW TREE: compute the log of the marginalised likelihood + log of the tree prior
                reslisttemp = tree_full_conditional_y_marg_savechol(new_trees_y,
                                                                    y_uncens, # current_partial_residuals,
                                                                    phi1,priorgammavar,
                                                                    sigma2_mu_y, binmat_all_y_new_z, BztBz_y_new, gamma0)
                l_new_y <- reslisttemp[[1]] + get_tree_prior(new_trees_y[[j]], alpha_y, beta_y)
                IR_new_y <- reslisttemp[[2]]
                S_j_new_y <- reslisttemp[[3]]

              }else{
                # NEW TREE: compute the log of the marginalised likelihood + log of the tree prior
                l_new_y = tree_full_conditional_y_marg(new_trees_y,
                                                       y_uncens, # current_partial_residuals,
                                                       phi1,priorgammavar,
                                                       sigma2_mu_y, binmat_all_y_new_z, BztBz_y_new, gamma0) +
                  get_tree_prior(new_trees_y[[j]], alpha_y, beta_y)
              }
            }else{
              if(one_chol ==TRUE){
                # NEW TREE: compute the log of the marginalised likelihood + log of the tree prior
                reslisttemp = tree_full_conditional_y_marg_nogamma_savechol(new_trees_y,
                                                                            y_resids, # current_partial_residuals,
                                                                            phi1,
                                                                            sigma2_mu_y, binmat_all_y_new, BtB_y_new)
                l_new_y <- reslisttemp[[1]] + get_tree_prior(new_trees_y[[j]], alpha_y, beta_y)
                IR_new_y <- reslisttemp[[2]]
                S_j_new_y <- reslisttemp[[3]]
              }else{
                # NEW TREE: compute the log of the marginalised likelihood + log of the tree prior
                l_new_y = tree_full_conditional_y_marg_nogamma(new_trees_y,
                                                               y_resids, # current_partial_residuals,
                                                               phi1,
                                                               sigma2_mu_y, binmat_all_y_new, BtB_y_new) +
                  get_tree_prior(new_trees_y[[j]], alpha_y, beta_y)
              }
            }
          }

          alpha_MH = alpha_mh(l_new_y,l_old_y, curr_trees_y[[j]],new_trees_y[[j]], type_y)

        }

        if(is.na(alpha_MH)){
          print("l_old_y = ")
          print(l_old_y)
          print("l_new_y = ")
          print(l_new_y)

          print("curr_tree_y = ")
          print(curr_tree_y)

          print("new_tree_y = ")
          print(new_tree_y)

          print("get_tree_prior(curr_trees_y[[j]], alpha_y, beta_y) = ")
          print(get_tree_prior(curr_trees_y[[j]], alpha_y, beta_y))

          print(" get_tree_prior(new_trees_y[[j]], alpha_y, beta_y) = ")
          print( get_tree_prior(new_trees_y[[j]], alpha_y, beta_y))

          print("any(is.na(binmat_all_y_new)) = ")
          print(any(is.na(binmat_all_y_new)))

          print("any(is.na(BtB_y_new)) = ")
          print(any(is.na(BtB_y_new)))

          print("any(is.na(binmat_all_y)) = ")
          print(any(is.na(binmat_all_y)))

          print("any(is.na(BtB_y)) = ")
          print(any(is.na(BtB_y)))

          print("any(is.infinite(binmat_all_y_new)) = ")
          print(any(is.infinite(binmat_all_y_new)))

          print("any(is.infinite(BtB_y_new)) = ")
          print(any(is.infinite(BtB_y_new)))

          print("any(is.infinite(binmat_all_y)) = ")
          print(any(is.infinite(binmat_all_y)))

          print("any(is.infinite(BtB_y)) = ")
          print(any(is.infinite(BtB_y)))

          print("any(is.infinite(y_resids)) = ")
          print(any(is.infinite(y_resids)))

          print("any(is.na(y_resids)) = ")
          print(any(is.na(y_resids)))

          print("Line 4445")
        }


        if(alpha_MH > runif(1)) {
          curr_trees_y[[j]] = new_trees_y[[j]]
          firstcolindtrees_y <- firstcolindtrees_y_new
          Btz <- Btz_new
          binmat_all_y_z <- binmat_all_y_new_z
          binmat_all_y <- binmat_all_y_new
          BtB_y <- BtB_y_new
          BztBz_y <- BztBz_y_new

          l_old_y <- l_new_y
          if( (one_chol == TRUE) ){
            IR_old_y <- IR_new_y
            S_j_old_y <- S_j_new_y
          }
          # if(sparse){

          if (type_y == "grow") {
            var_count_y[curr_trees_y[[j]]$var] <- var_count_y[curr_trees_y[[j]]$var] + 1
          } else if (type_y == "prune") {
            var_count_y[curr_trees_y[[j]]$var] <- var_count_y[curr_trees_y[[j]]$var] - 1
          } else {
            if(curr_trees_y[[j]]$var[1]!=0){ # What if change step returned $var equal to c(0,0) ????
              var_count_y[curr_trees_y[[j]]$var[1]] <- var_count_y[curr_trees_y[[j]]$var[1]] - 1
              var_count_y[curr_trees_y[[j]]$var[2]] <- var_count_y[curr_trees_y[[j]]$var[2]] + 1
            }
          }
          # }
        }

        # type_y_prev <- type_y

      } # end loop over y trees


      # print("Line 4519 iter = ")
      # print(iter)

      # in case it is somehow not updated earlier
      # print("line 3610")
      if(cov_prior == "VH"){
        priorgammavar <- tau*phi1
      }else{
        if(cov_prior == "Omori"){
          # stop("currently code only allows VH cov_prior")
          priorgammavar <- G0
        }else{
          if(jointgammanodes){
            stop("currently code only allows VH cov_prior with jointgammanodes")
          }
        }
      }



      # BtB_y <- crossprod(binmat_all_y)
      binmat_all_y_z <- cbind(binmat_all_y,z_epsilon[uncens_inds])
      # binmat_all_y_new_z <- cbind(binmat_all_y_new,z_epsilon[uncens_inds])
      # Btz <- crossprod(binmat_all_y,z_epsilon[uncens_inds])
      # this must be updated because ztz has been updated
      BztBz_y <- cbind(BtB_y, Btz)
      BztBz_y <- rbind(BztBz_y, t(c(Btz, ztz)))
      # both Btz_new and ztz have been updated
      # BtB_y_new <- crossprod(binmat_all_y_new)
      # Btz_new <- crossprod(binmat_all_y_new,z_epsilon[uncens_inds])
      # BztBz_y_new <- cbind(BtB_y_new, Btz_new)
      # BztBz_y_new <- rbind(BztBz_y_new, t(c(Btz_new, ztz)))
      # BztBz_y <- crossprod(binmat_all_y_z)



      # print("Line 4555 iter = ")
      # print(iter)

      if(mh_tau_bandwidth){
        # Compute the log of the marginalized likelihood and the log of the tau prior for the current tree

        l_old_y2 = l_old_y  +
          log_tau_prior(tau_vec_outcome[j], tau_rate) + log(tau_vec_outcome[j])

        # Calculate the new bandwidth using Random Walk
        # tau_new[[j]] = tau[[j]]*exp(runif(n = 1,min = -1,max = 1))
        tau_new = tau_vec_outcome[j]*(5^(runif(n = 1,min = -1,max = 1)))

        anc_new = get_branch(curr_trees_y[[j]])
        # if(ncol(as.matrix(anc_new))==1){
        #   stop("Line 4486. (ncol(as.matrix(anc_new))==1")
        # }


        # print("line 4603 iter = ")
        # print(iter)
        if(is.null(anc_new)){
          phi_matrix_new <- matrix(1, nrow = length(uncens_inds), ncol = 1)
        }else{
          phi_matrix_new = phi_app(as.matrix(x.train[uncens_inds,]), as.matrix(anc_new), tau_new)
        }


        # print("line 4607 iter = ")
        # print(iter)

        first_col <- firstcolindtrees_y[j]
        last_col <- 0
        if(j==n.trees_outcome){
          last_col <- ncol(binmat_all_y)
        }else{
          last_col <- firstcolindtrees_y[j+1]-1
        }
        #new bin mat
        binmat_all_y_new <- binmat_all_y
        binmat_all_y_new[,first_col:last_col] <- phi_matrix_new
        binmat_all_y_new_z <- cbind(binmat_all_y_new,z_epsilon[uncens_inds])
        # calculate new  B BtB etc
        Btz_new <- t(binmat_all_y_new) %*% z_epsilon[uncens_inds]
        BtB_y_new <- BtB_y
        BtB_y_new[first_col:last_col, ] <- t(phi_matrix_new) %*% binmat_all_y_new
        BtB_y_new[ , first_col:last_col ] <- t(BtB_y_new[first_col:last_col, ])
        BztBz_y_new <- cbind(BtB_y_new, Btz_new)
        BztBz_y_new <- rbind(BztBz_y_new, t(c(Btz_new, ztz)))


        # if(any((binmat_all_y_new) ==0 )){
        #   print("which(binmat_all_y_new ==0, arr.ind = TRUE) = ")
        #   print(which(binmat_all_y_new ==0, arr.ind = TRUE))
        #
        #   print("binmat_all_y_new = ")
        #   print(binmat_all_y_new)
        #   stop("Line 5033 any((binmat_all_y_new) ==0 )))")
        # }
        #
        # if(any((BtB_y_new) ==0 )){
        #   print("which(BtB_y_new ==0, arr.ind = TRUE) = ")
        #   print(which(BtB_y_new ==0, arr.ind = TRUE))
        #
        #   print("BtB_y_new = ")
        #   print(BtB_y_new)
        #   stop("Line 5042 any((BtB_y_new) ==0 )))")
        # }


        y_resids <- ystar[uncens_inds] - gamma1*(z[uncens_inds] - offsetz - mutemp_z[uncens_inds])

        if(linearterms){
          if(jointgammanodes){
              # NEW TREE: compute the log of the marginalised likelihood + log of the tree prior
              l_new_y = tree_full_conditional_y_marg_lin(curr_trees_y,
                                                         y_uncens, # current_partial_residuals,
                                                         phi1,priorgammavar,
                                                         sigma2_mu_y, binmat_all_y_new_z, BztBz_y_new,
                                                         Bmean_p, invBvar_p, xmat_train, gamma0) + get_tree_prior(curr_trees_y[[j]], alpha_y, beta_y)
          }else{
              # NEW TREE: compute the log of the marginalised likelihood + log of the tree prior
              l_new_y = tree_full_conditional_y_marg_nogamma_lin(curr_trees_y,
                                                                 y_resids, # current_partial_residuals,
                                                                 phi1,
                                                                 sigma2_mu_y, binmat_all_y_new, BtB_y_new,
                                                                 Bmean_p, invBvar_p, xmat_train)+ get_tree_prior(curr_trees_y[[j]], alpha_y, beta_y)
          }
        }else{
          if(jointgammanodes){
              # NEW TREE: compute the log of the marginalised likelihood + log of the tree prior
              l_new_y = tree_full_conditional_y_marg(curr_trees_y,
                                                     y_uncens, # current_partial_residuals,
                                                     phi1,priorgammavar,
                                                     sigma2_mu_y, binmat_all_y_new_z, BztBz_y_new, gamma0)+ get_tree_prior(curr_trees_y[[j]], alpha_y, beta_y)
          }else{
              # NEW TREE: compute the log of the marginalised likelihood + log of the tree prior
              l_new_y = tree_full_conditional_y_marg_nogamma(curr_trees_y,
                                                             y_resids, # current_partial_residuals,
                                                             phi1,
                                                             sigma2_mu_y, binmat_all_y_new, BtB_y_new)+ get_tree_prior(curr_trees_y[[j]], alpha_y, beta_y)
          }
        }

        l_new_y2 = l_new_y + log_tau_prior(tau_new, tau_rate) + log(tau_new)


        # Here, the calculation of alpha doesn't depend on any transition probabilities
        a = exp(l_new_y2 - l_old_y2)

        if(is.na(a)| is.null(a)){
          print("l_new_y2 = ")
          print(l_new_y2)
          print("l_new_y = ")
          print(l_new_y)
          print("log_tau_prior(tau_new, tau_rate) = ")
          print(log_tau_prior(tau_new, tau_rate))
          print("log(tau_new) = ")
          print(log(tau_new))
          print("l_old_y2 = ")
          print(l_old_y2)
          print("l_old_y = ")
          print(l_old_y)
          print("log_tau_prior(tau_vec_outcome[j], tau_rate) = ")
          print(log_tau_prior(tau_vec_outcome[j], tau_rate))
          print("log(tau_vec_outcome[j]) = ")
          print(log(tau_vec_outcome[j]))

        }

        if(a > runif(1)) { # In case the alpha is bigger than a uniformly sampled value between zero and one
          tau_vec_outcome[j] = tau_new # The current bandwidth "becomes" the new bandwidth, if the latter is better
          #new bin mat
          binmat_all_y <- binmat_all_y_new
          binmat_all_y_z <- binmat_all_y_new_z
          # calculate new  B BtB etc
          Btz <- Btz_new
          BtB_y <- BtB_y_new
          BztBz_y <- BztBz_y_new
          # UPDATE B, BTB
          # UPDATE PARTIAL RESIDUALS (IF ANY?)
          # UPDATE YHAT
        }

      }


      y_resids <- ystar[uncens_inds] - gamma1*(z[uncens_inds] - offsetz - mutemp_z[uncens_inds])


      # print("Line 4673 iter = ")
      # print(iter)

      if(linearterms){
        if(jointgammanodes){
          if(one_chol ==TRUE){

            mudrawlist_y = simulate_mu_all_y_fast_lin(curr_trees_y,
                                                      y_uncens, # current_partial_residuals,
                                                      phi1,
                                                      priorgammavar,
                                                      sigma2_mu_y, binmat_all_y_z, BztBz_y, firstcolindtrees_y, IR_old_y, S_j_old_y,
                                                      xmat_train, Bmean_p, invBvar_p, gamma0)
          }else{
            mudrawlist_y = simulate_mu_all_y_lin(curr_trees_y,
                                                 y_uncens, # current_partial_residuals,
                                                 phi1,
                                                 priorgammavar,
                                                 sigma2_mu_y, binmat_all_y_z, BztBz_y, firstcolindtrees_y,
                                                 xmat_train, Bmean_p, invBvar_p, gamma0)
          }

        }else{
          if(one_chol ==TRUE){
            mudrawlist_y = simulate_mu_all_y_nogamma_fast_lin(curr_trees_y,
                                                              y_resids, # current_partial_residuals,
                                                              phi1,
                                                              sigma2_mu_y, binmat_all_y, BtB_y, firstcolindtrees_y, IR_old_y, S_j_old_y,
                                                              xmat_train, Bmean_p, invBvar_p)
          }else{
            mudrawlist_y = simulate_mu_all_y_nogamma_lin(curr_trees_y,
                                                         y_resids, # current_partial_residuals,
                                                         phi1,
                                                         sigma2_mu_y, binmat_all_y, BtB_y, firstcolindtrees_y,
                                                         xmat_train, Bmean_p, invBvar_p)
          }
        }

      }else{
        if(jointgammanodes){
          if(one_chol ==TRUE){
            mudrawlist_y = simulate_mu_all_y_fast(curr_trees_y,
                                                  y_uncens, # current_partial_residuals,
                                                  phi1,
                                                  priorgammavar,
                                                  sigma2_mu_y, binmat_all_y_z, BztBz_y, firstcolindtrees_y, IR_old_y, S_j_old_y, gamma0)
          }else{
            mudrawlist_y = simulate_mu_all_y(curr_trees_y,
                                             y_uncens, # current_partial_residuals,
                                             phi1,
                                             priorgammavar,
                                             sigma2_mu_y, binmat_all_y_z, BztBz_y, firstcolindtrees_y, gamma0)
          }

        }else{
          if(one_chol ==TRUE){
            mudrawlist_y = simulate_mu_all_y_nogamma_fast(curr_trees_y,
                                                          y_resids, # current_partial_residuals,
                                                          phi1,
                                                          sigma2_mu_y, binmat_all_y, BtB_y, firstcolindtrees_y, IR_old_y, S_j_old_y)
          }else{
            mudrawlist_y = simulate_mu_all_y_nogamma(curr_trees_y,
                                                     y_resids, # current_partial_residuals,
                                                     phi1,
                                                     sigma2_mu_y, binmat_all_y, BtB_y, firstcolindtrees_y)
          }

        }
      }



      # print("Line 4744 iter = ")
      # print(iter)

      curr_trees_y <- mudrawlist_y[[1]]
      new_trees_y <- curr_trees_y
      mutemp_y <- binmat_all_y %*% mudrawlist_y[[3]]
      if(linearterms){
        if(jointgammanodes){
          mutemp_y <- mutemp_y + xmat_train %*% mudrawlist_y[[6]]
        }else{
          mutemp_y <- mutemp_y + xmat_train %*% mudrawlist_y[[4]]
          mutemp_y <- cbind(xmat_train, binmat_all_y) %*% mudrawlist_y[[5]]

        }
      }
      if(jointgammanodes){
        gamma1 <- mudrawlist_y[[5]]
      }

      # # Updating BART predictions
      # current_fit = get_predictions(curr_trees_y[j], x.train[uncens_inds,], single_tree = TRUE)
      # mutemp_y = mutemp_y - tree_fits_store_y[,j] # subtract the old fit
      # mutemp_y = mutemp_y + current_fit # add the new fit
      # tree_fits_store_y[,j] = current_fit # update the new fit


    }else{

      for (j in 1:n.trees_outcome) {
        current_partial_residuals = y_resids - mutemp_y + tree_fits_store_y[,j]

        # We need the new and old trees for the likelihoods
        new_trees_y <- curr_trees_y

        type_y = sample_move(curr_trees_y[[j]], i, 0, #n_burn
                             trans_prob)

        # Generate a new tree based on the current
        new_trees_y[[j]] <- update_tree(
          y = y_resids,
          X = x.train[uncens_inds,],
          type = type_y,
          curr_tree = curr_trees_y[[j]],
          node_min_size = node_min_size,
          s = s_y,
          max_bad_trees = max_bad_trees,
          splitting_rules = splitting_rules
        )

        # (c) Obtain the Metropolis-Hastings probability
        curr_tree_y <- curr_trees_y[[j]]
        new_tree_y <- new_trees_y[[j]]

        if((nrow(new_tree_y$tree_matrix) == nrow(curr_tree_y$tree_matrix) ) & (type_y != "change" )){
          alpha_MH <- 0
          # print("no good trees")
        }else{
          # CURRENT TREE: compute the log of the marginalised likelihood + log of the tree prior

          # if(j==1){
          l_old_y = tree_full_conditional(curr_trees_y[[j]],
                                          current_partial_residuals,
                                          phi1,
                                          sigma2_mu_y) +
            get_tree_prior(curr_trees_y[[j]], alpha_y, beta_y)
          # }

          # NEW TREE: compute the log of the marginalised likelihood + log of the tree prior
          l_new_y = tree_full_conditional(new_trees_y[[j]],
                                          current_partial_residuals,
                                          phi1,
                                          sigma2_mu_y) +
            get_tree_prior(new_trees_y[[j]], alpha_y, beta_y)

          alpha_MH = alpha_mh(l_new_y,l_old_y, curr_trees_y[[j]],new_trees_y[[j]], type_y)

        }

        if(is.na(alpha_MH)){
          print("l_old_y = ")
          print(l_old_y)
          print("l_new_y = ")
          print(l_new_y)

          print("curr_tree_y = ")
          print(curr_tree_y)

          print("new_tree_y = ")
          print(new_tree_y)

          print("get_tree_prior(curr_trees_y[[j]], alpha_y, beta_y) = ")
          print(get_tree_prior(curr_trees_y[[j]], alpha_y, beta_y))

          print(" get_tree_prior(new_trees_y[[j]], alpha_y, beta_y) = ")
          print( get_tree_prior(new_trees_y[[j]], alpha_y, beta_y))

          print("Line 4794")

        }


        if(alpha_MH > runif(1)) {
          curr_trees_y[[j]] = new_trees_y[[j]]
          l_old_y <- l_new_y
          # if(sparse){
          if (type_y == "grow") {
            var_count_y[curr_trees_y[[j]]$var] <- var_count_y[curr_trees_y[[j]]$var] + 1
          } else if (type_y == "prune") {
            var_count_y[curr_trees_y[[j]]$var] <- var_count_y[curr_trees_y[[j]]$var] - 1
          } else {
            if(curr_trees_y[[j]]$var[1]!=0){ # What if change step returned $var equal to c(0,0) ????
              var_count_y[curr_trees_y[[j]]$var[1]] <- var_count_y[curr_trees_y[[j]]$var[1]] - 1
              var_count_y[curr_trees_y[[j]]$var[2]] <- var_count_y[curr_trees_y[[j]]$var[2]] + 1
            }
          }
          # }

        }

        curr_trees_y[[j]] = simulate_mu(curr_trees_y[[j]],
                                        current_partial_residuals,
                                        phi1,
                                        sigma2_mu_y)

        # Updating BART predictions
        current_fit = get_predictions(curr_trees_y[j], x.train[uncens_inds,], single_tree = TRUE)
        mutemp_y = mutemp_y - tree_fits_store_y[,j] # subtract the old fit
        mutemp_y = mutemp_y + current_fit # add the new fit

        mutemp_y_trees = mutemp_y_trees - tree_fits_store_y[,j] # subtract the old fit
        mutemp_y_trees = mutemp_y_trees + current_fit # add the new fit

        tree_fits_store_y[,j] = current_fit # update the new fit

      } # end loop over y trees

    }



    # print("Line 4885 iter = ")
    # print(iter)

    # if(sparse){
    #   tempcounts <- fcount(sampler_y$getTrees()$var)
    #   tempcounts <- tempcounts[tempcounts$x != -1, ]
    #   var_count_y <- rep(0, p_y)
    #   var_count_y[tempcounts$x] <- tempcounts$N
    # }

    #update z_epsilon
    y_epsilon[uncens_inds] <- ystar[uncens_inds] - mutemp_y
    z_epsilon <- z - offsetz - mutemp_z # should be unnecessary

    mutemp_test_y <- get_predictions(curr_trees_y,
                                     x.test,
                                     single_tree = length(curr_trees_y) == 1)

    if(linearterms & !marginalize){
      mutemp_test_y_trees <- mutemp_test_y
      mutemp_test_y <- mutemp_test_y_trees + mutemp_test_y_lin
    }

    if(linearterms & marginalize){
      if(jointgammanodes){
        mutemp_test_y <- mutemp_test_y + xmat_test %*% mudrawlist_y[[6]]
      }else{
        mutemp_test_y <- mutemp_test_y + xmat_test %*% mudrawlist_y[[4]]
      }
    }

    ############# Covariance matrix samples ##########################
    # print("Draw Covariance matrix. iter = ")
    # print(iter)

    if(cov_prior == "Ding"){

      rho1 <- gamma1/sqrt(phi1 + (gamma1^2) )  #sqrt(Sigma_mat[2,2])

      sigz2 <- 1/rgamma(n = 1,
                        shape = nu0/2,
                        rate = cding/(2*(1- (rho1^2))) )

      z_epsilon2 <- sqrt(sigz2)*(z - offsetz - mutemp_z)

      zsquares <- crossprod(z_epsilon2[uncens_inds])[1] # crossprod(z_epsilon2[uncens_inds], z_epsilon2[uncens_inds])[1]
      ysquares <- crossprod(y_epsilon[uncens_inds])[1] # crossprod(y_epsilon[uncens_inds], y_epsilon[uncens_inds])[1]
      zycross <- crossprod(z_epsilon2[uncens_inds], y_epsilon[uncens_inds])[1]

      Stemp <- cbind(c(ysquares, zycross),
                     c(zycross, zsquares))


      # Cmat <- cbind(c(cding,0),c(0,1))

      tempsigma <- rinvwishart(nu = n1 + nu0,
                               S = Stemp+cding*diag(2))
      # tempsigma <- rinvwishart(nu = n1 + nu0,
      #                          S = Stemp+Cmat)

      transmat <- cbind(c(1,0),c(0,1/sqrt(tempsigma[2,2])))
      tempomega <- (transmat %*% tempsigma) %*% transmat

      temprho <- tempomega[1,2]/(sqrt(tempomega[1,1]))

      # if(tempomega[1,1] != tempsigma[1,1]){
      #     print("tempomega[1,1] = ")
      #     print(tempomega[1,1])
      #     print("tempsigma[1,1] = ")
      #     print(tempsigma[1,1])
      # }

      gamma1 <- tempomega[1,2]
      # if(temprho < -0.3){
      #   print("n1 + nu0 = ")
      #   print(n1 + nu0)
      #   print("cding = ")
      #   print(cding)
      #   print("temprho = ")
      #   print(temprho)
      #   print("sigz2 = ")
      #   print(sigz2)
      #   print("Stemp = ")
      #   print(Stemp)
      # }

      phi1 <- tempomega[1,1] - (gamma1^2)


    }else{

      ########### Simultaneous phi and gamma draw #####################

      if(simultaneous_covmat == TRUE){

        if(marginalize & jointgammanodes){
          stop("Code does not allow all 3 of simultaneous_covmat == TRUE, jointgammanodes and marginalize == TRUE")
        }
        if(cov_prior == "VH"){
          h_num <- (gamma0/tau) + crossprod(z_epsilon[uncens_inds], y_epsilon[uncens_inds])[1]
          a_temp <- (1/tau) + crossprod(z_epsilon[uncens_inds], z_epsilon[uncens_inds])[1]

          h_temp <- h_num/a_temp
          k_temp <- ((gamma0^2)/tau)+S0 +
            crossprod(y_epsilon[uncens_inds], y_epsilon[uncens_inds])[1] -
            ((h_num^2)/(a_temp))

          phi1 <- 1/rgamma(n = 1,
                           shape =  (nzero + n1 )/2,
                           rate = k_temp/2)

          gamma1 <- rnorm(n = 1, mean = h_temp, sd = sqrt(phi1/a_temp))


        }else{
          stop("If simultaneous_covmat == TRUE, then must use Van Hasselt Covariance prior. Set cov_prior to VH.")
        }
      }else{

        #########  set parameters for gamma draw  ######################################################

        # if(cov_prior == TRUE){
        #   G0 <- tau*phi1
        # }
        if(cov_prior == "VH"){
          G0draw <- tau*phi1
        }else{
          if(cov_prior == "Omori"){
            G0draw <- G0
          }else{
            mixind <- rbinom(n = 1,size = 1,prob = mixprob)
            if(mixind == 1){
              G0draw <- tau*phi1
            }else{
              G0draw <- G0
            }
          }
        }

        if( ( (!marginalize) & !(linearterms & jointbetagamma))| (!jointgammanodes) ){

          # G1inv <- (1/G0) + (1/phi1)*crossprod(z_epsilon)
          G1inv <- (1/G0draw) + (1/phi1)*crossprod(z_epsilon[uncens_inds])[1]
          # G1inv <- (1/tau) + (1/phi1)*crossprod(z_epsilon[uncens_inds])
          G1 <- (1/G1inv)#[1,1]

          # gamma_one <- (G1*( (1/G0draw)*gamma0 + (1/phi1)*crossprod(z_epsilon , y_epsilon   )   ))[1,1]
          gamma_one <- (G1*( (1/G0draw)*gamma0 +
                               (1/phi1)*crossprod(z_epsilon[uncens_inds] , y_epsilon[uncens_inds]   )[1]   ))

          if(cov_prior == "VH"){
            gamma_one <- ((gamma0/tau) + crossprod(z_epsilon[uncens_inds] , y_epsilon[uncens_inds]   )[1] )/
              ((1/tau) + crossprod(z_epsilon[uncens_inds])[1])
            G1 <- phi1/((1/tau) + crossprod(z_epsilon[uncens_inds])[1])
          }
          # gamma_one <- (G1*( (1/tau)*gamma0 + (1/phi1)*crossprod(z_epsilon[uncens_inds] , y_epsilon[uncens_inds]   )   ))[1,1]

          # print("gamma1 = ")
          # print(gamma1)
          gamma1 <- rnorm(n = 1, mean = gamma_one, sd =  sqrt(G1) )
        }

        # print("gamma1 = ")
        # print(gamma1)


        #########  set parameters for phi draw  ######################################################

        n_one <- nzero + n1 + 1

        # print("S0 = ")
        # print(S0)
        # print("(gamma1^2)*crossprod(z_epsilon) = ")
        # print((gamma1^2)*crossprod(z_epsilon))
        #
        # print("2*gamma1*crossprod(z_epsilon , y_epsilon   ) = ")
        # print(2*gamma1*crossprod(z_epsilon , y_epsilon   ))
        #
        # print("crossprod(y_epsilon) = ")
        # print(crossprod(y_epsilon))

        # S1 <- S0 + (gamma1^2)*crossprod(z_epsilon) - 2*gamma1*crossprod(z_epsilon , y_epsilon   )  + crossprod(y_epsilon)

        S1 <- 0 #S0 + (gamma1^2)/G0 + gamma1*crossprod( y_epsilon[uncens_inds] - gamma1*z_epsilon[uncens_inds]  )  + crossprod(y_epsilon)

        if(cov_prior == "VH"){
          # S1 <- S0 + (gamma1^2)/tau + gamma1*crossprod( y_epsilon[uncens_inds] - gamma1*z_epsilon[uncens_inds]  )  + crossprod(y_epsilon)
          S1 <- S0 + ((gamma1- gamma0)^2)/tau +
            crossprod( y_epsilon[uncens_inds] - gamma1*z_epsilon[uncens_inds]  )[1] # + crossprod(y_epsilon)
        }else{
          if(cov_prior == "Omori"){
            S1 <- S0 + #+ (gamma1^2)/G0 +
              crossprod( y_epsilon[uncens_inds] - gamma1*z_epsilon[uncens_inds]  )[1]  #+crossprod(y_epsilon)[1]
          }else{
            mixind <- rbinom(n = 1,size = 1,prob = mixprob)
            if(mixind == 1){
              # S1 <- S0 + (gamma1^2)/tau + gamma1*crossprod( y_epsilon[uncens_inds] - gamma1*z_epsilon[uncens_inds]  )  + crossprod(y_epsilon)
              S1 <- S0 + ((gamma1- gamma0)^2)/tau +
                crossprod( y_epsilon[uncens_inds] - gamma1*z_epsilon[uncens_inds]  )[1] # + crossprod(y_epsilon)
            }else{
              S1 <- S0 + #+ (gamma1^2)/G0 +
                crossprod( y_epsilon[uncens_inds] - gamma1*z_epsilon[uncens_inds]  )[1]  #+crossprod(y_epsilon)[1]
            }
          }
        }
        # print("S1 = ")
        # print(S1)
        # print("n_one = ")
        # print(n_one)


        # print("Line 883 phi1 = ")
        # print(phi1)

        # draw from inverse gamma
        phi1 <- 1/rgamma(n = 1, shape =  n_one/2, rate = S1/2)


        if(is.na(phi1)){
          print("n_one = ")
          print(n_one)
          print("S1 = ")
          print(S1)

          print("S1 = ")
          print(S1)

          print("gamma1 = ")
          print(gamma1)


          print(" y_epsilon[uncens_inds] = ")
          print( y_epsilon[uncens_inds])

          print("gamma1*z_epsilon[uncens_inds]  = ")
          print(gamma1*z_epsilon[uncens_inds] )

          print("((gamma1- gamma0)^2)/tau = ")
          print(((gamma1- gamma0)^2)/tau)
          print("crossprod( y_epsilon[uncens_inds] - gamma1*z_epsilon[uncens_inds]  )[1] = ")
          print(crossprod( y_epsilon[uncens_inds] - gamma1*z_epsilon[uncens_inds]  )[1])
          stop("is.na(phi1)")
        }

        # print("Line 890 phi1 = ")
        # print(phi1)
        #
        # print("Line 890 n_one = ")
        # print(n_one)
        #
        # print("Line 890 S1 = ")
        # print(S1)
        # print("n1 = ")
        # print(n1)

      } # end of else statement, for simultaneous_covmat = FALSE

    }

    ######### update Sigma matrix #####################################################

    # print("Update sigma. iter = ")
    # print(iter)


    Sigma_mat <- cbind(c(1,gamma1),c(gamma1,phi1+gamma1^2))
    Sigma_orig_scale <- cbind(c(1,tempsd*gamma1),c(tempsd*gamma1,  (tempsd^2)*(phi1+gamma1^2)) )

    ########## tau draws ############

    if(tau_hyperprior){
      tau <- 1/rgamma(n = 1,shape = alpha_tau + 1/2, rate = beta_tau + (1/(2*phi1))*(gamma1 - gamma0)^2)
    }

    ###### Accelerated sampler  ###############################


    # if(accelerate == TRUE){
    #
    #   meanmu_z <- (min(z - offsetz) +max(z- offsetz))/(2*n.trees_censoring)
    #
    #   # the variance should be zero?
    #   sigmu_z <- (max(z- offsetz) - min(z- offsetz))/(2*2*sqrt(n.trees_censoring))
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
    #   if(accept_bin == 1){
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


    ########### splitting probability draws #############################

    # print("Uppdate Split probabilities. iter = ")
    # print(iter)

    if (sparse & (iter > floor(n.burnin * 0.5))) {
      s_update_z <- update_s(var_count_z, p_z, alpha_s_z)
      s_z <- s_update_z[[1]]

      s_update_y <- update_s(var_count_y, p_y, alpha_s_y)
      s_y <- s_update_y[[1]]

      if(alpha_split_prior){
        alpha_s_z <- update_alpha(s_z, alpha_scale_z, alpha_a_z, alpha_b_z, p_z, s_update_z[[2]])
        alpha_s_y <- update_alpha(s_y, alpha_scale_y, alpha_a_y, alpha_b_y, p_y, s_update_y[[2]])
      }
    }


    ####### update sigma_mu #########################
    if(sigma_mu_prior){

      if(sigma_mu_dist == "Cauchy"){
        sigma2_mu_z <-  update_sigma_mu_par(curr_trees_z, sigma2_mu_z)
        if(is.na(sigma2_mu_z )){
          stop("Line 1728 sigma2_mu_z  NA")
        }
        sigma2_mu_y <-  update_sigma_mu_par(curr_trees_y, sigma2_mu_y)
        if(is.na(sigma2_mu_y )){
          stop("Line 1733 sigma2_mu_y  NA")
        }
      }else{
        if(sigma_mu_dist == "Normal"){
        sigma2_mu_z <-  update_sigma_mu_par_norm(curr_trees_z, sigma2_mu_z)
        if(is.na(sigma2_mu_z )){
          stop("Line 1728 sigma2_mu_z  NA")
        }
        sigma2_mu_y <-  update_sigma_mu_par_norm(curr_trees_y, sigma2_mu_y)
        if(is.na(sigma2_mu_y )){
          stop("Line 1733 sigma2_mu_y  NA")
        }
        }else{
          stop("sigma_mu_dist must be 'Cauchy or 'Normal'.")
        }
      }

    }


    ###### Store results   ###############################



    # print("Store results iter = ")
    # print(iter)


    if(iter > n.burnin){
      iter_min_burnin <- iter-n.burnin


      #NOTE y and z training sample values saved here
      #do not correspond to the the same means and errors as
      #the test values and expectations saved here.
      #However, they are the values to which the trees in this round were fitted.

      #draw z and y for test observations
      zytest <- matrix(NA, nrow = ntest, ncol = 2)


      if(fast == TRUE){
        zytest <- Rfast::rmvnorm(n = ntest,
                                 mu = c(0, 0),
                                 sigma = Sigma_mat)

      }else{
        zytest <- mvrnorm(n = ntest,
                          mu = c(0, 0),
                          Sigma = Sigma_mat)

      }



      # print("length(mutemp_test_z) = ")
      # print(length(mutemp_test_z))
      #
      # print("offsetz = ")
      # print(offsetz)
      #
      # print("length(zytest[,1]) = ")
      # print(length(zytest[,1]))

      zytest[,1] <- zytest[,1] + offsetz + mutemp_test_z
      zytest[,2] <- zytest[,2] + mutemp_test_y

      # for(i in 1:ntest){
      #   zytest[i,] <- mvrnorm(n = 1,
      #                         mu = c(offsetz + mutemp_test_z[i], mutemp_test_y[i]),
      #                         Sigma = Sigma_mat)
      # }

      if(fast == TRUE){
        probcens_train <- fastpnorm(- mutemp_z[uncens_inds] - offsetz )
        probcens_test <- fastpnorm(- mutemp_test_z - offsetz)
      }else{
        probcens_train <- pnorm(- mutemp_z[uncens_inds] - offsetz )
        probcens_test <- pnorm(- mutemp_test_z - offsetz)
      }

      #calculate conditional expectation

      # condexptrain <- mutemp_y + gamma1*(dnorm(- mutemp_z - offsetz ))/(1-probcens_train)
      # condexptrain <- mutemp_y + gamma1*(dnorm(- mutemp_z[uncens_inds] - offsetz ))/(1-probcens_train)
      # condexptest <- mutemp_test_y + gamma1*(dnorm(- mutemp_test_z - offsetz ))/(1-probcens_test)

      temp_ztrain <- mutemp_z[uncens_inds] + offsetz
      temp_ztest <- mutemp_test_z + offsetz
      #
      #       if(fast == TRUE){
      #         IMR_train <- exp( dnorm(temp_ztrain,log=T) - log(fastpnorm(temp_ztrain) ))
      #         IMR_test <- exp( dnorm(temp_ztest,log=T) - log(fastpnorm(temp_ztest) ))
      #
      #         # IMR_train <- fastnormdens(temp_ztrain)/fastpnorm(temp_ztrain)
      #         # IMR_test <- fastnormdens(temp_ztest)/fastpnorm(temp_ztest)
      #       }else{
      IMR_train <- exp( dnorm(temp_ztrain,log=T) - pnorm(temp_ztrain,log.p = T) )
      IMR_test <- exp( dnorm(temp_ztest,log=T) - pnorm(temp_ztest,log.p = T) )
      # }


      condexptrain <- mutemp_y + gamma1*IMR_train
      condexptest <- mutemp_test_y + gamma1*IMR_test

      # draw$Z.mat_train[,iter_min_burnin] <- z
      # draw$Z.mat_test[,iter_min_burnin] <-  zytest[,1]
      # draw$Y.mat_train = array(NA, dim = c(n, n.iter)),
      # draw$Y.mat_test = array(NA, dim = c(ntest, n.iter)),
      draw$mu_y_train[, iter_min_burnin] <- (mutemp_y + (y_max + y_min)/2 )*tempsd+tempmean
      draw$mu_y_test[, iter_min_burnin] <- (mutemp_test_y + (y_max + y_min)/2 )*tempsd+tempmean

      # draw$mucens_y_train[, iter_min_burnin] <- mutemp_y[cens_inds]
      # draw$muuncens_y_train[, iter_min_burnin] <- mutemp_y[uncens_inds]
      draw$muuncens_y_train[, iter_min_burnin] <- (mutemp_y + (y_max + y_min)/2 )*tempsd+tempmean

      draw$mu_z_train[, iter_min_burnin] <- mutemp_z
      draw$mu_z_test[, iter_min_burnin] <- mutemp_test_z

      draw$train.probcens[, iter_min_burnin] <-  probcens_train
      draw$test.probcens[, iter_min_burnin] <-  probcens_test

      draw$cond_exp_train[, iter_min_burnin] <- (condexptrain + (y_max + y_min)/2 )*tempsd+tempmean
      draw$cond_exp_test[, iter_min_burnin] <- (condexptest + (y_max + y_min)/2 )*tempsd+tempmean

      # draw$ystar_train[, iter_min_burnin] <- ystar*tempsd+tempmean
      draw$ystar_test[, iter_min_burnin] <- (zytest[,2] + (y_max + y_min)/2 )*tempsd+tempmean
      draw$zstar_train[,iter_min_burnin] <- z
      draw$zstar_test[,iter_min_burnin] <-  zytest[,1]

      # draw$ycond_draws_train[[iter_min_burnin]] <-  ystar[z >=0]*tempsd+tempmean
      draw$ycond_draws_test[[iter_min_burnin]] <-  (zytest[,2][zytest[,1] >= 0] + (y_max + y_min)/2 )*tempsd+tempmean

      draw$Sigma_draws[,, iter_min_burnin] <- Sigma_orig_scale#Sigma_mat


      if(is.numeric(censored_value)){

        # uncondexptrain <- censored_value*probcens_train +  mutemp_y*(1- probcens_train ) + gamma1*dnorm(- mutemp_z - offsetz )
        uncondexptrain <- censored_value*probcens_train +  mutemp_y*(1- probcens_train ) + gamma1*dnorm(- mutemp_z[uncens_inds] - offsetz )
        uncondexptest <- censored_value*probcens_test +  mutemp_test_y*(1- probcens_test ) + gamma1*dnorm(- mutemp_test_z - offsetz)

        draw$uncond_exp_train[, iter_min_burnin] <- (uncondexptrain + (y_max + y_min)/2 )*tempsd+tempmean
        draw$uncond_exp_test[, iter_min_burnin] <- (uncondexptest + (y_max + y_min)/2 )*tempsd+tempmean


        # draw$ydraws_train[, iter_min_burnin] <- ifelse(z < 0, censored_value, ystar )
        draw$ydraws_test[, iter_min_burnin] <- (ifelse(zytest[,1] < 0, censored_value, zytest[,2] ) + (y_max + y_min)/2 )*tempsd+tempmean
      }

      draw$var_count_y_store[iter_min_burnin,] <- var_count_y
      draw$var_count_z_store[iter_min_burnin,] <- var_count_z
      if(sparse){
        draw$alpha_s_y_store[iter_min_burnin] <- alpha_s_y
        draw$alpha_s_z_store[iter_min_burnin] <- alpha_s_z
        draw$s_prob_y_store[iter_min_burnin,] <- s_y
        draw$s_prob_z_store[iter_min_burnin,] <- s_z

      }



    } # end if iter > burnin

    # pb$tick()

    if(iter %% print.opt == 0){
      print(paste("Gibbs Iteration", iter))
      # print(c(sigma2.alpha, sigma2.beta))
    }


  }#end iterations of Giibs sampler

  draw$y_max <- y_max
  draw$y_min <- y_min


  return(draw)



}


#' @title Type II Tobit Bayesian Additive Regression Trees implemented using MCMC
#'
#' @description Type II Tobit Bayesian Additive Regression Trees implemented using MCMC
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

tbart2marg <- function(x.train,
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
                    proposal.probs = c(birth_death = 0.5, swap = 0.1, change = 0.4, birth = 0.5),
                    sigmadbarts = NA_real_,
                    print.opt = 100,
                    eq_by_eq = TRUE,
                    # accelerate = FALSE,
                    cov_prior = "Ding",
                    tau = 1/3,
                    mixprob = 0.5,
                    simultaneous_covmat = TRUE,
                    fast = TRUE,
                    nu0 = 3,
                    quantsig = 0.9,
                    sparse = FALSE,
                    alpha_a_y = 0.5,
                    alpha_b_y = 1,
                    alpha_a_z = 0.5,
                    alpha_b_z = 1,
                    alpha_split_prior = TRUE,
                    sigma_mu_prior = FALSE,
                    node_min_size = 5,
                    centre_y = TRUE,
                    splitting_rules = "discrete",
                    marginalize = FALSE){



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
  sigma2 <- 1                          # !!!!!!!!!!!!!!
  mu_mu <- 0 # (y_min + y_max) / (2 * m)

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

  if(sparse){
    s_y <- rep(1 / p_y, p_y) # probability vector to be used during the growing process for DART feature weighting
    rho_y <- p_y # For DART

    if(alpha_split_prior){
      alpha_s_y <- p_y
    }else{
      alpha_s_y <- 1
    }
    alpha_scale_y <- p_y


    s_z <- rep(1 / p_z, p_z) # probability vector to be used during the growing process for DART feature weighting
    rho_z <- p_z # For DART

    if(alpha_split_prior){
      alpha_s_z <- p_z
    }else{
      alpha_s_z <- 1
    }
    alpha_scale_z <- p_z

  }


  tree_fits_store_z = matrix(0, ncol = n.trees_censoring, nrow = n)
  tree_fits_store_y = matrix(0, ncol = n.trees_outcome, nrow = n1)




  offsetz <- 0 # qnorm(n1/n)

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
        dtemp <-  1*(!(is.na(y)))
      }else{
        dtemp <-  1*(y != censored_value)
      }

      df = data.frame(x = cbind(x.train,w.train), y = y, d = dtemp )

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
    } else {
      sigest = sd(y[uncens_inds])
      correst <- 0
    }
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


  # if(sigest > G0){
  #   S0 <- (nzero - 2)*(sigest-G0)
  # }else{
  # sigquant <- 0.9
  # qchi <- qchisq(1.0-sigquant,nzero/2)
  # S0 <- 2*(sigest*sigest*qchi)/(nzero/2)
  # }

  # S0 <- (sigest^2)*(nzero-2)/(1+tau)

  S0 <- (sigest^2)*(1 - correst^2)*(nzero-2)/(1+tau)

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

  if(sparse){
    var_count_y <- rep(0, p_y)
    var_count_z <- rep(0, p_z)

    draw$alpha_s_y_store <- rep(NA, n.iter)
    draw$alpha_s_z_store <- rep(NA, n.iter)
    draw$var_count_y_store <- matrix(0, ncol = p_y, nrow = n.iter)
    draw$var_count_z_store <- matrix(0, ncol = p_z, nrow = n.iter)
    draw$s_prob_y_store <- matrix(0, ncol = p_y, nrow = n.iter)
    draw$s_prob_z_store <- matrix(0, ncol = p_z, nrow = n.iter)

  }


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



  #initialize sum-of-tree sampler
  z_resids <- z - offsetz #z_epsilon
  z_resids[uncens_inds] <- z[uncens_inds] - offsetz - (ystar[uncens_inds]  - 0)*gamma1/(phi1 + gamma1^2)

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
  sigma2_mu_z <- ((max(z_resids)-min(z_resids))/(2 * k_z * sqrt(n.trees_censoring)))^2

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

  mu_z <- mutemp_z

  # mutemp_test_z <- sampler_z$predict(xdf_z_test)[,1]#samplestemp_z$test[,1]


  if(sparse){
    var_count_z <- rep(0, p_z)
  }
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
  sigma2_mu_y <- ((max(y_resids)-min(y_resids))/(2 * k_y * sqrt(n.trees_outcome)))^2

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

  mu_y <- mutemp_y

  if(sparse){
    var_count_y <- rep(0, p_y)
  }
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


  #########  Begin Gibbs sampler ######################################################

  # pb <- progress_bar$new(total = n.iter+n.burnin)
  # pb <- progress_bar$new(
  #   format = " [:bar] :percent eta: :eta",
  #   total = n.iter+n.burnin, clear = FALSE, width= 60)

  #loop through the Gibbs sampler iterations
  for(iter in 1:(n.iter+n.burnin)){

    if(mutemp_y != rowSums(tree_fits_store_y)){

      stop(" print mutemp_y != rowSums(tree_fits_store_y)")

    }


    if(mutemp_z != rowSums(tree_fits_store_z)){

      stop(" print mutemp_z != rowSums(tree_fits_store_z)")

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

    # #set the response for draws of z trees
    # sampler_z$setResponse(y = z_resids)
    # #set the standard deivation
    # sampler_z$setSigma(sigma = 1)

    weightstemp[uncens_inds] <- (gamma1^2 + phi1)/phi1


    if(marginalize){

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
          # CURRENT TREE: compute the log of the marginalised likelihood + log of the tree prior
          l_old = tree_full_conditional_weighted(curr_trees_z[[j]],
                                                 current_partial_residuals,# sigma2,
                                                 sigma2_mu_z,
                                                 weightstemp) +
            get_tree_prior(curr_trees_z[[j]], alpha_z, beta_z)

          # NEW TREE: compute the log of the marginalised likelihood + log of the tree prior
          l_new = tree_full_conditional_weighted(new_trees_z[[j]],
                                                 current_partial_residuals,# sigma2,
                                                 sigma2_mu_z,
                                                 weightstemp) +
            get_tree_prior(new_trees_z[[j]], alpha_z, beta_z)

          alpha_MH = alpha_mh(l_new,l_old, curr_trees_z[[j]],new_trees_z[[j]], type_z)

        }


        if(is.na(alpha_MH)){
          print("l_old = ")
          print(l_old)
          print("l_new = ")
          print(l_new)

          print("curr_tree_z = ")
          print(curr_tree_z)

          print("new_tree_z = ")
          print(new_tree_z)

          print("get_tree_prior(curr_trees_z[[j]], alpha_z, beta_z) = ")
          print(get_tree_prior(curr_trees_z[[j]], alpha_z, beta_z))

          print(" get_tree_prior(new_trees_z[[j]], alpha_z, beta_z) = ")
          print( get_tree_prior(new_trees_z[[j]], alpha_z, beta_z))

          stop(";ome 12-0. alpha_MH NA")

        }

        if(alpha_MH > runif(1)) {
          curr_trees_z[[j]] = new_trees_z[[j]]

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

        }



      } # end loop over z trees


      curr_trees_z[[j]] = simulate_mu_weighted_all(curr_trees_z,
                                               current_partial_residuals,
                                               # sigma2,
                                               sigma2_mu_z,
                                               weightstemp)

      # # Updating BART predictions
      # current_fit = get_predictions(curr_trees_z[j], w.train, single_tree = TRUE)
      # mutemp_z = mutemp_z - tree_fits_store_z[,j] # subtract the old fit
      # mutemp_z = mutemp_z + current_fit # add the new fit
      # tree_fits_store_z[,j] = current_fit # update the new fit
      #

    }else{

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
          # CURRENT TREE: compute the log of the marginalised likelihood + log of the tree prior
          l_old = tree_full_conditional_weighted(curr_trees_z[[j]],
                                        current_partial_residuals,# sigma2,
                                        sigma2_mu_z,
                                        weightstemp) +
            get_tree_prior(curr_trees_z[[j]], alpha_z, beta_z)

          # NEW TREE: compute the log of the marginalised likelihood + log of the tree prior
          l_new = tree_full_conditional_weighted(new_trees_z[[j]],
                                        current_partial_residuals,# sigma2,
                                        sigma2_mu_z,
                                        weightstemp) +
            get_tree_prior(new_trees_z[[j]], alpha_z, beta_z)

          alpha_MH = alpha_mh(l_new,l_old, curr_trees_z[[j]],new_trees_z[[j]], type_z)

        }


        if(is.na(alpha_MH)){
          print("l_old = ")
          print(l_old)
          print("l_new = ")
          print(l_new)

          print("curr_tree_z = ")
          print(curr_tree_z)

          print("new_tree_z = ")
          print(new_tree_z)

          print("get_tree_prior(curr_trees_z[[j]], alpha_z, beta_z) = ")
          print(get_tree_prior(curr_trees_z[[j]], alpha_z, beta_z))

          print(" get_tree_prior(new_trees_z[[j]], alpha_z, beta_z) = ")
          print( get_tree_prior(new_trees_z[[j]], alpha_z, beta_z))

          stop(";ome 12-0. alpha_MH NA")

        }

        if(alpha_MH > runif(1)) {
          curr_trees_z[[j]] = new_trees_z[[j]]

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
          l_old = tree_full_conditional(curr_trees_y[[j]],
                                        current_partial_residuals,
                                        phi1,
                                        sigma2_mu_y) +
            get_tree_prior(curr_trees_y[[j]], alpha_y, beta_y)

          # NEW TREE: compute the log of the marginalised likelihood + log of the tree prior
          l_new = tree_full_conditional(new_trees_y[[j]],
                                        current_partial_residuals,
                                        phi1,
                                        sigma2_mu_y) +
            get_tree_prior(new_trees_y[[j]], alpha_y, beta_y)

          alpha_MH = alpha_mh(l_new,l_old, curr_trees_y[[j]],new_trees_y[[j]], type_y)

        }

        if(is.na(alpha_MH)){
          print("l_old = ")
          print(l_old)
          print("l_new = ")
          print(l_new)

          print("curr_tree_y = ")
          print(curr_tree_y)

          print("new_tree_y = ")
          print(new_tree_y)

          print("get_tree_prior(curr_trees_y[[j]], alpha_y, beta_y) = ")
          print(get_tree_prior(curr_trees_y[[j]], alpha_y, beta_y))

          print(" get_tree_prior(new_trees_y[[j]], alpha_y, beta_y) = ")
          print( get_tree_prior(new_trees_y[[j]], alpha_y, beta_y))

        }


        if(alpha_MH > runif(1)) {
          curr_trees_y[[j]] = new_trees_y[[j]]

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

        }


      } # end loop over y trees

      curr_trees_y[[j]] = simulate_mu_all(curr_trees_y[[j]],
                                      current_partial_residuals,
                                      phi1,
                                      sigma2_mu_y)

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
          l_old = tree_full_conditional(curr_trees_y[[j]],
                                        current_partial_residuals,
                                        phi1,
                                        sigma2_mu_y) +
            get_tree_prior(curr_trees_y[[j]], alpha_y, beta_y)

          # NEW TREE: compute the log of the marginalised likelihood + log of the tree prior
          l_new = tree_full_conditional(new_trees_y[[j]],
                                        current_partial_residuals,
                                        phi1,
                                        sigma2_mu_y) +
            get_tree_prior(new_trees_y[[j]], alpha_y, beta_y)

          alpha_MH = alpha_mh(l_new,l_old, curr_trees_y[[j]],new_trees_y[[j]], type_y)

        }

        if(is.na(alpha_MH)){
          print("l_old = ")
          print(l_old)
          print("l_new = ")
          print(l_new)

          print("curr_tree_y = ")
          print(curr_tree_y)

          print("new_tree_y = ")
          print(new_tree_y)

          print("get_tree_prior(curr_trees_y[[j]], alpha_y, beta_y) = ")
          print(get_tree_prior(curr_trees_y[[j]], alpha_y, beta_y))

          print(" get_tree_prior(new_trees_y[[j]], alpha_y, beta_y) = ")
          print( get_tree_prior(new_trees_y[[j]], alpha_y, beta_y))

        }


        if(alpha_MH > runif(1)) {
          curr_trees_y[[j]] = new_trees_y[[j]]

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

        }

        curr_trees_y[[j]] = simulate_mu(curr_trees_y[[j]],
                                                 current_partial_residuals,
                                                 phi1,
                                                 sigma2_mu_y)

        # Updating BART predictions
        current_fit = get_predictions(curr_trees_y[j], x.train[uncens_inds,], single_tree = TRUE)
        mutemp_y = mutemp_y - tree_fits_store_y[,j] # subtract the old fit
        mutemp_y = mutemp_y + current_fit # add the new fit
        tree_fits_store_y[,j] = current_fit # update the new fit

      } # end loop over y trees

    }

    # if(sparse){
    #   tempcounts <- fcount(sampler_y$getTrees()$var)
    #   tempcounts <- tempcounts[tempcounts$x != -1, ]
    #   var_count_y <- rep(0, p_y)
    #   var_count_y[tempcounts$x] <- tempcounts$N
    # }

    #update z_epsilon
    y_epsilon[uncens_inds] <- ystar[uncens_inds] - mutemp_y

    mutemp_test_y <- get_predictions(curr_trees_y,
                                     x.test,
                                     single_tree = length(curr_trees_y) == 1)

    ############# Covariance matrix samples ##########################


    if(cov_prior == "Ding"){
      rho1 <- gamma1/sqrt(Sigma_mat[2,2])

      sigz2 <- 1/rgamma(n = 1,
                        shape = nu0/2,
                        rate = cding/(2*(1- (rho1^2))))


      z_epsilon2 <- sqrt(sigz2)*(z - offsetz - mutemp_z)

      zsquares <- crossprod(z_epsilon2[uncens_inds], z_epsilon2[uncens_inds])[1]
      ysquares <- crossprod(y_epsilon[uncens_inds], y_epsilon[uncens_inds])[1]
      zycross <- crossprod(z_epsilon2[uncens_inds], y_epsilon[uncens_inds])[1]

      Stemp <- cbind(c(ysquares,zycross),
                     c(zycross, zsquares))


      tempsigma <- rinvwishart(nu = n1 + nu0,
                               S = Stemp+cding*diag(2))


      transmat <- cbind(c(1,0),c(0,1/sqrt(tempsigma[2,2])))
      tempomega <- (transmat %*% tempsigma) %*% transmat

      temprho <- tempomega[1,2]/(sqrt(tempomega[1,1]))

      gamma1 <- tempomega[1,2]
      phi1 <- tempomega[1,1] - (gamma1^2)


      # if(tempomega[2,2] != 1){
      #   print("tempomega[2,2] = ")
      #   print(tempomega[2,2])
      # }
      #
      # if(phi1 < 0){
      #   print("phi1 = ")
      #   print(phi1)
      # }


    }else{

      ########### Simultaneous phi and gamma draw #####################

      if(simultaneous_covmat == TRUE){

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

        # G1inv <- (1/G0) + (1/phi1)*crossprod(z_epsilon)
        G1inv <- (1/G0draw) + (1/phi1)*crossprod(z_epsilon[uncens_inds])[1]
        # G1inv <- (1/tau) + (1/phi1)*crossprod(z_epsilon[uncens_inds])
        G1 <- (1/G1inv)#[1,1]

        # gamma_one <- (G1*( (1/G0draw)*gamma0 + (1/phi1)*crossprod(z_epsilon , y_epsilon   )   ))[1,1]
        gamma_one <- (G1*( (1/G0draw)*gamma0 + (1/phi1)*crossprod(z_epsilon[uncens_inds] , y_epsilon[uncens_inds]   )[1]   ))
        # gamma_one <- (G1*( (1/tau)*gamma0 + (1/phi1)*crossprod(z_epsilon[uncens_inds] , y_epsilon[uncens_inds]   )   ))[1,1]

        # print("phi1 = ")
        # print(phi1)
        # print("G0draw = ")
        # print(G0draw)
        #
        # print("(G1*( (1/G0draw)*gamma0 + (1/phi1)*crossprod(z_epsilon[uncens_inds] , y_epsilon[uncens_inds]   )   )) = ")
        # print((G1*( (1/G0draw)*gamma0 + (1/phi1)*crossprod(z_epsilon[uncens_inds] , y_epsilon[uncens_inds]   )   )))
        #
        # print("gamma_one = ")
        # print(gamma_one)
        #
        # print("crossprod(z_epsilon , y_epsilon   ) = ")
        # print(crossprod(z_epsilon , y_epsilon   ))
        #
        # print("crossprod(z_epsilon    ) = ")
        # print(crossprod(z_epsilon   ))
        #
        # print("crossprod(z_epsilon[uncens_inds])[1]     = ")
        # print(crossprod(z_epsilon[uncens_inds])[1]   )
        #
        # print("G1 = ")
        # print(G1)

        gamma1 <- rnorm(n = 1, mean = gamma_one, sd =  sqrt(G1) )

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
          S1 <- S0 + (gamma1^2)/tau +
            crossprod( y_epsilon[uncens_inds] - gamma1*z_epsilon[uncens_inds]  )[1] # + crossprod(y_epsilon)
        }else{
          if(cov_prior == "Omori"){
            S1 <- S0 + #+ (gamma1^2)/G0 +
              crossprod( y_epsilon[uncens_inds] - gamma1*z_epsilon[uncens_inds]  )[1]  #+crossprod(y_epsilon)[1]
          }else{
            mixind <- rbinom(n = 1,size = 1,prob = mixprob)
            if(mixind == 1){
              # S1 <- S0 + (gamma1^2)/tau + gamma1*crossprod( y_epsilon[uncens_inds] - gamma1*z_epsilon[uncens_inds]  )  + crossprod(y_epsilon)
              S1 <- S0 + (gamma1^2)/tau +
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

    Sigma_mat <- cbind(c(1,gamma1),c(gamma1,phi1+gamma1^2))
    Sigma_orig_scale <- cbind(c(1,tempsd*gamma1),c(tempsd*gamma1,  (tempsd^2)*(phi1+gamma1^2)) )


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
      sigma2_mu_z <-  update_sigma_mu_par(curr_trees_z, sigma2_mu_z)
      if(is.na(sigma2_mu_z )){
        stop("Line 1728 sigma2_mu_z  NA")
      }
      sigma2_mu_y <-  update_sigma_mu_par(curr_trees_y, sigma2_mu_y)
      if(is.na(sigma2_mu_y )){
        stop("Line 1733 sigma2_mu_y  NA")
      }
    }


    ###### Store results   ###############################






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


      if(sparse){
        draw$alpha_s_y_store[iter_min_burnin] <- alpha_s_y
        draw$alpha_s_z_store[iter_min_burnin] <- alpha_s_z
        draw$var_count_y_store[iter_min_burnin,] <- var_count_y
        draw$var_count_z_store[iter_min_burnin,] <- var_count_z
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

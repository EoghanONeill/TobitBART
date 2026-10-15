
#' @title Type II Tobit Bayesian Additive Regression Trees with heteroskedasticity (in outcome variance and selection-outcome covariance)
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
#' @param nzero A prior parameter which when divided by 2 gives the shape parameter of the inverse gamma prior on phi, where phi + gamma^2 is the variance of the errors of the outcome model.
#' @param S0 A prior parameter which when divided by 2 gives the scale parameter of the inverse gamma prior on phi, where phi + gamma^2 is the variance of the errors of the outcome model.
#' @param sigest Estimate of standard deviation of the error term.
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
#' @param tree_power_y Tree prior parameter for outcome model.
#' @param tree_base_y Tree prior parameter for outcome model.
#' @param tree_power_z Tree prior parameter for selection model.
#' @param tree_base_z Tree prior parameter for selection model.
#' @param node.prior (dbarts option) An expression of the form dbarts:::normal or dbarts:::normal(k) that sets the prior used on the averages within nodes.
#' @param proposal.probs (dbarts option) Named numeric vector or NULL, optionally specifying the proposal rules and their probabilities. Elements should be "birth_death", "change", and "swap" to control tree change proposals, and "birth" to give the relative frequency of birth/death in the "birth_death" step.
#' @param print.opt Print every print.opt number of Gibbs samples.
#' @param accelerate If TRUE, add extra parameter for accelerated sampler as descibed by Omori (2007).
#' @param cov_prior Prior for the heteroskedastic error covariance matrix. If VH (not yet supported), the covariance tree leaf priors are dependent on the product of trees function, phi, which is included in the outcome error variance (phi + gamma^2). If Omori, apply the prior N(gamma0,G0) to the sum-of-tree function gamma and a HBART product of tree prior to the function phi in the outcome error variance (phi + gamma^2). Mixture and Ding priors are not currently supported.
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
#' @param yinit_resid If TRUE initialize outcome equation trees with linear tobit-2 residualized outcome (for better prior calibration of terminal node parameters)
#' @param draw_censored_predictions If TRUE, draw ystar and posterior conditional mean samples for censored observations. Not possible if missing predictors in x.train for outcome-censored (i.e. unselected) observations.
#' @param node_min_size Minimum number of observations in nodes for phi parameter trees.
#' @export
#' @return The following objects are returned:
#' \item{mu_y_train}{Matrix of draws of the outcome model sums of terminal nodes, i.e. f(x_i), for all training observations. Number of rows equals number of training observations. Number of columns equals n.iter .}
#' \item{mu_y_test}{Matrix of draws of the outcome model sums of terminal nodes, i.e. f(x_i), for all test observations. Number of rows equals number of test observations. Number of columns equals n.iter .}
#' \item{muuncens_y_train}{Matrix of draws of the outcome model sums of terminal nodes, i.e. f(x_i), for all uncensored training observations. Number of rows equals number of uncensored training observations. Number of columns equals n.iter .}
#' \item{mu_z_train}{Matrix of draws of the censoring model sums of terminal nodes, i.e. f(w_i), for all training observations. Number of rows equals number of training observations. Number of columns equals n.iter .}
#' \item{mu_z_test}{Matrix of draws of the censoring model sums of terminal nodes, i.e. f(w_i), for all test observations. Number of rows equals number of test observations. Number of columns equals n.iter .}
#' \item{train.probcens}{Matrix of draws of probabilities of training sample observations being censored. Number of rows equals number of training observations. Number of columns equals n.iter .}
#' \item{test.probcens}{Matrix of draws of probabilities of test sample observations being censored. Number of rows equals number of test observations. Number of columns equals n.iter .}
#' \item{cond_exp_train}{Matrix of draws of the conditional (i.e. possibly censored) expectations of the outcome for all training observations. Number of rows equals number of training observations. Number of columns equals n.iter .}
#' \item{cond_exp_test}{Matrix of draws of the conditional (i.e. possibly censored) expectations of the outcome for all test observations. Number of rows equals number of test observations. Number of columns equals n.iter .}
#' \item{uncond_exp_train}{Only defined if censored_value is a number. Matrix of draws of the unconditional (i.e. possibly censored) expectations of the outcome for all training observations. Number of rows equals number of training observations. Number of columns equals n.iter .}
#' \item{uncond_exp_test}{Only defined if censored_value is a number. Matrix of draws of the unconditional (i.e. possibly censored) expectations of the outcome for all test observations. Number of rows equals number of test observations. Number of columns equals n.iter .}
#' \item{ystar_train}{Matrix of training sample draws of the outcome assuming uncensored. Fixed at true values for selected/uncensored observations. Number of rows equals number of training observations. Number of columns equals n.iter .}
#' \item{ystar_test}{Matrix of test sample draws of the outcome assuming uncensored . Number of rows equals number of test observations. Number of columns equals n.iter .}
#' \item{zstar_train}{Matrix of training sample draws of the censoring model latent outcome. Number of rows equals number of training observations. Number of columns equals n.iter.}
#' \item{zstar_test}{Matrix of test sample draws of the censoring model latent outcome. Number of rows equals number of test observations. Number of columns equals n.iter.}
#' \item{ydraws_test}{Only defined if censored_value is a number. Matrix of test sample unconditional (i.e. possibly censored) draws of the outcome . Number of rows equals number of test observations. Number of columns equals n.iter .}
#' \item{ycond_draws_train}{List of training sample conditional (i.e. zstar >0 for draw) draws of the outcome. Number of rows equals number of training observations. Number of columns equals n.iter .}
#' \item{ycond_draws_test}{List of test sample conditional (i.e. zstar >0 for draw) draws of the outcome . Number of rows equals number of test observations. Number of columns equals n.iter .}
#' \item{alpha_s_y_store}{For Dirichlet prior on splitting probabilities in outcome equation, vector of alpha hyperparameter draws for each iteration.}
#' \item{alpha_s_z_store}{For Dirichlet prior on splitting probabilities in selection equation, vector of alpha hyperparameter draws for each iteration }
#' \item{var_count_y_store}{Matrix of counts of splits on each variable in outcome observation. The number of columns is the number of potential splitting variables. The number of rows is the number of post-burn-in iterations.}
#' \item{var_count_z_store}{Matrix of counts of splits on each variable in selection observation. The number of columns is the number of potential splitting variables. The number of rows is the number of post-burn-in iterations. }
#' \item{s_prob_y_store}{Splitting probabilities for the outcome equation. The number of columns is the number of potential splitting variables. The number of rows is the number of post-burn-in iterations. }
#' \item{s_prob_z_store}{Splitting probabilities for the selection equation. The number of columns is the number of potential splitting variables. The number of rows is the number of post-burn-in iterations. }
#' @export

htbart2 <- function(x.train,
                    x.test,
                    w.train,
                    w.test,
                    v.train,
                    v.test,
                    y,
                    n.iter=1000,
                    n.burnin=100,
                    censored_value = NA,
                    gamma0_prior = NA,
                    # G0=1,
                    nzero = 6,#0.002,
                    S0 = NA,# 12 #0.002,
                    SGratio = 9,
                    sigest = NA,
                    n.trees_outcome = 200L,
                    n.trees_censoring = 200L,
                    n.trees_phi = 50L,
                    n.trees_gamma = 50L,
                    # n.burn = 0L,
                    # n.samples = 1L,
                    # n.thin = 1L,
                    # n.chains = 1L,
                    # n.threads = 1L, #guessNumCores(),
                    # printEvery = 100L,
                    # printCutoffs = 0L,
                    rngKind = "default",
                    rngNormalKind = "default",
                    rngSeed = NA_integer_,
                    updateState = TRUE,
                    tree_power_z = 2,
                    tree_power_y = 2,
                    tree_power_phi = 2,
                    tree_power_gamma = 2,
                    tree_base_z = 0.95,
                    tree_base_y = 0.95,
                    tree_base_phi = 0.95,
                    tree_base_gamma = 0.95,
                    node.prior = dbarts:::normal,
                    # resid.prior = dbarts:::chisq,
                    proposal.probs = c(birth_death = 0.5, swap = 0, change = 0.5, birth = 0.5),
                    # sigmadbarts = NA_real_,
                    print.opt = 100,
                    # accelerate = FALSE,
                    cov_prior = "Omori",
                    tau = 5,
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
                    alpha_a_phi = 0.5,
                    alpha_b_phi = 1,
                    alpha_a_gamma = 0.5,
                    alpha_b_gamma = 1,
                    alpha_split_prior = TRUE,
                    tau_hyperprior = FALSE, alpha_tau = 1, beta_tau = 10,
                    yinit_resid = FALSE,
                    draw_censored_predictions = TRUE,
                    fixed_resp_update = TRUE,
                    node_min_size = 5,
                    max_bad_trees = 10,
                    splitting_rules = "discrete",
                    linHeck_centring = TRUE){

  if (nrow(x.test) == 0L) {stop("x.test with zero rows is currently not supported; pass training rows if no test set is needed.")}
  if(!(cov_prior %in% c("VH","Omori","Mixture", "Ding"))){stop("cov_prior must equal VH, Omori, Mixture, or Ding")}
  if(cov_prior == "Mixture"){stop("Experimental option for Mixture cov_prior not currently supported.")}
  if((mixprob < 0)| (mixprob > 1) ){stop("mixprob must be between zero and one.")}

  if (cov_prior != "Omori") {
    stop("The current heterogeneous gamma forest implements the independent G0 prior only.")
  }

  if(is.vector(x.train) | is.factor(x.train)| is.data.frame(x.train)) x.train = as.matrix(x.train)
  if(is.vector(x.test) | is.factor(x.test)| is.data.frame(x.test)) x.test = as.matrix(x.test)
  if(is.vector(w.train) | is.factor(w.train)| is.data.frame(w.train)) w.train = as.matrix(w.train)
  if(is.vector(w.test) | is.factor(w.test)| is.data.frame(w.test)) w.test = as.matrix(w.test)
  if(is.vector(v.train) | is.factor(v.train)| is.data.frame(v.train)) v.train = as.matrix(v.train)
  if(is.vector(v.test) | is.factor(v.test)| is.data.frame(v.test)) v.test = as.matrix(v.test)
  if (length(gamma0_prior) != 1L) {stop("gamma0_prior must be a numeric scalar or NA.")}
  if (length(sigest) != 1L) {stop("sigest must be a numeric scalar or NA.")}
  if (length(S0) != 1L) {stop("S0 must be a numeric scalar or NA.")}
  if (length(censored_value) != 1L) {stop("censored_value must be a scalar or NA.")}
  if (!is.finite(node_min_size) ||node_min_size < 1 ||node_min_size != floor(node_min_size)) {
    stop("node_min_size must be a positive integer.")
  }
  x.train <- as.matrix(x.train)
  x.test  <- as.matrix(x.test)
  w.train <- as.matrix(w.train)
  w.test  <- as.matrix(w.test)
  v.train <- as.matrix(v.train)
  v.test  <- as.matrix(v.test)
  y <- as.numeric(y)
  storage.mode(x.train) <- "double"
  storage.mode(x.test)  <- "double"
  storage.mode(w.train) <- "double"
  storage.mode(w.test)  <- "double"
  storage.mode(v.train) <- "double"
  storage.mode(v.test)  <- "double"

  # if((!is.matrix(x.train))) stop("argument x.train must be a double matrix")
  # if((!is.matrix(x.test)) ) stop("argument x.test must be a double matrix")
  if(nrow(x.train) != length(y)) stop("number of rows in x.train must equal length of y.train")
  if(nrow(w.train) != length(y)) stop("number of rows in x.train must equal length of y.train")
  if(nrow(v.train) != length(y)) stop("number of rows in x.train must equal length of y.train")
  if((ncol(x.test)!=ncol(x.train))) stop("input x.test must have the same number of columns as x.train")
  if((ncol(w.test)!=ncol(w.train))) stop("input w.test must have the same number of columns as w.train")
  if((ncol(v.test)!=ncol(v.train))) stop("input v.test must have the same number of columns as v.train")
  # if all equal to length(y), these lines are unnecessary.
  if((nrow(w.test)!=nrow(x.test))) stop("input w.test must have the same number of rows as x.test")
  if((nrow(v.test)!=nrow(x.test))) stop("input v.test must have the same number of rows as x.test")
  # if((nrow(w.train)!=nrow(x.train))) stop("input w.train must have the same number of rows as x.train")
  # if((nrow(w.train)!=nrow(x.train))) stop("input w.train must have the same number of rows as x.train")

  #indexes of censored observations
  if(is.na(censored_value)){
    cens_inds <- which(is.na(y))
    uncens_inds <- which(!(is.na(y)))
  }else{
    cens_inds <- which(y == censored_value)
    uncens_inds <- which(y != censored_value)
  }
  if(length(cens_inds)==0) stop("No censored observations")
  if (sd(y[uncens_inds]) <= 0 || !is.finite(sd(y[uncens_inds]))) {stop("Uncensored outcomes must have positive finite standard deviation.")}


  if ( anyNA(w.train)|| anyNA(w.test) ) {stop("Missing covariate values in w.train or w.test.")}
  if ( anyNA(v.train)|| anyNA(v.test) ) {stop("Missing covariate values in v.train or v.test.")}
  if(draw_censored_predictions){
    if (anyNA(x.train)  || anyNA(x.test) ) {stop("There are missing covariate values in x.train or x.test. This is not allowed with draw_censored_predictions == TRUE. If draw_censored_predictions == FALSE, Censored/training observations with missing predictors are supported, but predictors are required for selected or test observations.")}
  }else{
    if (anyNA(x.train[uncens_inds,])  || anyNA(x.test) ) {stop("There are missing covariate values in uncensored x.train or any of x.test. Censored/training observations with missing predictors are supported, but predictors are required for selected or test observations.")}
  }

  n <- length(y)
  n0 <- length(cens_inds)
  n1 <- length(uncens_inds)
  ntest = nrow(x.test)

  # normalize the outcome
  tempmean <- mean(y[uncens_inds])
  tempsd <- sd(y[uncens_inds])
  if(linHeck_centring){
    if(is.na(sigest)) {
      if(ncol(x.train) < n1) {
        if(is.na(censored_value)){
          dtemp <-  1*(!(is.na(y)))
        }else{
          dtemp <-  1*(y != censored_value)
        }
        df = data.frame(x = cbind(x.train,w.train), y = y, d = dtemp )
        seleq <- paste0("d ~ " , paste(colnames(df)[(ncol(x.train)+1):(ncol(x.train) + ncol(w.train))],
                                       collapse = " + "))
        outeq <- paste0("y ~ " , paste(colnames(df)[1:ncol(x.train)],
                                       collapse = " + "))
        fit_ok <- FALSE
        if (is.na(sigest) && ncol(x.train) < n1) {
          heckit.ml <- tryCatch(
            sampleSelection::heckit(
              selection = as.formula(seleq),
              outcome   = as.formula(outeq),
              data = df,
              method = "ml"
            ),
            error = function(e) NULL
          )
          if (!is.null(heckit.ml) && !is.null(heckit.ml$estimate) && all(c("rho", "sigma") %in% names(heckit.ml$estimate))) {
            rho_hat <- heckit.ml$estimate["rho"]
            sigma_hat <- heckit.ml$estimate["sigma"]
            if (length(rho_hat) == 1L && length(sigma_hat) == 1L && is.finite(rho_hat) && is.finite(sigma_hat) && sigma_hat > 0) {
              zhat_candidate <- tryCatch(as.numeric(predict(heckit.ml,newdata = df,part = "selection",type = "link")),error = function(e) NULL)
              if (!is.null(zhat_candidate) && length(zhat_candidate) == nrow(df) && all(is.finite(zhat_candidate))) {
                tempsd <- sigma_hat
                fit_ok <- TRUE
                y_initpreds <- predict(heckit.ml, newdata = df, part = "outcome", type = "unconditional")
                tempmean <- mean(y_initpreds)
              }
            }
          }
        }
        if (!fit_ok) {
          if(is.na(censored_value)){
            dtemp <-  1*(!(is.na(y)))
          }else{
            dtemp <-  1*(y != censored_value)
          }
          df = data.frame(y = y, d = dtemp ,intercept_covariate = rep(0, length(y)))
          seleq <- "d ~ 1"
          outeq <- "y ~ 1"
          fit_ok2 <- FALSE
          if (is.na(sigest) && ncol(x.train) < n1) {
            heckit.ml <- tryCatch(
              sampleSelection::heckit(
                selection = as.formula(seleq),
                outcome   = as.formula(outeq),
                data = df,
                method = "ml"
              ),
              error = function(e) NULL
            )
            if (!is.null(heckit.ml) && !is.null(heckit.ml$estimate) && all(c("rho", "sigma") %in% names(heckit.ml$estimate))) {
              rho_hat <- heckit.ml$estimate["rho"]
              sigma_hat <- heckit.ml$estimate["sigma"]
              if (length(rho_hat) == 1L && length(sigma_hat) == 1L && is.finite(rho_hat) && is.finite(sigma_hat) && sigma_hat > 0) {
                zhat_candidate <- tryCatch(as.numeric(predict(heckit.ml,newdata = df,part = "selection",type = "link")),error = function(e) NULL)
                if (!is.null(zhat_candidate) && length(zhat_candidate) == nrow(df) && all(is.finite(zhat_candidate))) {
                  tempsd <- sigma_hat
                  fit_ok2 <- TRUE
                  y_initpreds <- predict(heckit.ml, newdata = df, part = "outcome", type = "unconditional")
                  tempmean <- mean(y_initpreds)
                }
              }
            }
          }
          if (!fit_ok2) {
            tempsd <- sd(y[uncens_inds])
            # y_initpreds <- rep(mean(y[uncens_inds]), nrow(df))
            tempmean <- mean(y[uncens_inds])
          }
          # tempsd <- sd(y[uncens_inds])
          # # y_initpreds <- rep(mean(y[uncens_inds]), nrow(df))
          # tempmean <- mean(y[uncens_inds])
        }
      } else {
        if(is.na(censored_value)){
          dtemp <-  1*(!(is.na(y)))
        }else{
          dtemp <-  1*(y != censored_value)
        }
        df = data.frame(y = y, d = dtemp )
        seleq <- "d ~ ."
        outeq <- "y ~ ."
        fit_ok <- FALSE
        if (is.na(sigest) && ncol(x.train) < n1) {
          heckit.ml <- tryCatch(
            sampleSelection::heckit(
              selection = as.formula(seleq),
              outcome   = as.formula(outeq),
              data = df,
              method = "ml"
            ),
            error = function(e) NULL
          )
          if (!is.null(heckit.ml) && !is.null(heckit.ml$estimate) && all(c("rho", "sigma") %in% names(heckit.ml$estimate))) {
            rho_hat <- heckit.ml$estimate["rho"]
            sigma_hat <- heckit.ml$estimate["sigma"]
            if (length(rho_hat) == 1L && length(sigma_hat) == 1L && is.finite(rho_hat) && is.finite(sigma_hat) && sigma_hat > 0) {
              zhat_candidate <- tryCatch(as.numeric(predict(heckit.ml,newdata = df,part = "selection",type = "link")),error = function(e) NULL)
              if (!is.null(zhat_candidate) && length(zhat_candidate) == nrow(df) && all(is.finite(zhat_candidate))) {
                tempsd <- sigma_hat
                fit_ok <- TRUE
                y_initpreds <- predict(heckit.ml, newdata = df, part = "outcome", type = "unconditional")
                tempmean <- mean(y_initpreds)
              }
            }
          }
        }
        if (!fit_ok) {
          tempsd <- sd(y[uncens_inds])
          # y_initpreds <- rep(mean(y[uncens_inds]), nrow(df))
          tempmean <- mean(y[uncens_inds])
        }
      }
    }else{
      tempmean <- mean(y[uncens_inds])
      tempsd <- sigest
    }
  }

  originaly <- y
  y <- (y-tempmean)/tempsd
  if (!is.na(sigest)) { sigest <- sigest / tempsd}
  if (!is.na(gamma0_prior)) {gamma0_prior <- gamma0_prior / tempsd}
  if (!is.na(S0)) {S0 <- S0 / tempsd^2}

  has_numeric_censor <- length(censored_value) == 1L && !is.na(censored_value) && is.numeric(censored_value)
  if(has_numeric_censor){
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
  ecdfsv   <- list()
  for(i in 1:ncol(v.train)) {
    ecdfsv[[i]] <- ecdf(v.train[,i])
    if(length(unique(v.train[,i])) == 1) ecdfsv[[i]] <- identity
    if(length(unique(v.train[,i])) == 2) ecdfsv[[i]] <- make_01_norm(v.train[,i])
  }
  for(i in 1:ncol(v.train)) {
    v.train[,i] <- ecdfsv[[i]](v.train[,i])
    v.test[,i] <- ecdfsv[[i]](v.test[,i])
  }
  rm(ecdfsv)

  xdf_y_all <- data.frame(x = x.train)
  xdf_gamma_all <- data.frame(x = v.train)
  xdf_phi_all <- data.frame(x = v.train)
  #create z vector
  #create ystar vector
  ystar <- rep(NA, length(y))
  ystar[uncens_inds] <- y[uncens_inds]


  if (n1 < 2L) {stop("At least two uncensored observations are required to construct gamma_reference.")}

  p_y <- ncol(x.train)
  p_z <- ncol(w.train)
  p_gamma <- ncol(v.train)
  p_phi <- ncol(v.train)

  s_y <- rep(1 / p_y, p_y) # probability vector to be used during the growing process for DART feature weighting
  s_z <- rep(1 / p_z, p_z) # probability vector to be used during the growing process for DART feature weighting
  s_gamma <- rep(1 / p_gamma, p_gamma) # probability vector to be used during the growing process for DART feature weighting
  s_phi <- rep(1 / p_phi, p_phi) # probability vector to be used during the growing process for DART feature weighting

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

    rho_gamma <- p_gamma # For DART
    if(alpha_split_prior){
      alpha_s_gamma <- p_gamma
    }else{
      alpha_s_gamma <- 1
    }
    alpha_scale_gamma <- p_gamma

    rho_phi <- p_phi # For DART
    if(alpha_split_prior){
      alpha_s_phi <- p_phi
    }else{
      alpha_s_phi <- 1
    }
    alpha_scale_phi <- p_phi
  }
  if(offsetz){
    offsetz <- qnorm(n1/n)
  }else{
    offsetz <- 0 # qnorm(n1/n)
  }

  z <- rep(offsetz, length(y))
  # z[cens_inds] <- qnorm(0.001) #rtruncnorm(n0, a= -Inf, b = 0, mean = offsetz, sd = 1)
  # z[uncens_inds] <- qnorm(0.999) #rtruncnorm(n1, a= 0, b = Inf, mean = offsetz, sd = 1)
  z[cens_inds] <- rtruncnorm(n0, a= -Inf, b = 0, mean = offsetz, sd = 1)
  z[uncens_inds] <- rtruncnorm(n1, a= 0, b = Inf, mean = offsetz, sd = 1)
  # z <- rnorm(n = length(y), mean = offsetz, sd =1)

  # if(is.null(S0)){
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
  #   S0 <- 2*(sigest^2 -   (1/(8*(G0^2))) - 4*(gamma0^2)*G0   )
  #   # S0 <- 2*(sigest^2)# -   (1/(8*(G0^2))) - 4*(gamma0^2)*G0   )
  #   # S0 <- 0.002
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
      # colnames(df)[1:ncol(x.train)] <- paste("x",1:ncol(x.train),sep = ".")
      # colnames(df)[(ncol(x.train)+1):(ncol(x.train) + ncol(w.train))] <- paste("x",(ncol(x.train)+1):(ncol(x.train) + ncol(w.train)),sep = ".")
      # seleq <- paste0("d ~ " , paste(paste("x",(ncol(x.train)+1):(ncol(x.train) + ncol(w.train)),sep = "."),collapse = " + "))
      # outeq <- paste0("y ~ " , paste(paste("x",1:ncol(x.train),sep = "."),collapse = " + "))
      seleq <- paste0("d ~ " , paste(colnames(df)[(ncol(x.train)+1):(ncol(x.train) + ncol(w.train))],
                                     collapse = " + "))
      outeq <- paste0("y ~ " , paste(colnames(df)[1:ncol(x.train)],
                                     collapse = " + "))
      # heckit.ml <- heckit(selection = as.formula(seleq),
      #                     outcome = as.formula(outeq),
      #                     data = df,
      #                     method = "ml")
      # correst <- heckit.ml$estimate["rho"]
      # sigest <- heckit.ml$estimate["sigma"]
      fit_ok <- FALSE
      if (is.na(sigest) && ncol(x.train) < n1) {
        heckit.ml <- tryCatch(
          sampleSelection::heckit(
            selection = as.formula(seleq),
            outcome   = as.formula(outeq),
            data = df,
            method = "ml"
          ),
          error = function(e) NULL
        )
        if (!is.null(heckit.ml) && !is.null(heckit.ml$estimate) && all(c("rho", "sigma") %in% names(heckit.ml$estimate))) {
          rho_hat <- heckit.ml$estimate["rho"]
          sigma_hat <- heckit.ml$estimate["sigma"]
          if (length(rho_hat) == 1L && length(sigma_hat) == 1L && is.finite(rho_hat) && is.finite(sigma_hat) && sigma_hat > 0) {
            zhat_candidate <- tryCatch(as.numeric(predict(heckit.ml,newdata = df,part = "selection",type = "link")),error = function(e) NULL)
            if (!is.null(zhat_candidate) && length(zhat_candidate) == nrow(df) && all(is.finite(zhat_candidate))) {
              correst <- max(min(rho_hat, 0.99), -0.99)
              sigest <- sigma_hat
              gamma0 <- correst * sigest
              zhat_heckman <- zhat_candidate
              fit_ok <- TRUE
            }
          }
        }
      }
      if (!fit_ok) {
        sigest <- sd(y[uncens_inds])
        correst <- 0
        gamma0 <- 0
        zhat_heckman <- rep(0, nrow(df))
      }
      # correst <- heckit.2step$coefficients["rho"]
      # sigest <- heckit.2step$coefficients["sigma"]
      gamma0 <- correst*sigest
    } else {
      sigest = sd(y[uncens_inds])
      correst <- 0
      gamma0 <- 0
      zhat_heckman <- rep(0, nrow(w.train))
    }
  }else{
    correst <- 0
    gamma0 <- 0
    zhat_heckman <- rep(0, nrow(w.train))
  }
  # if(is.null(nzero)){
  #   nzero <- 2*(sigest^2)
  # }
  # set prior mean for gamma
  if (cov_prior == "Ding") {
    if (!is.na(gamma0_prior) && gamma0_prior != 0) {
      print("Ding prior selected, therefore setting gamma0 prior mean to 0 instead of user-specified gamma0_prior.")
    }
    gamma0 <- 0
  }else{if(!is.na(gamma0_prior)) {gamma0 <- gamma0_prior}}

  #set initial sigma
  #alternatively, draw this from the prior
  Sigma_mat <- cbind(c(1,0),c(0,sigest^2))

  #set initial gamma
  gamma1 <- 0 # correst*sigest  #0#cov(ystar,z)
  #set initial phi
  phi1 <- sigest^2 - gamma1^2

  # if(sigest > G0){
  #   S0 <- (nzero - 2)*(sigest-G0)
  # }else{
  # sigquant <- 0.9
  # qchi <- qchisq(1.0-sigquant,nzero/2)
  # S0 <- 2*(sigest*sigest*qchi)/(nzero/2)
  # }

  # S0 <- (sigest^2)*(nzero-2)/(1+tau)
  # S0 <- (sigest^2)*(1 - correst^2)*(nzero-2)/(1+tau)
  # S0 <- 0.5*(sigest^2 - gamma0^2)*(nzero-2)/(2+tau)
  if(!is.na(S0)){
    # keep S0 as user-specified value
    G0 <- S0/SGratio
  }else{
    if(cov_prior == "Omori"){
      # S0 <- (sigest^2)/( 1/(nzero-2) +  (1 + gamma0^2)/SGratio)
      if(sigest^2 > gamma0^2){
        S0 <- (sigest^2 - gamma0^2)/( 1/(nzero-2) +  1/SGratio)
      }else{
        S0 <- (sigest^2)/( 1/(nzero-2) +  1/SGratio)
      }
      G0 <- S0/SGratio
      # # alternative based on moment matching with van Hasselt prior
      # S0 <- (sigest^2 - gamma0^2)*(nzero-2)/(1+tau)
      # G0 <-  tau*(sigest^2 - gamma0^2)/(1+tau) # (  (sigest^2 - gamma0^2)*tau/(1+tau) + gamma0^2) /(1 + gamma0^2)
    }else{
      if(cov_prior == "VH"){
        # S0 <- (sigest^2)*(1 - correst^2)*(nzero-2)/(1+tau)
        if(sigest^2 > gamma0^2){
          S0 <- (sigest^2 - gamma0^2)*(nzero-2)/(1+tau)

          # if really implemented with VH prior, there should not be a parameter F0
          G0 <- tau * S0 / (nzero - 2)
        }else{
          S0 <- (sigest^2)*(1 - correst^2)*(nzero-2)/(1+tau)
          # if really implemented with VH prior, there should not be a parameter F0
          G0 <- tau * S0 / (nzero - 2)
        }
        # relying on user-specified tau.
        # alternative would be estimated tau, either from moment matching with Omori prior given SGratio
        # tau <- ((S0/SGratio)*(1+gamma0)-gamma0^2)*(nzero-2)/S0
        # or from linear Heckman model estimates
        # if(exists("heckit.ml")){
        #   if(heckit.ml$method == "ml"){
        #     # delta method to obtain estiamted variance of gamma hat
        #     vargammahat <- model1$estimate["rho"]*model1$hessian["sigma", "sigma"] +
        #       model1$hessian["rho", "rho"]*model1$estimate["sigma"]^2 +
        #       2*model1$estimate["rho"]*model1$estimate["sigma"]*model1$hessian["rho", "sigma"]
        #     tau <- vargammahat/(model1$hessian["sigma", "sigma"] - vargammahat - gamma0)
        #   }else{
        #     tau <- ((S0/SGratio)*(1+gamma0)-gamma0^2)*(nzero-2)/S0
        #   }
        # }else{
        #   tau <- ((S0/SGratio)*(1+gamma0)-gamma0^2)*(nzero-2)/S0
        # }
      }else{
        S0 <- (sigest^2)*(1 - correst^2)*(nzero-2)/(1+tau)
        G0 <- S0/SGratio
      }
    }
  }
  #
  # # # alternative: calibrate prior on phi as if gamma equals zero
  # # qchi = qchisq(1.0-quantsig,nzero)
  # # lambda = (sigest*sigest*qchi)/nzero #lambda parameter for sigma prior
  # # S0 <- nzero*lambda
  # # S0 <- 2*(sigest^2) * (nzero/2 - 1)
  #

  # S0 <- 2
  # nzero <- 2

  if(cov_prior == "Ding"){
    gamma0 <- 0
    # sigquant <- 0.9
    # qchi <- qchisq(1.0-quantsig,nu0-1)
    # cdivnu <- (sigest*sigest*qchi)/(nu0-1) #lambda parameter for sigma prior
    # cding <- cdivnu*(nu0-1)
    qig_noscale <- qgamma(p =  1- quantsig, shape = (nu0-1)/2, rate = 1/2)
    cding <- sigest*sigest*qig_noscale
    gamma1 <- correst*sigest  #0#cov(ystar,z)
    phi1 <- sigest^2 - gamma1^2
    if (!is.finite(phi1) || phi1 <= 0) {
      stop("Invalid Ding initialization: phi1 must be positive.")
    }
    # rhoinit <- 0
    # siginit <- sigest
    Sigma_mat <- cbind(c(1,gamma1),c(gamma1,phi1 + gamma1^2))
  }

  # if(cov_prior == "Omori"){
  #   # S0 <- (sigest^2 - G0*(1 + gamma0^2))*(nzero-2)
  #   # can be negative if G0 not chosen appropriately
  #   S0 <- (sigest^2)*(1 - correst^2)*(nzero-2)/(1+tau)
  #   G0 <- tau*S0/(nzero-2) # tau*E[phi]
  # }



  # before initializing sums of trees
  gamma1_vec_train <- rep(gamma0,n)
  gamma1_vec_test <- rep(gamma0,ntest)
  phi1_vec_train <- rep(phi1, n)
  phi1_vec_test <- rep(phi1, ntest)



  draw = list(
    # Z.mat_train = array(NA, dim = c(n, n.iter)),
    # Z.mat_test = array(NA, dim = c(ntest, n.iter)),
    # Y.mat_train = array(NA, dim = c(n, n.iter)),
    # Y.mat_test = array(NA, dim = c(ntest, n.iter)),
    # mu_y_train = array(NA, dim = c(n, n.iter)),# array(NA, dim = c(n, n.iter)),
    mu_y_test = array(NA, dim = c(ntest, n.iter)),
    # mucens_y_train = array(NA, dim = c(n0, n.iter)),
    muuncens_y_train = array(NA, dim = c(n1, n.iter)),
    mu_z_train = array(NA, dim = c(n, n.iter)),
    mu_z_test = array(NA, dim = c(ntest, n.iter)),
    train.probcens =  array(NA, dim = c(n, n.iter)),#array(NA, dim = c(n, n.iter)),#,
    test.probcens =  array(NA, dim = c(ntest, n.iter)),#,
    cond_exp_train_uncens = array(NA, dim = c(n1, n.iter)),#cond_exp_train = array(NA, dim = c(n, n.iter)),
    cond_exp_test = array(NA, dim = c(ntest, n.iter)),
    ystar_test = array(NA, dim = c(ntest, n.iter)),
    zstar_train = array(NA, dim = c(n, n.iter)),
    zstar_test = array(NA, dim = c(ntest, n.iter)),
    ycond_draws_train = list(),
    ycond_draws_test = list(),
    ycond_test_matrix =  array(NA, dim = c(ntest, n.iter)),
    gammavec_draws_train = array(NA, dim = c(n, n.iter)),
    phivec_draws_train = array(NA, dim = c(n, n.iter)),
    gammavec_draws_test = array(NA, dim = c(ntest, n.iter)),
    phivec_draws_test = array(NA, dim = c(ntest, n.iter))
  )

  if(draw_censored_predictions){
    draw$ystar_train = array(NA, dim = c(n, n.iter))
    draw$mu_y_train = array(NA, dim = c(n, n.iter))# array(NA, dim = c(n, n.iter)),
    draw$cond_exp_train = array(NA, dim = c(n, n.iter))#cond_exp_train = array(NA, dim = c(n, n.iter)),
  }

  if(has_numeric_censor){
    if(draw_censored_predictions){
      draw$uncond_exp_train <- array(NA, dim = c(n, n.iter)) #array(NA, dim = c(n, n.iter))
    }
    draw$uncond_exp_train_uncens <- array(NA, dim = c(n1, n.iter)) #array(NA, dim = c(n, n.iter))
    draw$uncond_exp_test <- array(NA, dim = c(ntest, n.iter))
    # draw$ydraws_train <- array(NA, dim = c(n, n.iter))
    draw$ydraws_test <- array(NA, dim = c(ntest, n.iter))
  }

  draw$var_count_y_store <- matrix(0, ncol = p_y, nrow = n.iter)
  draw$var_count_z_store <- matrix(0, ncol = p_z, nrow = n.iter)
  draw$var_count_gamma_store <- matrix(0, ncol = p_gamma, nrow = n.iter)
  draw$var_count_phi_store <- matrix(0, ncol = p_phi, nrow = n.iter)
  var_count_y <- rep(0, p_y)
  var_count_z <- rep(0, p_z)
  var_count_gamma <- rep(0, p_gamma)
  var_count_phi <- rep(0, p_phi)
  if(sparse){
    draw$alpha_s_y_store <- rep(NA, n.iter)
    draw$alpha_s_z_store <- rep(NA, n.iter)
    draw$alpha_s_gamma_store <- rep(NA, n.iter)
    draw$alpha_s_phi_store <- rep(NA, n.iter)
    draw$s_prob_y_store <- matrix(0, ncol = p_y, nrow = n.iter)
    draw$s_prob_z_store <- matrix(0, ncol = p_z, nrow = n.iter)
    draw$s_prob_gamma_store <- matrix(0, ncol = p_gamma, nrow = n.iter)
    draw$s_prob_phi_store <- matrix(0, ncol = p_phi, nrow = n.iter)
  }

  if(tau_hyperprior){
    draw$tau_par <- rep(NA, n.iter)
  }

  ########## Initialize dbarts #####################

  rngSeed_z <- rngSeed
  rngSeed_y <- if (is.na(rngSeed)) NA_integer_ else as.integer(rngSeed + 104729L)
  rngSeed_gamma <- if (is.na(rngSeed)) NA_integer_ else as.integer(rngSeed + 2*104729L)

  control_z <- dbartsControl(updateState = updateState, verbose = FALSE,  keepTrainingFits = TRUE,
                             keepTrees = TRUE,
                             n.trees = n.trees_censoring,
                             n.burn = 0L, # n.burn,
                             n.samples = 1L, # n.samples,
                             n.thin = 1L, # n.thin,
                             n.chains = 1L, # n.chains,
                             n.threads = 1L, # n.threads,
                             printEvery = print.opt, # printEvery,
                             printCutoffs = 0L, # printCutoffs,
                             rngKind = rngKind,
                             rngNormalKind = rngNormalKind,
                             rngSeed = rngSeed_z)

  control_gamma <- dbartsControl(updateState = updateState, verbose = FALSE,  keepTrainingFits = TRUE,
                             keepTrees = TRUE,
                             n.trees = n.trees_gamma,
                             n.burn = 0L, # n.burn,
                             n.samples = 1L, # n.samples,
                             n.thin = 1L, # n.thin,
                             n.chains = 1L, # n.chains,
                             n.threads = 1L, # n.threads,
                             printEvery = print.opt, # printEvery,
                             printCutoffs = 0L, # printCutoffs,
                             rngKind = rngKind,
                             rngNormalKind = rngNormalKind,
                             rngSeed = rngSeed_gamma)


  control_y <- dbartsControl(updateState = updateState, verbose = FALSE,  keepTrainingFits = TRUE,
                             keepTrees = TRUE,
                             n.trees = n.trees_outcome,
                             n.burn = 0L, # n.burn,
                             n.samples = 1L, # n.samples,
                             n.thin = 1L, # n.thin,
                             n.chains = 1L, # n.chains,
                             n.threads = 1L, # n.threads,
                             printEvery = print.opt, # printEvery,
                             printCutoffs = 0L, # printCutoffs,
                             rngKind = rngKind,
                             rngNormalKind = rngNormalKind,
                             rngSeed = rngSeed_y)

  weightstemp_z <- rep(1,n)
  # weightstemp_z[uncens_inds] <- (gamma1^2 + phi1)/phi1
  weightstemp_z[uncens_inds] <- (gamma1_vec_train[uncens_inds]^2 + phi1_vec_train[uncens_inds])/phi1_vec_train[uncens_inds]

  # weightstemp_z[uncens_inds] <- ifelse((gamma1_vec_train[uncens_inds]^2 + phi1_vec_train[uncens_inds]) < 1e-6, 0, weightstemp_z[uncens_inds]  )

  # weightstemp_y <- rep(1,n1)
  weightstemp_y  <- 1/phi1_vec_train[uncens_inds]

  # better to initialize with  sigest^2 - gamma1^2
  # but not obvious how to do this with dbarts
  # perhaps edit sampler object after created?
  # or use adaptive hyperprior option for terminal node parameter.
  # node.prior_z <- node.prior
  # node.prior_y <- node.prior
  yinit_vec <- ystar[uncens_inds]
  if(yinit_resid == TRUE){
    yinit_vec <- ystar[uncens_inds] - gamma0*(z[uncens_inds] - offsetz - zhat_heckman[uncens_inds])
    # node.prior_y = dbarts:::normal
  }

  ## Robust scale of the initial standardized outcome
  y_reference_half_range <- 0.5 * diff(quantile(yinit_vec,probs = c(0.025, 0.975),na.rm = TRUE,names = FALSE))
  if (!is.finite(y_reference_half_range) ||y_reference_half_range <= 0) {
    y_reference_half_range <- 2
  }
  y_reference <- c(-y_reference_half_range,y_reference_half_range,rep(0, n1 - 2L))

  # y_reference <- yinit_vec

  Z_initial <-  z - offsetz
  reference_half_range <- 3.0
  # Z_reference <- z - offsetz
  if(fixed_resp_update){
    Z_reference <- ifelse(Z_initial== max(Z_initial), reference_half_range,
                          ifelse(Z_initial==min(Z_initial), -reference_half_range,
                                 -reference_half_range +
                                   (Z_initial- min(Z_initial))*(2*reference_half_range)/(max(Z_initial)-min(Z_initial)) ) )

    # mid-point of range of Z_reference defined above is 0 by construction
    # but want it to be equal to mid-point of z - offsetz to avoid impacting the prior mean?
    Z_reference <- Z_reference + (max(Z_initial) + min(Z_initial))/2

    probit_k <- 2.0
    # k_sig_mu <- probit_k # reference_half_range*probit_k/3.0
    k_sig_mu <- reference_half_range*probit_k/3.0
  }else{
    Z_reference <- z - offsetz
  }

  # colnames(x.train) <- paste0("x", seq_len(ncol(x.train)))
  # colnames(x.test) <- colnames(x.train)
  # colnames(w.train) <- paste0("w", seq_len(ncol(w.train)))
  # colnames(w.test) <- colnames(w.train)

  if(nrow(x.test ) == 0){

    # x_train_df <- as.data.frame( x.train[ uncens_inds, , drop = FALSE ] )
    # x_test_df <- as.data.frame(x.test)
    # colnames(x_train_df) <- paste0( "x", seq_len(ncol(x_train_df)) )
    # # colnames(x_test_df) <- colnames(x_train_df)
    #
    # xdf_y <- data.frame( y = y_reference, x_train_df, check.names = FALSE )
    # # xdf_y_test <- data.frame( x_test_df, check.names = FALSE )

    xdf_y <- data.frame(y = y_reference, x = x.train[uncens_inds,]#, check.names = FALSE
                        )

    sampler_y <- dbarts(y ~ .,
                        data = xdf_y,
                        #test = x.test,
                        weights = rep(1,n1), # weightstemp_y,
                        control = control_y,
                        tree.prior = dbarts:::cgm(power = tree_power_y, base =  tree_base_y,  split.probs = rep(1 / p_y, p_y)),
                        node.prior = node.prior,
                        resid.prior = fixed(1),
                        proposal.probs = proposal.probs,
                        sigma = 1)

    xdf_z <- data.frame(y = Z_reference, x = w.train#, check.names = FALSE
                        )

    # w_train_df <- as.data.frame( w.train )
    # # w_test_df <- as.data.frame(w.test)
    # colnames(w_train_df) <- paste0( "w", seq_len(ncol(w_train_df)) )
    # # colnames(w_test_df) <- colnames(w_train_df)
    #
    # xdf_z <- data.frame( y = Z_reference, w_train_df, check.names = FALSE )
    # # xdf_z_test <- data.frame( w_test_df, check.names = FALSE )

    sampler_z <- dbarts(y ~ .,
                        data = xdf_z,
                        #test = x.test,
                        weights = rep(1,n), # weightstemp_z,
                        control = control_z,
                        tree.prior = dbarts:::cgm(power = tree_power_z, base = tree_base_z,  split.probs = rep(1 / p_z, p_z)),
                        node.prior = normal(k_sig_mu),
                        resid.prior = fixed(1),
                        proposal.probs = proposal.probs,
                        sigma = 1)
  }else{
    xdf_y <- data.frame(y = y_reference, # ystar[uncens_inds],
                        x = x.train[uncens_inds,]#, check.names = FALSE
                        )
    xdf_y_test <- data.frame(x = x.test#, check.names = FALSE
                             )
    # xdf_y_test$weights <- 1
    # x_train_df <- as.data.frame( x.train[ uncens_inds, , drop = FALSE ] )
    # x_test_df <- as.data.frame(x.test)
    # colnames(x_train_df) <- paste0( "x", seq_len(ncol(x_train_df)) )
    # colnames(x_test_df) <- colnames(x_train_df)
    #
    # xdf_y <- data.frame( y = y_reference, x_train_df, check.names = FALSE )
    # xdf_y_test <- data.frame( x_test_df, check.names = FALSE )

    sampler_y <- dbarts(y ~ .,
                        data = xdf_y,
                        test = xdf_y_test,
                        weights = rep(1,n1), # weightstemp_y,
                        control = control_y,
                        tree.prior = dbarts:::cgm(power = tree_power_y, base = tree_base_y,  split.probs = rep(1 / p_y, p_y)),
                        node.prior = node.prior,
                        resid.prior = fixed(1),
                        proposal.probs = proposal.probs,
                        sigma = 1)

    xdf_z <- data.frame(y = Z_reference, x = w.train#, check.names = FALSE
                        )
    xdf_z_test <- data.frame(x = w.test#, check.names = FALSE
                             )
    # xdf_z_test$weights <- 1
    # w_train_df <- as.data.frame( w.train )
    # w_test_df <- as.data.frame(w.test)
    # colnames(w_train_df) <- paste0( "w", seq_len(ncol(w_train_df)) )
    # colnames(w_test_df) <- colnames(w_train_df)
    #
    # xdf_z <- data.frame( y = Z_reference, w_train_df, check.names = FALSE )
    # xdf_z_test <- data.frame( w_test_df, check.names = FALSE )

    sampler_z <- dbarts(y ~ .,
                        data = xdf_z,
                        test = xdf_z_test,
                        weights = rep(1,n), # weightstemp_z,
                        control = control_z,
                        tree.prior = dbarts:::cgm(power = tree_power_z, base = tree_base_z,  split.probs = rep(1 / p_z, p_z)),
                        node.prior = normal(k_sig_mu),
                        resid.prior = fixed(1),
                        proposal.probs = proposal.probs,
                        sigma = 1 # sigmadbarts
    )
  }

  # mutemp_y <- rep(mean(ystar[uncens_inds]), length(uncens_inds))
  # mutemp_y_all <- rep(mean(ystar[uncens_inds]), length(uncens_inds))

  # mutemp_y <- rep(mean(yinit_vec), length(uncens_inds))
  # if(draw_censored_predictions){
  #   mutemp_y_all <- rep(mean(yinit_vec), n)
  # }
  mutemp_y <- rep(0, n1)
  if (draw_censored_predictions) {
    mutemp_y_all <- rep(0, n)
  }

  preds.train_ystar <- matrix(NA, n, 1)
  preds.train_z <- matrix(NA, n, 1)
  preds.test_ystar <- matrix(NA, ntest, 1)
  preds.test_z <- matrix(NA, ntest, 1)
  #initialize sum-of-tree sampler
  z_resids <- z - offsetz #z_epsilon
  z_resids[uncens_inds] <- z[uncens_inds] - offsetz - (ystar[uncens_inds]  - mutemp_y)*gamma1/(phi1 + gamma1^2)
  # z_resids[uncens_inds] <- z[uncens_inds] - offsetz - 0*gamma1/(phi1 + gamma1^2)

  if(fixed_resp_update){
    computational_offset <- Z_reference - z_resids
    # dbarts_offset <- computational_offset + offsetz_vec # genuine_offset_vec
    sampler_z$setOffset(offset = computational_offset, updateScale = FALSE)
  }else{
    sampler_z$setResponse(y = z_resids)
  }
  sampler_z$setSigma(sigma = 1)
  sampler_z$setWeights(weights = weightstemp_z)
  # sampler_z$model@node.scale <- 3

  if(sparse){
    tempmodel <- sampler_z$model
    tempmodel@tree.prior@splitProbabilities <- s_z
    sampler_z$setModel(newModel = tempmodel)
    # print("check z probs")
    # print("sampler_z@tree.prior@splitProbabilities = ")
    # print(sampler_z@tree.prior@splitProbabilities)
  }

  # sampler_z$sampleTreesFromPrior()
  # priormean_z <- sampler_z$predict(xdf_z)[1,]
  # sampler_z$sampleNodeParametersFromPrior()
  samplestemp_z <- sampler_z$run()

  # # mutemp_z <- rep(0,n) # samplestemp_z$train[,1]
  # # mutemp_test_z <- rep(0,ntest) #samplestemp_z$test[,1]
  # mutemp_z <- samplestemp_z$train[,1]
  # mutemp_test_z <- samplestemp_z$test[,1]
  # # mutemp_test_z <- sampler_z$predict(xdf_z_test)[,1]#samplestemp_z$test[,1]

  mutemp_z <- as.numeric(sampler_z$predict(xdf_z)[, 1] )
  mutemp_test_z <- as.numeric(sampler_z$predict(xdf_z_test)[, 1])


  # if(sparse){
  tempcounts <- fcount(sampler_z$getTrees()$var)
  tempcounts <- tempcounts[tempcounts$x != -1, ]
  var_count_z <- rep(0, p_z)
  var_count_z[tempcounts$x] <- tempcounts$N
  # }

  # adjust scale to reflect wider true variance of y
  # sampler_y$model@node.scale <- 0.5*4*sigest/(max(ystar[uncens_inds]) - min(ystar[uncens_inds]))

  y_resids <- ystar[uncens_inds] - gamma1_vec_train[uncens_inds]*(z[uncens_inds] - offsetz - mutemp_z[uncens_inds])
  if(fixed_resp_update){
    computational_offset_y <- y_reference - y_resids
    # dbarts_offset <- computational_offset + offsetz_vec # genuine_offset_vec
    sampler_y$setOffset(offset = computational_offset_y, updateScale = FALSE)
  }else{
    sampler_y$setResponse(y = y_resids)
  }
  # weightstemp_y <- ifelse(weightstemp_y < 1e-6, 0, weightstemp_y  )
  # sampler_y$setSigma(sigma = sqrt(phi1) )
  sampler_y$setSigma(sigma = 1)
  sampler_y$setWeights(weights = weightstemp_y)

  if(sparse){
    tempmodel <- sampler_y$model
    tempmodel@tree.prior@splitProbabilities <- s_y
    sampler_y$setModel(newModel = tempmodel)
    # print("check y probs")
    # print("sampler_y@tree.prior@splitProbabilities = ")
    # print(sampler_y@tree.prior@splitProbabilities)
  }

  # sampler_y$sampleTreesFromPrior()
  # priormean_y <- sampler_y$predict(xdf_y)[1,]
  # sampler_y$sampleNodeParametersFromPrior()

  samplestemp_y <- sampler_y$run()
  # mutemp_y <- rep(mean(y),n) #samplestemp_y$train[,1]
  # mutemp_test_y <- rep(mean(y),ntest) # samplestemp_y$test[,1]
  # mutemp_y <- samplestemp_y$train[,1]
  # mutemp_test_y <- samplestemp_y$test[,1]

  mutemp_y <- as.numeric(sampler_y$predict(xdf_y)[, 1])
  mutemp_test_y <- as.numeric( sampler_y$predict(xdf_y_test)[, 1])


  if(draw_censored_predictions){
    mutemp_y_all[uncens_inds] <- mutemp_y
    mutemp_y_all[cens_inds] <- as.numeric(sampler_y$predict(xdf_y_all[cens_inds, , drop = FALSE])[, 1])
  }

  # if(sparse){
  tempcounts <- fcount(sampler_y$getTrees()$var)
  tempcounts <- tempcounts[tempcounts$x != -1, ]
  var_count_y <- rep(0, p_y)
  var_count_y[tempcounts$x] <- tempcounts$N
  # }

  # Create transition probabilities for objects passed to HBART
  trans_prob <- c(proposal.probs["birth_death"]*proposal.probs["birth"],
                  proposal.probs["birth_death"]*(1 - proposal.probs["birth"]),
                  proposal.probs["change"])
  # normalize in case swap was set t nonzero number (swap proposal not supported for heteroscedastic trees)
  trans_prob <- trans_prob/sum(trans_prob)

  ######## Initialize Gamma Trees ##############
  # Initialize gamma sum-of-trees sampler
  # can use dbarts or R implementation
  # standard BART with weighted sum of trees
  a_gamma <- (z[uncens_inds] - offsetz - mutemp_z[uncens_inds])

  # in homoscedastic case, gamma has a normal prior with variance G_0
  # dbarts sets the prior standard deviation of gamma(v) to
  # (max(gamma_resids) + min(gamma_resids))/(2*k_dbarts )
  # setting \sqrt(G0) = (max(gamma_resids) + min(gamma_resids))/(2*k_dbarts )
  # gives the dbarts parameter value
  # k_dbarts <- (max(gamma_resids) + min(gamma_resids))/(2*sqrt(G0) )
  # k_dbarts_gamma <- (max(gamma_resids) - min(gamma_resids))/(2*sqrt(G0) )

  # mid-point of range of Z_reference defined above is 0 by construction
  # but want it to be equal to mid-point of z - offsetz to avoid impacting the prior mean?

  # However, must ensure that prior median is zero or gamma0
  # dbarts sets the prior mean to the mid-point of the input
  # gamma_reference <- gamma_resids - (max(gamma_resids) + min(gamma_resids))/2 + gamma0
  # gamma_reference <- ifelse(gamma_resids== max(gamma_resids),
  #                       sqrt(G0),
  #                       ifelse(gamma_resids==min(gamma_resids), -sqrt(G0),
  #                              0 ) )

  if (length(G0) != 1L || !is.finite(G0) || G0 <= 0) {
    stop("G0 must be a finite positive scalar.")
  }

  # gamma_reference is not a pseudo-outcome from the model. It is a fixed computational response used to prevent
  # dbarts from recalibrating the terminal-node prior when the actual gamma pseudo-response changes.
  gamma_reference <- c(sqrt(G0),-sqrt(G0), rep(0, n1-2))
  # make gamma0 the prior mean
  gamma_reference <- gamma_reference + gamma0

  k_dbarts_gamma <- 1

  # gamma_tol <- 1e-8
  gamma_tol <- sqrt(.Machine$double.eps) * max(1, max(abs(a_gamma), na.rm = TRUE))
  gamma_numerator <- ystar[uncens_inds] - mutemp_y

  raw_gamma_weight <- a_gamma^2 /phi1_vec_train[uncens_inds]
  min_gamma_weight <- 1e-6
  gamma_ok <- is.finite(a_gamma) & is.finite(gamma_numerator) & is.finite(phi1_vec_train[uncens_inds]) & phi1_vec_train[uncens_inds] > 0 &
    is.finite(raw_gamma_weight) & raw_gamma_weight >= min_gamma_weight
  # gamma_ok <- is.finite(a_gamma) & is.finite(gamma_numerator) & is.finite(phi1_vec_train[uncens_inds]) & phi1_vec_train[uncens_inds] > 0 & abs(a_gamma) > gamma_tol
  # weights_gamma <- (z[uncens_inds] - offsetz - mutemp_z[uncens_inds])^2/phi1_vec_train[uncens_inds]
  weights_gamma <- a_gamma^2/phi1_vec_train[uncens_inds]
  weights_gamma[!gamma_ok] <- 0

  if (any(!is.finite(weights_gamma)) || any(weights_gamma < 0)) {
    stop(sprintf("Invalid gamma weights during initialization."))
  }

  # colnames(v.train) <- paste0("v", seq_len(ncol(v.train)))
  # colnames(v.test) <- colnames(v.train)

  if(nrow(x.test ) == 0){
    xdf_gamma <- data.frame(y = gamma_reference,
                            x = v.train[uncens_inds, , drop = FALSE]#, check.names = FALSE
                            )
    sampler_gamma <- dbarts(y ~ .,
                        data = xdf_gamma,
                        #test = x.test,
                        weights = rep(1, n1), #weights_gamma,
                        control = control_gamma,
                        tree.prior = dbarts:::cgm(power = tree_power_gamma,
                                                  base =  tree_base_gamma,  split.probs = rep(1 / p_gamma, p_gamma)),
                        node.prior = normal(k_dbarts_gamma),
                        resid.prior = fixed(1),
                        proposal.probs = proposal.probs,
                        sigma = 1)
  }else{
    xdf_gamma <- data.frame(y = gamma_reference, # ystar[uncens_inds],
                            x = v.train[uncens_inds, , drop = FALSE]#, check.names = FALSE
                            )
    xdf_gamma_test <- data.frame(x = v.test#, check.names = FALSE
                                 )
    sampler_gamma <- dbarts(y ~ .,
                        data = xdf_gamma,
                        test = xdf_gamma_test,
                        weights = rep(1, n1), #weights_gamma,
                        control = control_gamma,
                        tree.prior = dbarts:::cgm(power = tree_power_gamma, base = tree_base_gamma,
                                                  split.probs = rep(1 / p_gamma, p_gamma)),
                        node.prior = normal(k_dbarts_gamma),
                        resid.prior = fixed(1),
                        proposal.probs = proposal.probs,
                        sigma = 1)
  }

  # computational_offset_gamma <- gamma_reference - gamma_resids
  # # dbarts_offset <- computational_offset + offsetz_vec # genuine_offset_vec
  # sampler_gamma$setOffset(offset = computational_offset_gamma, updateScale = FALSE)

  # gamma_resids <- (ystar[uncens_inds] - mutemp_y)/a_gamma
  # computational_offset_gamma <- gamma_reference - gamma_resids
  # gamma_tol <- 1e-8
  # gamma_ok <- is.finite(a_gamma) & is.finite(gamma_numerator) & is.finite(phi1_vec_train[uncens_inds]) & phi1_vec_train[uncens_inds] > 0 & abs(a_gamma) > gamma_tol
  # computational_offset_gamma[!gamma_ok] <- 0
  # sampler_gamma$setOffset(offset = computational_offset_gamma, updateScale = FALSE)

  # a_gamma <- (z[uncens_inds] - offsetz - mutemp_z[uncens_inds])
  # gamma_resids <- (ystar[uncens_inds] - mutemp_y)/a_gamma

  # gamma_resids <- (ystar[uncens_inds] - mutemp_y)/a_gamma # (z[uncens_inds] - offsetz - mutemp_z[uncens_inds])
  gamma_resids <- gamma_reference
  gamma_resids[gamma_ok] <- gamma_numerator[gamma_ok] / a_gamma[gamma_ok]
  computational_offset_gamma <- gamma_reference - gamma_resids
  computational_offset_gamma[!gamma_ok] <- 0
  sampler_gamma$setOffset(offset = computational_offset_gamma, updateScale = FALSE)
  # weights_gamma <- a_gamma^2/phi1_vec_train[uncens_inds]
  # weights_gamma[!gamma_ok] <- 0



  sampler_gamma$setSigma(sigma = 1)
  sampler_gamma$setWeights(weights = weights_gamma)
  # initialize phi sum of trees sampler
  if(sparse){
    tempmodel <- sampler_gamma$model
    tempmodel@tree.prior@splitProbabilities <- s_gamma
    sampler_gamma$setModel(newModel = tempmodel)
  }



  samplestemp_gamma <- sampler_gamma$run()

  # # if(iter %% 10 == 0){
  #   gamma_train_run <- as.numeric( samplestemp_gamma$train[, 1])
  #   gamma_train_predict <- as.numeric(sampler_gamma$predict(xdf_gamma)[, 1])
  #   gamma_compare <- c(max_abs_run = max(abs(gamma_train_run)),
  #                      max_abs_predict = max(abs(gamma_train_predict)),
  #                      max_abs_offset = max(abs(computational_offset_gamma)),
  #                      run_predict_diff = max(abs(gamma_train_run -gamma_train_predict)),
  #                      run_minus_predict_minus_offset = max(abs(gamma_train_run - gamma_train_predict - computational_offset_gamma)),
  #                      run_minus_predict_plus_offset = max(abs(gamma_train_run - gamma_train_predict +computational_offset_gamma)))
  #   print(gamma_compare)
  # # }


  # gamma1_vec_train[uncens_inds] <- samplestemp_gamma$train[,1]
  # gamma1_vec_test <- samplestemp_gamma$test[,1]

  gamma1_vec_train[uncens_inds] <-as.numeric(sampler_gamma$predict(xdf_gamma)[, 1]  )
  gamma1_vec_test <-as.numeric(sampler_gamma$predict( xdf_gamma_test)[, 1])

  if(draw_censored_predictions){
    gamma1_vec_train[cens_inds] <- as.numeric(sampler_gamma$predict(xdf_gamma_all[cens_inds, , drop = FALSE])[, 1])
  }

  # if(sparse){
  tempcounts <- fcount(sampler_gamma$getTrees()$var)
  tempcounts <- tempcounts[tempcounts$x != -1, ]
  var_count_gamma <- rep(0, p_gamma)
  var_count_gamma[tempcounts$x] <- tempcounts$N

  ######## Initialize Phi Trees ##############

  nuprime <- 10
  if( nzero - 2 > 0){
    phi_prior_mean <- S0 / (nzero - 2)
    phi_prior_cv <- 1
    A_phi <- (1 + phi_prior_cv^2)^(1 / n.trees_phi)
    nuprime <- (4 * A_phi - 2) / (A_phi - 1)
    lambdaprime <- ((S0/(nzero-2))^(1/n.trees_phi))*(nuprime - 2)/nuprime
  }else{
    print("nzero -2 <= 0, setting lambdaprime as if (S0/(nzero-2) = 1")
    lambdaprime <- ((1)^(1/n.trees_phi))*(nuprime - 2)/nuprime
  }

  implied_phi_mean <- (lambdaprime *nuprime /(nuprime - 2))^n.trees_phi
  implied_phi_cv2 <- ((nuprime - 2) /(nuprime - 4))^n.trees_phi -1

  print(c(S0 = S0,
          target_phi_mean = phi_prior_mean,
          implied_phi_mean = implied_phi_mean,
          target_phi_cv = phi_prior_cv,
          implied_phi_cv = sqrt(implied_phi_cv2),
          nuprime = nuprime,
          lambdaprime = lambdaprime))

  # lambdaprime <- nzero^(1/n.trees_phi)
  # ratio_phi <- nzero*(nzero - 2)/ S0
  # if (!is.finite(ratio_phi) || ratio_phi <= 0 || ratio_phi >= 1) {
  #   print("nuprime cannot be set by moment matching with homoscedastic phi prior.\\")
  #   print("Setting nuprime as if S0 = 2*nzero*(nzero - 2) .\\")
  #   nuprime <- 2/(1- (1/2)^(1/n.trees_phi)) # this scales approximately linearly with n.trees_phi
  # }else{
  #   nuprime <- 2/(1- (nzero*(nzero-2)/S0)^(1/n.trees_phi))
  # }

  # v_phi_train <- v.train[ uncens_inds, , drop = FALSE]
  tree_fits_store_logphi <- matrix( NA_real_, nrow = n1, ncol = n.trees_phi )


  phi_outcome_resids <- ystar[uncens_inds] - mutemp_y - gamma1_vec_train[uncens_inds]*(z[uncens_inds] - offsetz - mutemp_z[uncens_inds])
  # Create a list of trees for the initial stump
  # all logs of leaf parameters initialized to zero,
  # sum is zero, therefore initial variances are constant and equal to 1.
  curr_trees_phi = create_stump(num_trees = n.trees_phi,
                              y = as.vector(phi_outcome_resids),
                              X = v.train[uncens_inds, , drop = FALSE])
  # Initialise the new trees as current one
  new_trees_phi = curr_trees_phi

  for (j in seq_len(n.trees_phi)) {
    tree_prediction_j <- get_predictions(curr_trees_phi[[j]],
                                                   v.train[ uncens_inds, , drop = FALSE],
                                                   single_tree = TRUE)
    tree_prediction_j <- as.numeric(tree_prediction_j)
    if (length(tree_prediction_j) != n1 || any(!is.finite(tree_prediction_j))) {
      stop(sprintf("Invalid initial prediction for phi tree %d.",j))
    }
    tree_fits_store_logphi[, j] <- tree_prediction_j
  }
  phi_log_s2prod <- rowSums(tree_fits_store_logphi)
  phi_log_check <- get_predictions(curr_trees_phi, v.train[ uncens_inds, , drop = FALSE], single_tree =n.trees_phi == 1L)
  cat("length(phi_log_s2prod) =",length(phi_log_s2prod),"\n")
  cat("length(phi_log_check) =",length(phi_log_check),"\n")
  cat("class(phi_log_s2prod) =",paste(class(phi_log_s2prod), collapse = ", "),"\n")
  cat("class(phi_log_check) =",paste(class(phi_log_check), collapse = ", "),"\n")
  phi_log_s2prod_num <- as.numeric(phi_log_s2prod)
  phi_log_check_num <- as.numeric(phi_log_check)
  phi_log_diff <- phi_log_s2prod_num - phi_log_check_num
  cat("max absolute phi prediction difference =",max(abs(phi_log_diff)),"\n")
  cat("mean difference =",mean(phi_log_diff),"\n")
  cat("range of difference =",paste(range(phi_log_diff), collapse = ", "),"\n")
  cat("all.equal ignoring attributes:\n")

  str(phi_log_s2prod)
  str(phi_log_check)

  attributes(phi_log_s2prod)
  attributes(phi_log_check)
  print(all.equal(phi_log_s2prod,phi_log_check,tolerance = 1e-12))

  print(all.equal(phi_log_s2prod_num,phi_log_check_num,tolerance = 1e-12,check.attributes = FALSE))
  stopifnot(isTRUE(all.equal(as.numeric(phi_log_s2prod),as.numeric(phi_log_check),tolerance = 1e-10,check.attributes = FALSE)))
  print("Line 1110")
  # stopifnot(isTRUE(all.equal(phi_log_s2prod,phi_log_check,tolerance = 1e-12)))

  phi_log_s2prod_num <- as.numeric(phi_log_s2prod)
  phi_log_check_num <- as.numeric(phi_log_check)
  if (length(phi_log_s2prod_num) != length(phi_log_check_num)) {
    stop( sprintf(paste0("Phi prediction lengths differ: %d versus %d."),length(phi_log_s2prod_num),length(phi_log_check_num)))
  }
  phi_log_error <- max(abs(phi_log_s2prod_num -phi_log_check_num))
  if (!is.finite(phi_log_error) ||phi_log_error > 1e-10) {
    stop(sprintf("Phi prediction mismatch: %.3e.",phi_log_error))
  }

  # Initialise the predicted values to zero
  phi_log_s2prod<- get_predictions(curr_trees_phi, v.train[uncens_inds,, drop = FALSE], single_tree = n.trees_phi == 1)
  phi1_vec_train[uncens_inds] <- exp(phi_log_s2prod)
  if(draw_censored_predictions){
    phi1_vec_train[cens_inds] = exp(get_predictions(curr_trees_phi, v.train[cens_inds, , drop = FALSE], single_tree = n.trees_phi == 1))
  }

  var_count_phi <- rep(0, p_phi)
  binmat_list_phi <- list()
  for (j in 1:n.trees_phi) {
    binmat_list_phi[[j]] <- rep(1, n1)
  }

  # f_y, f_z, and phi initialized with sampling of trees, so it
  # is logically consistent to also sample phi trees here
  # before the main Gibbs loop
  for (j in 1:n.trees_phi) {
    current_partial_logproduct = phi_log_s2prod - tree_fits_store_logphi[,j]
    Esqvec = phi_outcome_resids^2/exp(current_partial_logproduct)
    # We need the new and old trees for the likelihoods
    new_trees_phi <- curr_trees_phi
    type_phi = sample_move(curr_trees_phi[[j]],
                           0, #iter,
                           0, #n_burn,
                           trans_prob)

    # Generate a new tree based on the current
    new_trees_phi[[j]] <- update_tree(
      y = Esqvec,
      X = v.train[uncens_inds, , drop = FALSE],
      type = type_phi,
      curr_tree = curr_trees_phi[[j]],
      node_min_size = node_min_size,
      s = s_phi,
      max_bad_trees = max_bad_trees,
      splitting_rules = splitting_rules
    )

    # (c) Obtain the Metropolis-Hastings probability
    curr_tree_phi <- curr_trees_phi[[j]]
    new_tree_phi <- new_trees_phi[[j]]
    if((nrow(new_tree_phi$tree_matrix) == nrow(curr_tree_phi$tree_matrix) ) & (type_phi != "change" )){
      alpha_MH <- 0
      # print("no good trees")
    }else{
      # if(j==1){
      # CURRENT TREE: compute the log of the marginalised likelihood + log of the tree prior
      l_old_phi = tree_full_conditional_HBART(curr_trees_phi[[j]], Esqvec, nuprime, lambdaprime) +
        get_tree_prior(curr_trees_phi[[j]], tree_base_phi, tree_power_phi)
      # }
      # NEW TREE: compute the log of the marginalised likelihood + log of the tree prior
      l_new_phi = tree_full_conditional_HBART(new_trees_phi[[j]], Esqvec, nuprime, lambdaprime) +
        get_tree_prior(new_trees_phi[[j]], tree_base_phi, tree_power_phi)
      alpha_MH = alpha_mh(l_new_phi,l_old_phi, curr_trees_phi[[j]],new_trees_phi[[j]], type_phi)
    }

    if(is.na(alpha_MH)){ stop("line 3075. alpha_MH is NA") }

    if(alpha_MH > runif(1)) {
      curr_trees_phi[[j]] = new_trees_phi[[j]]
      l_old_phi <- l_new_phi
      # if(sparse){
      if (type_phi == "grow") {
        var_count_phi[curr_trees_phi[[j]]$var] <- var_count_phi[curr_trees_phi[[j]]$var] + 1
      } else if (type_phi == "prune") {
        var_count_phi[curr_trees_phi[[j]]$var] <- var_count_phi[curr_trees_phi[[j]]$var] - 1
      } else {
        # if(curr_trees_phi[[j]]$var[1]!=0){ # What if change step returned $var equal to c(0,0) ????
        if (length(curr_trees_phi[[j]]$var) == 2L && all(is.finite(curr_trees_phi[[j]]$var)) && all(curr_trees_phi[[j]]$var > 0L)) {
          var_count_phi[curr_trees_phi[[j]]$var[1]] <- var_count_phi[curr_trees_phi[[j]]$var[1]] - 1
          var_count_phi[curr_trees_phi[[j]]$var[2]] <- var_count_phi[curr_trees_phi[[j]]$var[2]] + 1
        }
      }
      # }
    }
    curr_trees_phi[[j]] = simulate_S_HBART(curr_trees_phi[[j]],
                                           Esqvec,
                                           nuprime,
                                           lambdaprime)
    # Updating BART predictions
    current_fit = get_predictions(curr_trees_phi[[j]], v.train[uncens_inds, , drop = FALSE], single_tree = TRUE)
    phi_log_s2prod = phi_log_s2prod - tree_fits_store_logphi[,j] # subtract the old fit
    phi_log_s2prod = phi_log_s2prod + current_fit # add the new fit
    tree_fits_store_logphi[,j] = current_fit # update the new fit
  } # end loop over phi trees

  #update z_epsilon
  phi_log_s2prod_test <- get_predictions(curr_trees_phi,
                                         v.test,
                                         single_tree = length(curr_trees_phi) == 1)

  if (any(!is.finite(phi_log_s2prod))) {stop(sprintf("Nonfinite log phi at initialization."))}
  if (any(!is.finite(phi_log_s2prod_test))) {stop(sprintf("Nonfinite log phi test at initialization."))}

  phi1_vec_train[uncens_inds] <- exp(phi_log_s2prod)
  if(draw_censored_predictions){
    phi1_vec_train[cens_inds] = exp(get_predictions(curr_trees_phi, v.train[cens_inds, , drop = FALSE], single_tree = n.trees_phi == 1))
  }
  phi1_vec_test <- exp(phi_log_s2prod_test)

  if (any(!is.finite(phi1_vec_train[uncens_inds])) ||any(phi1_vec_train[uncens_inds] <= 0)) { stop(sprintf("Invalid phi values at initialization."))}
  if(draw_censored_predictions){
    if (any(!is.finite(phi1_vec_train[cens_inds])) ||any(phi1_vec_train[cens_inds] <= 0)) { stop(sprintf("Invalid phi values at initialization."))}
  }
  if (any(!is.finite(phi1_vec_test)) ||any(phi1_vec_test <= 0)) { stop(sprintf("Invalid phi test values at initialization."))}

  y_epsilon <- rep(0, n)
  y_epsilon[uncens_inds] <- ystar[uncens_inds] - mutemp_y
  z_epsilon <- z - offsetz - mutemp_z

  #########  Begin Gibbs sampler ######################################################

  # pb <- progress_bar$new(total = n.iter+n.burnin)
  # pb <- progress_bar$new(
  #   format = " [:bar] :percent eta: :eta",
  #   total = n.iter+n.burnin, clear = FALSE, width= 60)

  #loop through the Gibbs sampler iterations
  for(iter in 1:(n.iter+n.burnin)){

    phi_uncens <- phi1_vec_train[uncens_inds]
    if (any(!is.finite(phi_uncens)) || any(phi_uncens <= 0)) {
      stop( sprintf( paste0( "Nonpositive or nonfinite phi before mean-tree updates at iteration %d. Range: [%g, %g]."),
          iter, min(phi_uncens), max(phi_uncens) ))
    }

    temp_sd_z <- sqrt( phi1_vec_train/(phi1_vec_train+gamma1_vec_train^2)   )
    ######### #draw the latent outcome z ##########################
    # z[cens_inds] <- rtruncnorm(n0, a= below_cens, b = above_cens, mean = mu[cens_inds], sd = sigma)
    if(length(cens_inds)>0){
      # temp_sd_y <- sqrt(phi1 + gamma1^2)
      # ystar[cens_inds] <- rnorm(n0,  mean = mutemp_y[cens_inds], sd = temp_sd_y)
      # temp_zmean_cens <- offsetz + mutemp_z[cens_inds] + (ystar[cens_inds]  - mutemp_y[cens_inds])*gamma1/(phi1 + gamma1^2)
      temp_zmean_cens <- offsetz + mutemp_z[cens_inds] #+ (ystar[cens_inds]  - mutemp_y[cens_inds])*gamma1/(phi1 + gamma1^2)
      z[cens_inds] <- rtruncnorm(n0, a= -Inf, b = 0, mean = temp_zmean_cens, sd = 1)
    }
    # # temp_zmean_uncens <- offsetz + mutemp_z[uncens_inds] + (ystar[uncens_inds]  - mutemp_y[uncens_inds])*gamma1/(phi1 + gamma1^2)
    # temp_zmean_uncens <- offsetz + mutemp_z[uncens_inds] + (ystar[uncens_inds]  - mutemp_y)*gamma1/(phi1 + gamma1^2)
    # z[uncens_inds] <- rtruncnorm(n1, a= 0, b = Inf, mean = temp_zmean_uncens, sd = temp_sd_z)

    temp_zmean_uncens <- offsetz + mutemp_z[uncens_inds] +
      (ystar[uncens_inds]  - mutemp_y )*gamma1_vec_train[uncens_inds]/(phi1_vec_train[uncens_inds] + gamma1_vec_train[uncens_inds]^2)
    z[uncens_inds] <- rtruncnorm(n1, a= 0, b = Inf, mean = temp_zmean_uncens, sd = temp_sd_z[uncens_inds])

    # z_epsilon <- z - offsetz - mutemp_z
    # y_epsilon <- ystar - mutemp_y
    z_epsilon <- z - offsetz - mutemp_z
    y_epsilon <- rep(0, n)
    y_epsilon[uncens_inds] <- ystar[uncens_inds] - mutemp_y

    ####### draw sums of trees for z #######################################################
    #create residuals for z and set variance

    z_resids <- z - offsetz #z_epsilon
    z_resids[uncens_inds] <- z[uncens_inds] - offsetz -
      (ystar[uncens_inds] - mutemp_y)*gamma1_vec_train[uncens_inds]/(phi1_vec_train[uncens_inds] + gamma1_vec_train[uncens_inds]^2)

    if(fixed_resp_update){
      computational_offset <- Z_reference - z_resids
      # dbarts_offset <- computational_offset + offsetz_vec # genuine_offset_vec
      sampler_z$setOffset(offset = computational_offset, updateScale = FALSE)
    }else{
      sampler_z$setResponse(y = z_resids)
    }

    #set the response for draws of z trees
    # sampler_z$setResponse(y = z_resids)
    #set the standard deivation
    sampler_z$setSigma(sigma = 1)

    # weightstemp_z[uncens_inds] <- (gamma1^2 + phi1)/phi1
    # weightstemp_z[uncens_inds] <- (gamma1_vec_train^2 + phi1_vec_train)/phi1_vec_train
    # weightstemp_z[uncens_inds] <- (gamma1_vec_train[uncens_inds]^2 + phi1_vec_train[uncens_inds])/phi1_vec_train[uncens_inds]

    # weightstemp_z[uncens_inds] <- exp(log(gamma1_vec_train[uncens_inds]^2 + phi1_vec_train[uncens_inds]) - phi_log_s2prod)
    # weightstemp_z[uncens_inds] <- exp(2*log(abs(gamma1_vec_train[uncens_inds]))  - phi_log_s2prod) + 1

    # weightstemp_z[uncens_inds] <- ifelse((gamma1_vec_train[uncens_inds]^2 + phi1_vec_train[uncens_inds]) < 1e-6, 0, weightstemp_z[uncens_inds]  )
    gamma_uncens <- gamma1_vec_train[uncens_inds]
    logphi_uncens <- phi_log_s2prod

    if (any(!is.finite(gamma_uncens))) {stop(sprintf("Nonfinite gamma values before selection update at iteration %d.", iter))}
    if (any(!is.finite(logphi_uncens))) {stop(sprintf("Nonfinite log phi values before selection update at iteration %d.", iter))}

    log_gamma_sq <- rep(-Inf, n1)
    nonzero_gamma <- gamma_uncens != 0
    log_gamma_sq[nonzero_gamma] <- 2 * log(abs(gamma_uncens[nonzero_gamma]))
    log_gamma2_over_phi <- log_gamma_sq - logphi_uncens

    log_weights_z_uncens <- ifelse( log_gamma2_over_phi > 0,
                                    log_gamma2_over_phi + log1p(exp(-log_gamma2_over_phi)),
                                    log1p(exp(log_gamma2_over_phi)) )
    if (any(!is.finite(log_weights_z_uncens))) { stop(sprintf("Nonfinite log selection weights at iteration %d.", iter)) }
    weightstemp_z[uncens_inds] <- exp(log_weights_z_uncens)


    if (any(!is.finite(weightstemp_z)) || any(weightstemp_z <= 0)) {

      phi_diag <- c(
        min_logphi = min(phi_log_s2prod),
        q001_logphi = quantile(phi_log_s2prod, 0.001),
        q01_logphi = quantile(phi_log_s2prod, 0.01),
        median_logphi = median(phi_log_s2prod),
        max_logphi = max(phi_log_s2prod),
        min_phi = min(phi1_vec_train[uncens_inds]),
        max_phi = max(phi1_vec_train[uncens_inds])
      )

      print(phi_diag)
      print("min(gamma1_vec_train[uncens_inds]^2 ) =  ")
      print(min(gamma1_vec_train[uncens_inds]^2))
      print("max(gamma1_vec_train[uncens_inds]^2 ) =  ")
      print(max(gamma1_vec_train[uncens_inds]^2))
      stop( sprintf("Invalid selection weights at iteration %d.",iter))
    }
    sampler_z$setWeights(weights = weightstemp_z)


    # raw_weights_z <- rep(1, n)
    # raw_weights_z[uncens_inds] <- 1 + gamma1_vec_train[uncens_inds]^2 / phi_uncens
    # weight_scale_z <- max(raw_weights_z)
    # weightstemp_z <- raw_weights_z / weight_scale_z
    # sigma_z_working <- 1 / sqrt(weight_scale_z)
    # sampler_z$setSigma( sigma = sigma_z_working )
    # sampler_z$setWeights(weights = weightstemp_z)



    if(sparse){
      tempmodel <- sampler_z$model
      tempmodel@tree.prior@splitProbabilities <- s_z
      sampler_z$setModel(newModel = tempmodel)
      # print("check z probs")
      # print("sampler_z@tree.prior@splitProbabilities = ")
      # print(sampler_z@tree.prior@splitProbabilities)
    }

    samplestemp_z <- sampler_z$run()

    # mutemp_z <- samplestemp_z$train[,1]
    # # mutemp_z <- sampler_z$predict(xdf_z)[,1]
    # mutemp_test_z <- samplestemp_z$test[,1]
    # # mutemp_test_z <- sampler_z$test[,1]#samplestemp_z$test[,1]
    # # mutemp_test_z <- sampler_z$predict(xdf_z_test)[,1]#samplestemp_z$test[,1]

    mutemp_z <- as.numeric(sampler_z$predict(xdf_z)[, 1] )
    mutemp_test_z <- as.numeric(sampler_z$predict(xdf_z_test)[, 1])

    # if(sparse){
    tempcounts <- fcount(sampler_z$getTrees()$var)
    tempcounts <- tempcounts[tempcounts$x != -1, ]
    var_count_z <- rep(0, p_z)
    var_count_z[tempcounts$x] <- tempcounts$N
    # }

    #update z_epsilon
    z_epsilon <- z - offsetz - mutemp_z

    ####### draw sums of trees for y #######################################################

    #create residuals for z and set variance
    # y_resids <- ystar[uncens_inds] - gamma1*(z[uncens_inds] - offsetz - mutemp_z[uncens_inds])
    y_resids <- ystar[uncens_inds] - gamma1_vec_train[uncens_inds]*(z[uncens_inds] - offsetz - mutemp_z[uncens_inds])
    # sd_ydraw <- sqrt(phi1)

    #set the response for draws of z trees
    if(fixed_resp_update){
      computational_offset_y <- y_reference - y_resids
      # dbarts_offset <- computational_offset + offsetz_vec # genuine_offset_vec
      sampler_y$setOffset(offset = computational_offset_y, updateScale = FALSE)
    }else{
      sampler_y$setResponse(y = y_resids)
    }

    weightstemp_y  <- 1/phi1_vec_train[uncens_inds]

    logphi_uncens <- phi_log_s2prod
    log_weight_y <- -logphi_uncens
    max_log_double <- log(.Machine$double.xmax)
    if (any(log_weight_y >= max_log_double)) {
      stop(sprintf(paste0("Outcome precision overflow at iteration %d; minimum log phi is %g."), iter, min(logphi_uncens)))
    }
    weightstemp_y <- exp(log_weight_y)
    # weightstemp_y <- ifelse(weightstemp_y < 1e-6, 0, weightstemp_y  )
    #
    # if (any(!is.finite(weightstemp_y)) || any(weightstemp_y <= 0)) {
    #   stop( sprintf("Invalid outcome weights at iteration %d.",iter))
    # }

    #set the standard deviation
    # sampler_y$setSigma(sigma = sd_ydraw)
    sampler_y$setSigma(sigma = 1)
    sampler_y$setWeights(weights = weightstemp_y)



    # raw_weights_y <- exp(-phi_log_s2prod)
    # weight_scale_y <- max(raw_weights_y)
    # weightstemp_y <- raw_weights_y / weight_scale_y
    # sigma_y_working <- 1 / sqrt(weight_scale_y)
    # sampler_y$setSigma(sigma = sigma_y_working)
    # sampler_y$setWeights(weights = weightstemp_y)









    if(sparse){
      tempmodel <- sampler_y$model
      tempmodel@tree.prior@splitProbabilities <- s_y
      sampler_y$setModel(newModel = tempmodel)
      # print("check y probs")
      # print("sampler_y@tree.prior@splitProbabilities = ")
      # print(sampler_y@tree.prior@splitProbabilities)
    }

    samplestemp_y <- sampler_y$run()
    # mutemp_y <- samplestemp_y$train[,1]
    # # mutemp_y <- sampler_y$predict(xdf_y)[,1]
    # mutemp_test_y <- samplestemp_y$test[,1]

    mutemp_y <- as.numeric(sampler_y$predict(xdf_y)[, 1])
    mutemp_test_y <- as.numeric( sampler_y$predict(xdf_y_test)[, 1])


    # if(sparse){
    tempcounts <- fcount(sampler_y$getTrees()$var)
    tempcounts <- tempcounts[tempcounts$x != -1, ]
    var_count_y <- rep(0, p_y)
    var_count_y[tempcounts$x] <- tempcounts$N
    # }

    #update z_epsilon
    y_epsilon[uncens_inds] <- ystar[uncens_inds] - mutemp_y
    if(draw_censored_predictions){
      mutemp_y_all[uncens_inds] <- mutemp_y
      mutemp_y_all[cens_inds] <- as.numeric(sampler_y$predict(xdf_y_all[cens_inds, , drop = FALSE])[, 1])
    }
    ############# Covariance matrix samples ##########################

    ############# Gamma tree samples ##########################
    a_gamma <- (z[uncens_inds] - offsetz - mutemp_z[uncens_inds])
    # gamma_resids <- (ystar[uncens_inds] - mutemp_y)/a_gamma
    # computational_offset_gamma <- gamma_reference - gamma_resids

    # gamma_tol <- 1e-8
    gamma_tol <- sqrt(.Machine$double.eps) * max(1, max(abs(a_gamma), na.rm = TRUE))
    gamma_numerator <- ystar[uncens_inds] - mutemp_y
    raw_gamma_weight <- a_gamma^2 /phi1_vec_train[uncens_inds]
    min_gamma_weight <- 0 # 1e-6
    gamma_ok <- is.finite(a_gamma) & is.finite(gamma_numerator) & is.finite(phi1_vec_train[uncens_inds]) & phi1_vec_train[uncens_inds] > 0 &
      is.finite(raw_gamma_weight) & raw_gamma_weight >= min_gamma_weight
    # gamma_ok <- is.finite(a_gamma) & is.finite(gamma_numerator) & is.finite(phi1_vec_train[uncens_inds]) & phi1_vec_train[uncens_inds] > 0 & abs(a_gamma) > gamma_tol
    if (!any(gamma_ok)) {warning(paste0("No observations provide usable information for the gamma forest at iteration ",iter,"; this update is from the prior."))}

    gamma_resids <- gamma_reference
    gamma_resids[gamma_ok] <- gamma_numerator[gamma_ok] / a_gamma[gamma_ok]
    computational_offset_gamma <- gamma_reference - gamma_resids
    computational_offset_gamma[!gamma_ok] <- 0


    sampler_gamma$setOffset(offset = computational_offset_gamma, updateScale = FALSE)
    # weights_gamma <- a_gamma^2/phi1_vec_train[uncens_inds]
    # weights_gamma[!gamma_ok] <- 0
    weights_gamma <- numeric(n1)
    weights_gamma[gamma_ok] <- a_gamma[gamma_ok]^2 / phi1_vec_train[uncens_inds][gamma_ok]
    if (any(!is.finite(weights_gamma)) || any(weights_gamma < 0)) {
      stop(sprintf("Invalid gamma weights at iteration %d.",iter))
    }


    positive_gamma <- weights_gamma > 0

    # if (any(positive_gamma)) {
    #   weight_scale_gamma <- max(weights_gamma[positive_gamma])
    #   weights_gamma[positive_gamma] <- weights_gamma[positive_gamma] / weight_scale_gamma
    #   sigma_gamma_working <- 1 / sqrt(weight_scale_gamma)
    # } else {
    #   sigma_gamma_working <- 1
    # }


    #set the standard deviation
    sampler_gamma$setSigma(sigma = 1)
    sampler_gamma$setWeights(weights = weights_gamma)

    positive_gamma <- weights_gamma > 0
    if (any(positive_gamma)) {
      gamma_weight_scale <- max(weights_gamma[positive_gamma])
      weights_gamma_scaled <- weights_gamma / gamma_weight_scale
      sigma_gamma_scaled <- 1 /sqrt(gamma_weight_scale)
    } else {
      weights_gamma_scaled <- weights_gamma
      sigma_gamma_scaled <- 1
    }
    # sampler_gamma$setSigma(sigma = sigma_gamma_scaled)
    # sampler_gamma$setWeights(weights = weights_gamma_scaled)

    if(sparse){
      tempmodel <- sampler_gamma$model
      tempmodel@tree.prior@splitProbabilities <- s_gamma
      sampler_gamma$setModel(newModel = tempmodel)
    }


    if (any(!is.finite(gamma_resids))) {
      stop(sprintf("Nonfinite gamma pseudo-response at iteration %d.", iter))
    }

    if (any(!is.finite(computational_offset_gamma))) {
      stop(sprintf("Nonfinite gamma computational offset at iteration %d.", iter))
    }

    if (any(!is.finite(weights_gamma)) || any(weights_gamma < 0)) {
      stop(sprintf("Invalid raw gamma weights at iteration %d.", iter))
    }

    if (any(!is.finite(weights_gamma_scaled)) ||any(weights_gamma_scaled < 0) ||!is.finite(sigma_gamma_scaled) ||sigma_gamma_scaled <= 0) {
      stop(sprintf("Invalid scaled gamma likelihood at iteration %d.", iter))
    }

    positive_gamma <- weights_gamma > 0
    variance_raw <- 1 / weights_gamma[positive_gamma]
    variance_scaled <- sigma_gamma_scaled^2 /weights_gamma_scaled[positive_gamma]
    if (!isTRUE(all.equal(variance_raw,variance_scaled,tolerance = 1e-10,check.attributes = FALSE))) {stop(sprintf("Gamma weight rescaling failed at iteration %d.", iter))}


    # if(iter > 110L){
      # gamma_pre_run_diag <- list(
      #   iteration = iter,
      #   gamma_ok = gamma_ok,
      #   gamma_resids = gamma_resids,
      #   computational_offset_gamma = computational_offset_gamma,
      #   weights_gamma = weights_gamma,
      #   weights_gamma_scaled = weights_gamma_scaled,
      #   sigma_gamma_scaled = sigma_gamma_scaled,
      #   a_gamma = a_gamma,
      #   gamma_numerator = gamma_numerator,
      #   phi_uncens = phi1_vec_train[uncens_inds],
      #   gamma_previous = gamma1_vec_train[uncens_inds]
      # )
      # print("gamma_pre_run_diag = ")
      # print(gamma_pre_run_diag)
      # saveRDS(
      #   gamma_pre_run_diag,
      #   file = sprintf(
      #     "gamma_pre_run_iter_%05d.rds",
      #     iter
      #   )
      # )

      positive_gamma <- weights_gamma > 0

      # gamma_summary <- c(
      #   n_positive = sum(positive_gamma),
      #   n_zero = sum(!positive_gamma),
      #   sum_weights = sum(weights_gamma),
      #   min_positive_weight = min(
      #     weights_gamma[positive_gamma]
      #   ),
      #   max_weight = max(weights_gamma),
      #   max_abs_response = max(abs(gamma_resids)),
      #   max_abs_offset = max(
      #     abs(computational_offset_gamma)
      #   ),
      #   sigma_scaled = sigma_gamma_scaled,
      #   max_scaled_weight = max(
      #     weights_gamma_scaled
      #   ),
      #   min_positive_scaled_weight = min(
      #     weights_gamma_scaled[
      #       weights_gamma_scaled > 0
      #     ]
      #   )
      # )
      # print("gamma_summary = ")
      # print(gamma_summary)

    # }

    trees_before <- sampler_gamma$getTrees()
    samplestemp_gamma <- sampler_gamma$run()
    trees_after <- sampler_gamma$getTrees()

    mu_before <- trees_before$mu
    mu_after <- trees_after$mu




    # if(iter %% 10 == 0){
    #   gamma_train_run <- as.numeric( samplestemp_gamma$train[, 1])
    #   gamma_train_predict <- as.numeric(sampler_gamma$predict(xdf_gamma)[, 1])
    #   gamma_compare <- c(max_abs_run = max(abs(gamma_train_run)),
    #                      max_abs_predict = max(abs(gamma_train_predict)),
    #                      max_abs_offset = max(abs(computational_offset_gamma)),
    #                      run_predict_diff = max(abs(gamma_train_run -gamma_train_predict)),
    #                      run_minus_predict_minus_offset = max(abs(gamma_train_run - gamma_train_predict - computational_offset_gamma)),
    #                      run_minus_predict_plus_offset = max(abs(gamma_train_run - gamma_train_predict +computational_offset_gamma)))
    #   print(gamma_compare)
    # }
    #
    #
    # gamma_train_new <- samplestemp_gamma$train[, 1]
    # gamma_test_new <- samplestemp_gamma$test[, 1]
    # if (any(!is.finite(gamma_train_new)) || any(!is.finite(gamma_test_new))) {
    #   gamma_diag <- c(n_ok = sum(gamma_ok),n_zero = sum(weights_gamma == 0),min_positive_weight = if (any(weights_gamma > 0)) {min(weights_gamma[weights_gamma > 0])
    #     } else {NA_real_},
    #     max_weight = max(weights_gamma),
    #     min_gamma_resid = min(gamma_resids),
    #     max_gamma_resid = max(gamma_resids),
    #     max_abs_gamma_resid = max(abs(gamma_resids)),
    #     min_gamma_offset = min(computational_offset_gamma),
    #     max_gamma_offset = max(computational_offset_gamma),
    #     max_abs_gamma_offset = max(abs(computational_offset_gamma)),
    #     min_abs_a_ok = if (any(gamma_ok)) {min(abs(a_gamma[gamma_ok]))} else {NA_real_},
    #     max_abs_numerator = max(abs(gamma_numerator))
    #   )
    #   print(gamma_diag)
    #
    #   stop( sprintf( paste0("Gamma sampler returned nonfinite predictions at iteration %d. phi range: [%g, %g], positive gamma-weight range: [%g, %g]."),
    #       iter, min(phi1_vec_train[uncens_inds]), max(phi1_vec_train[uncens_inds]), min( weights_gamma[ weights_gamma > 0],na.rm = TRUE),max(weights_gamma, na.rm = TRUE)))
    # }

    # gamma1_vec_train[uncens_inds] <- gamma_train_new
    # gamma1_vec_test <- gamma_test_new
    #
    #
    # gamma1_vec_train[uncens_inds] <- samplestemp_gamma$train[,1]
    # gamma1_vec_test <- samplestemp_gamma$test[,1]

    # gamma_trees <- sampler_gamma$getTrees()
    # bad_gamma_mu <- which(!is.finite(gamma_trees$mu) &!is.na(gamma_trees$mu))
    # if (length(bad_gamma_mu) > 0L) {stop(sprintf("Gamma forest contains %d nonfinite terminal-node parameters at iteration %d.",length(bad_gamma_mu),iter))}
    gamma_train_new <- as.numeric(sampler_gamma$predict(xdf_gamma)[, 1])
    gamma_test_new <- as.numeric(sampler_gamma$predict(xdf_gamma_test)[, 1])


    # gamma_train_new <- as.numeric(sampler_gamma$predict(xdf_gamma)[, 1])
    # gamma_test_new <- as.numeric(sampler_gamma$predict(xdf_gamma_test)[, 1])

    # if (any(!is.finite(gamma_train_new)) ||any(!is.finite(gamma_test_new))) {
    #   positive_weights <- weights_gamma[weights_gamma > 0]
    #   diagnostic <- list(iteration = iter,bad_gamma_train = which(!is.finite(gamma_train_new)),bad_gamma_test = which(!is.finite(gamma_test_new)),
    #     gamma_train = gamma_train_new,gamma_test = gamma_test_new,gamma_resids = gamma_resids,
    #     computational_offset_gamma = computational_offset_gamma,weights_gamma = weights_gamma,weights_gamma_scaled = weights_gamma_scaled,sigma_gamma_scaled = sigma_gamma_scaled,
    #     a_gamma = a_gamma,gamma_numerator = gamma_numerator,phi_uncens = phi1_vec_train[uncens_inds],gamma_ok = gamma_ok
    #   )
    #
    #   diagnostic_file <- file.path(diagnostic_path,sprintf("htbart2_gamma_failure_rep_%03d_iter_%05d.rds",replication_id,iter))
    #   saveRDS(diagnostic, diagnostic_file)
    #
    #   stop(sprintf(paste0("Nonfinite offset-free gamma prediction at iteration %d. ","n_ok=%d, n_zero=%d, raw positive weight range=[%g,%g], ","scaled sigma=%g, max_abs_offset=%g. Diagnostic: %s"),
    #     iter,
    #     sum(gamma_ok),
    #     sum(weights_gamma == 0),
    #     if (length(positive_weights)) min(positive_weights) else NA_real_,
    #     if (length(positive_weights)) max(positive_weights) else NA_real_,
    #     sigma_gamma_scaled,
    #     max(abs(computational_offset_gamma)),
    #     diagnostic_file
    #   ))
    # }
    if (length(gamma_train_new) != n1 || length(gamma_test_new) != ntest || any(!is.finite(gamma_train_new)) || any(!is.finite(gamma_test_new))) {
      bad_train <- which(!is.finite(gamma_train_new))
      bad_test <- which(!is.finite(gamma_test_new))

      # print("gamma_summary = ")
      # print(gamma_summary)

      # print("gamma_pre_run_diag = ")
      # print(gamma_pre_run_diag)

      compact_gamma_diag <- list(
        iteration = iter,
        n_ok = sum(gamma_ok),
        n_not_ok = sum(!gamma_ok),
        failed_indices = which(!gamma_ok),
        a_failed = a_gamma[!gamma_ok],
        numerator_failed = gamma_numerator[!gamma_ok],
        raw_weight_failed = raw_gamma_weight[!gamma_ok],
        gamma_previous_failed = gamma_previous[!gamma_ok],
        raw_weight_quantiles = quantile(
          raw_gamma_weight[is.finite(raw_gamma_weight)],
          c(0, 0.001, 0.01, 0.5, 0.99, 0.999, 1)
        ),
        working_response_quantiles = quantile(
          gamma_working_response,
          c(0, 0.001, 0.01, 0.5, 0.99, 0.999, 1)
        ),
        max_abs_offset = max(
          abs(computational_offset_gamma)
        ),
        phi_range = range(phi_uncens)
      )

      print(compact_gamma_diag)


      print("mu Na outputs = ")
      print(c(
        before_nan = sum(is.nan(mu_before)),
        after_nan = sum(is.nan(mu_after)),
        before_inf = sum(is.infinite(mu_before)),
        after_inf = sum(is.infinite(mu_after)),
        before_finite = sum(is.finite(mu_before)),
        after_finite = sum(is.finite(mu_after))
      ))

      new_nan <- which(!is.nan(mu_before) & is.nan(mu_after))
      new_inf <- which(!is.infinite(mu_before) & is.infinite(mu_after))

      print("new_nan = ")
      print(new_nan)
      print("new_inf = ")
      print(new_inf)

      print("trees_after[new_nan, , drop = FALSE] = ")
      print(trees_after[new_nan, , drop = FALSE])
      print("trees_after[new_inf, , drop = FALSE] = ")
      print(trees_after[new_inf, , drop = FALSE])

      stop(sprintf( paste0( "Offset-free gamma prediction failed at iteration %d. ", "Nonfinite train predictions: %d; test predictions: %d."),
        iter, length(bad_train),length(bad_test)))
    }

    gamma1_vec_train[uncens_inds] <- gamma_train_new
    gamma1_vec_test <- gamma_test_new

    # gamma1_vec_train[uncens_inds] <-as.numeric(sampler_gamma$predict(xdf_gamma)[, 1]  )
    # gamma1_vec_test <-as.numeric(sampler_gamma$predict( xdf_gamma_test)[, 1])


    if(draw_censored_predictions){
      gamma1_vec_train[cens_inds] <- as.numeric(sampler_gamma$predict(xdf_gamma_all[cens_inds,, drop = FALSE])[, 1])
    }

    # if(sparse){
    tempcounts <- fcount(sampler_gamma$getTrees()$var)
    tempcounts <- tempcounts[tempcounts$x != -1, ]
    var_count_gamma <- rep(0, p_gamma)
    var_count_gamma[tempcounts$x] <- tempcounts$N
    # }

    ############# Phi tree samples ##########################

    # phi_outcome_resids <- ystar[uncens_inds] - mutemp_y - gamma1_vec_train[uncens_inds]*(z[uncens_inds] - offsetz - mutemp_z[uncens_inds])
    a_gamma <- z[uncens_inds] - offsetz - mutemp_z[uncens_inds]
    gamma_numerator <- ystar[uncens_inds] - mutemp_y
    phi_outcome_resids <- gamma_numerator - gamma1_vec_train[uncens_inds] * a_gamma

    # if(iter %% 10 == 0){
    #   a_gamma <- z[uncens_inds] - offsetz - mutemp_z[uncens_inds]
    #   gamma_numerator <- ystar[uncens_inds] - mutemp_y
    #   gamma_component <- gamma1_vec_train[uncens_inds] * a_gamma
    #   phi_outcome_resids <- gamma_numerator - gamma_component
    #   phi_component_diag <- c( sd_ystar = sd(ystar[uncens_inds]),
    #                            sd_mu_y = sd(mutemp_y),
    #                            sd_y_minus_mu = sd(gamma_numerator),
    #                            sd_a_gamma = sd(a_gamma),
    #                            sd_gamma = sd(gamma1_vec_train[uncens_inds]),
    #                            max_abs_gamma = max(abs( gamma1_vec_train[uncens_inds] )),
    #                            sd_gamma_a = sd(gamma_component),
    #                            max_abs_gamma_a = max(abs(gamma_component)),
    #                            sd_phi_resid =  sd(phi_outcome_resids),
    #                            mse_phi_resid = mean(phi_outcome_resids^2))
    #   print(phi_component_diag)
    #   phi_resid_diag <- c(residual_mean = mean(phi_outcome_resids),
    #                       residual_sd = sd(phi_outcome_resids),
    #                       residual_median_abs = median(abs(phi_outcome_resids)),
    #                       residual_q99_abs = quantile( abs(phi_outcome_resids), 0.99 ),
    #                       residual_max_abs = max(abs(phi_outcome_resids)),
    #                       residual_mse = mean(phi_outcome_resids^2),
    #                       phi_median = median( phi1_vec_train[uncens_inds] ),
    #                       phi_max = max( phi1_vec_train[uncens_inds] ) )
    #   print(phi_resid_diag)
    # }


    for (j in 1:n.trees_phi) {
      current_partial_logproduct = phi_log_s2prod - tree_fits_store_logphi[,j]
      Esqvec = phi_outcome_resids^2/exp(current_partial_logproduct)
      # We need the new and old trees for the likelihoods
      new_trees_phi <- curr_trees_phi
      type_phi = sample_move(curr_trees_phi[[j]],
                             iter,
                             n.burnin,
                             trans_prob)

      # Generate a new tree based on the current
      new_trees_phi[[j]] <- update_tree(
        y = Esqvec,
        X = v.train[uncens_inds, , drop = FALSE],
        type = type_phi,
        curr_tree = curr_trees_phi[[j]],
        node_min_size = node_min_size,
        s = s_phi,
        max_bad_trees = max_bad_trees,
        splitting_rules = splitting_rules
      )

      # (c) Obtain the Metropolis-Hastings probability
      curr_tree_phi <- curr_trees_phi[[j]]
      new_tree_phi <- new_trees_phi[[j]]

      # phi_log_check <- get_predictions( curr_trees_phi, v.train[uncens_inds, , drop = FALSE], single_tree = n.trees_phi == 1L )
      # phi_log_error <- max( abs( phi_log_s2prod - phi_log_check ) )
      # if (!is.finite(phi_log_error) || phi_log_error > 1e-10) {
      #   stop( sprintf(paste0( "Phi backfitting mismatch at iteration %d, tree %d: %.3e."),iter,j,phi_log_error))
      # }
      # stored_log_check <- rowSums(tree_fits_store_logphi)
      # stored_log_error <- max( abs( phi_log_s2prod - stored_log_check ) )
      # if (!is.finite(stored_log_error) || stored_log_error > 1e-10) {
      #   stop( sprintf( paste0( "Stored phi-tree fits mismatch at iteration %d, tree %d: %.3e."), iter, j, stored_log_error))
      # }


      if((nrow(new_tree_phi$tree_matrix) == nrow(curr_tree_phi$tree_matrix) ) & (type_phi != "change" )){
        alpha_MH <- 0
        # print("no good trees")
      }else{
        # if(j==1){
        # CURRENT TREE: compute the log of the marginalised likelihood + log of the tree prior
        l_old_phi = tree_full_conditional_HBART(curr_trees_phi[[j]], Esqvec, nuprime, lambdaprime) +
          get_tree_prior(curr_trees_phi[[j]], tree_base_phi, tree_power_phi)
        # }
        # NEW TREE: compute the log of the marginalised likelihood + log of the tree prior
        l_new_phi = tree_full_conditional_HBART(new_trees_phi[[j]], Esqvec, nuprime, lambdaprime) +
          get_tree_prior(new_trees_phi[[j]], tree_base_phi, tree_power_phi)
        alpha_MH = alpha_mh(l_new_phi,l_old_phi, curr_trees_phi[[j]],new_trees_phi[[j]], type_phi)
      }

      if(is.na(alpha_MH)){ stop("line 3075. alpha_MH is NA") }

      if(alpha_MH > runif(1)) {
        curr_trees_phi[[j]] = new_trees_phi[[j]]
        l_old_phi <- l_new_phi
        # if(sparse){
        if (type_phi == "grow") {
          var_count_phi[curr_trees_phi[[j]]$var] <- var_count_phi[curr_trees_phi[[j]]$var] + 1
        } else if (type_phi == "prune") {
          var_count_phi[curr_trees_phi[[j]]$var] <- var_count_phi[curr_trees_phi[[j]]$var] - 1
        } else {
          # if(curr_trees_phi[[j]]$var[1]!=0){ # What if change step returned $var equal to c(0,0) ????
          if (length(curr_trees_phi[[j]]$var) == 2L && all(is.finite(curr_trees_phi[[j]]$var)) && all(curr_trees_phi[[j]]$var > 0L)) {
            var_count_phi[curr_trees_phi[[j]]$var[1]] <- var_count_phi[curr_trees_phi[[j]]$var[1]] - 1
            var_count_phi[curr_trees_phi[[j]]$var[2]] <- var_count_phi[curr_trees_phi[[j]]$var[2]] + 1
          }
        }
        # }
      }
      curr_trees_phi[[j]] = simulate_S_HBART(curr_trees_phi[[j]],
                                             Esqvec,
                                             nuprime,
                                             lambdaprime)
      # Updating BART predictions
      current_fit = get_predictions(curr_trees_phi[[j]], v.train[uncens_inds, , drop = FALSE], single_tree = TRUE)
      phi_log_s2prod = phi_log_s2prod - tree_fits_store_logphi[,j] # subtract the old fit
      phi_log_s2prod = phi_log_s2prod + current_fit # add the new fit
      tree_fits_store_logphi[,j] = current_fit # update the new fit
    } # end loop over phi trees


    #update z_epsilon
    phi_log_s2prod_test <- get_predictions(curr_trees_phi,
                                           v.test,
                                           single_tree = length(curr_trees_phi) == 1)

    if (any(!is.finite(phi_log_s2prod))) {stop(sprintf("Nonfinite log phi at iteration %d.",iter))}
    if (any(!is.finite(phi_log_s2prod_test))) {stop(sprintf("Nonfinite log phi test at iteration %d.",iter))}
    phi1_vec_train[uncens_inds] <- exp(phi_log_s2prod)
    if(draw_censored_predictions){
      phi1_vec_train[cens_inds] = exp(get_predictions(curr_trees_phi, v.train[cens_inds, , drop = FALSE], single_tree = n.trees_phi == 1))
    }
    phi1_vec_test <- exp(phi_log_s2prod_test)
    if (any(!is.finite(phi1_vec_train[uncens_inds])) ||any(phi1_vec_train[uncens_inds] <= 0)) { stop(sprintf("Invalid phi values at iteration %d.",iter))}
    if(draw_censored_predictions){
      if (any(!is.finite(phi1_vec_train[cens_inds])) ||any(phi1_vec_train[cens_inds] <= 0)) { stop(sprintf("Invalid phi values at iteration %d.",iter))}
    }
    if (any(!is.finite(phi1_vec_test)) ||any(phi1_vec_test <= 0)) { stop(sprintf("Invalid phi test values at iteration %d.",iter))}



    # ######### update Sigma matrix #####################################################
    # Sigma_mat <- cbind(c(1,gamma1),c(gamma1,phi1+gamma1^2))
    # Sigma_orig_scale <- cbind(c(1,tempsd*gamma1),c(tempsd*gamma1,  (tempsd^2)*(phi1+gamma1^2)) )
    # ########## tau draws ############
    # if(tau_hyperprior){
    #   tau <- 1/rgamma(n = 1,shape = alpha_tau + 1/2, rate = beta_tau + (1/(2*phi1))*(gamma1 - gamma0)^2)
    # }

    # ########### splitting probability draws #############################
    if (sparse & (iter > floor(n.burnin * 0.5))) {
      s_update_z <- update_s(var_count_z, p_z, alpha_s_z)
      s_z <- s_update_z[[1]]
      s_update_y <- update_s(var_count_y, p_y, alpha_s_y)
      s_y <- s_update_y[[1]]
      s_update_gamma <- update_s(var_count_gamma, p_gamma, alpha_s_gamma)
      s_gamma <- s_update_gamma[[1]]
      s_update_phi <- update_s(var_count_phi, p_phi, alpha_s_phi)
      s_phi <- s_update_phi[[1]]
      if(alpha_split_prior){
        alpha_s_z <- update_alpha(s_z, alpha_scale_z, alpha_a_z, alpha_b_z, p_z, s_update_z[[2]])
        alpha_s_y <- update_alpha(s_y, alpha_scale_y, alpha_a_y, alpha_b_y, p_y, s_update_y[[2]])
        alpha_s_gamma <- update_alpha(s_gamma, alpha_scale_gamma, alpha_a_gamma, alpha_b_gamma, p_gamma, s_update_gamma[[2]])
        alpha_s_phi <- update_alpha(s_phi, alpha_scale_phi, alpha_a_phi, alpha_b_phi, p_phi, s_update_phi[[2]])
      }
    }

    ###### Store results   ###############################

    if(iter > n.burnin){
      iter_min_burnin <- iter-n.burnin
      if (draw_censored_predictions) {
        if (anyNA(x.train[cens_inds, , drop = FALSE])) {
          warning("Cannot draw ystar for censored observations with missing outcome covariates.")
        } else {
          xdf_y_all <- data.frame(x = x.train)
          # mutemp_y_all <- as.numeric(sampler_y$predict(xdf_y_all)[, 1])
          mutemp_y_all[uncens_inds] <- mutemp_y
          mutemp_y_all[cens_inds] <- as.numeric(sampler_y$predict(xdf_y_all[cens_inds, , drop = FALSE])[, 1])
          # ystar_draw_full <- ystar
          # mean_cens_y <- mutemp_y_all[cens_inds] + gamma1_vec_train[cens_inds] * (z[cens_inds] - offsetz - mutemp_z[cens_inds])
          # ystar_draw_full[cens_inds] <- rnorm( length(cens_inds), mean = mean_cens_y, sd = sqrt(phi1_vec_train[cens_inds]) )
          # draw$ystar_train[, iter_min_burnin] <- ystar_draw_full * tempsd + tempmean
        }
      }
      draw$zstar_train[, iter_min_burnin] <- z

      # # after mutemp_y_all, mutemp_z, gamma1, phi1 have been updated
      # zy_train_pred <- if (fast) {
      #   Rfast::rmvnorm(n = n, mu = c(0, 0), sigma = Sigma_mat)
      # } else {
      #   MASS::mvrnorm(n = n, mu = c(0, 0), Sigma = Sigma_mat)
      # }
      # zy_train_pred[, 1] <- zy_train_pred[, 1] + offsetz + mutemp_z
      # zy_train_pred[, 2] <- zy_train_pred[, 2] + mutemp_y_all
      #
      # draw$ystar_train[, iter_min_burnin] <- zy_train_pred[, 2] * tempsd + tempmean
      # draw$zstar_train[, iter_min_burnin] <- zy_train_pred[, 1]
      # draw$ystar_train[uncens_inds, iter_min_burnin] <- ystar[uncens_inds]*tempsd+tempmean

      #NOTE y and z training sample values saved here
      #do not correspond to the the same means and errors as
      #the test values and expectations saved here.
      #However, they are the values to which the trees in this round were fitted.

      #draw z and y for test observations
      zytest <- matrix(NA, nrow = ntest, ncol = 2)
      # for(i in 1:ntest){
      #   # varthetamat_test[i,] <- cbind(mutilde, phitilde, gammatilde)
      #   Sigma_mattemp <- cbind(c(1,gamma1_vec_test[i]), c(gamma1_vec_test[i],phi1_vec_test[i]+gamma1_vec_test[i]^2))
      #   zytest[i,] <- Rfast::rmvnorm(n = 1, mu = c(offsetz + mutemp_test_z[i], mutemp_test_y[i]), sigma = Sigma_mattemp)
      # }
      e1 <- rnorm(ntest)
      e2 <- rnorm(ntest)
      zytest[,1] <- offsetz + mutemp_test_z + e1
      zytest[,2] <- mutemp_test_y + gamma1_vec_test * e1 + sqrt(phi1_vec_test) * e2
      if(fast == TRUE){
        # probcens_train <- fastpnorm(- mutemp_z[uncens_inds] - offsetz )
        probcens_train <- fastpnorm(- mutemp_z - offsetz )
        probcens_test <- fastpnorm(- mutemp_test_z - offsetz)
      }else{
        # probcens_train <- pnorm(- mutemp_z[uncens_inds] - offsetz )
        probcens_train <- pnorm(- mutemp_z - offsetz )
        probcens_test <- pnorm(- mutemp_test_z - offsetz)
      }

      #calculate conditional expectation
      # condexptrain <- mutemp_y + gamma1*(dnorm(- mutemp_z - offsetz ))/(1-probcens_train)
      # condexptrain <- mutemp_y + gamma1*(dnorm(- mutemp_z[uncens_inds] - offsetz ))/(1-probcens_train)
      # condexptest <- mutemp_test_y + gamma1*(dnorm(- mutemp_test_z - offsetz ))/(1-probcens_test)

      # temp_ztrain <- mutemp_z[uncens_inds] + offsetz
      temp_ztrain <- mutemp_z + offsetz
      temp_ztest <- mutemp_test_z + offsetz
      #
      #       if(fast == TRUE){
      #         IMR_train <- exp( dnorm(temp_ztrain,log=T) - log(fastpnorm(temp_ztrain) ))
      #         IMR_test <- exp( dnorm(temp_ztest,log=T) - log(fastpnorm(temp_ztest) ))
      #         # IMR_train <- fastnormdens(temp_ztrain)/fastpnorm(temp_ztrain)
      #         # IMR_test <- fastnormdens(temp_ztest)/fastpnorm(temp_ztest)
      #       }else{
      IMR_train <- exp( dnorm(temp_ztrain,log=TRUE) - pnorm(temp_ztrain,log.p = TRUE) )
      IMR_test <- exp( dnorm(temp_ztest,log=TRUE) - pnorm(temp_ztest,log.p = TRUE) )
      # }

      # if(cor(IMR_train,mutemp_y_all)> 0.9){
      #   stop("Highly correlated f_y and IMR")
      # }

      # cor_IMR_fy <- suppressWarnings(cor(IMR_train[uncens_inds], mutemp_y))
      # if (is.finite(cor_IMR_fy) && abs(cor_IMR_fy) > 0.9 && verbose_diagnostics) {
      #   warning(sprintf("High posterior draw correlation between IMR and f_y: %.3f", cor_IMR_fy))
      # }

      # if(draw_censored_predictions){
      #   condexptrain <- mutemp_y_all + gamma1*IMR_train
      #   draw$cond_exp_train[, iter_min_burnin] <- condexptrain*tempsd+tempmean
      #   draw$mu_y_train[, iter_min_burnin] <- mutemp_y_all*tempsd+tempmean
      # }
      # draw$cond_exp_train_uncens[, iter_min_burnin] <- (mutemp_y + gamma1*IMR_train[uncens_inds])*tempsd+tempmean
      # condexptest <- mutemp_test_y + gamma1*IMR_test

      if(draw_censored_predictions){
        condexptrain <- mutemp_y_all + gamma1_vec_train*IMR_train#*
        draw$cond_exp_train[, iter_min_burnin] <- condexptrain*tempsd+tempmean
        draw$mu_y_train[, iter_min_burnin] <- (mutemp_y_all) * tempsd + tempmean
      }
      draw$cond_exp_train_uncens[, iter_min_burnin] <- (mutemp_y  + gamma1_vec_train[uncens_inds]*IMR_train[uncens_inds])*tempsd+tempmean#*
      condexptest <- mutemp_test_y + gamma1_vec_test*IMR_test #(dnorm(- mutemp_test_z - offsetz - varthetamat_test[,1] ))/(1-probcens_test)

      # draw$Z.mat_train[,iter_min_burnin] <- z
      # draw$Z.mat_test[,iter_min_burnin] <-  zytest[,1]
      # draw$Y.mat_train = array(NA, dim = c(n, n.iter)),
      # draw$Y.mat_test = array(NA, dim = c(ntest, n.iter)),

      draw$mu_y_test[, iter_min_burnin] <- mutemp_test_y*tempsd+tempmean
      # draw$mucens_y_train[, iter_min_burnin] <- mutemp_y[cens_inds]
      # draw$muuncens_y_train[, iter_min_burnin] <- mutemp_y[uncens_inds]
      draw$muuncens_y_train[, iter_min_burnin] <- mutemp_y*tempsd+tempmean
      draw$mu_z_train[, iter_min_burnin] <- mutemp_z + offsetz
      draw$mu_z_test[, iter_min_burnin] <- mutemp_test_z + offsetz
      draw$train.probcens[, iter_min_burnin] <-  probcens_train
      draw$test.probcens[, iter_min_burnin] <-  probcens_test
      draw$cond_exp_test[, iter_min_burnin] <- condexptest*tempsd+tempmean
      # draw$ystar_train[, iter_min_burnin] <- ystar*tempsd+tempmean
      draw$ystar_test[, iter_min_burnin] <- zytest[,2]*tempsd+tempmean
      # draw$zstar_train[,iter_min_burnin] <- z
      draw$zstar_test[,iter_min_burnin] <-  zytest[,1]
      draw$ycond_test_matrix[, iter_min_burnin] <- ifelse( draw$zstar_test[,iter_min_burnin] >= 0, draw$ystar_test[, iter_min_burnin], NA_real_ )

      draw$ycond_draws_train[[iter_min_burnin]] <-  ystar[uncens_inds]*tempsd+tempmean
      draw$ycond_draws_test[[iter_min_burnin]] <-  zytest[,2][zytest[,1] >= 0]*tempsd+tempmean

      # draw$Sigma_draws[,, iter_min_burnin] <- Sigma_orig_scale#Sigma_mat
      draw$phivec_draws_train[, iter_min_burnin] <- phi1_vec_train * tempsd^2
      draw$phivec_draws_test[, iter_min_burnin] <- phi1_vec_test * tempsd^2
      draw$gammavec_draws_train[, iter_min_burnin] <- gamma1_vec_train * tempsd
      draw$gammavec_draws_test[, iter_min_burnin] <- gamma1_vec_test * tempsd

      if (draw_censored_predictions) {
        ystar_store <- ystar
        cens_mean <- mutemp_y_all[cens_inds] + gamma1_vec_train[cens_inds] *(z[cens_inds] - offsetz - mutemp_z[cens_inds] )
        ystar_store[cens_inds] <- rnorm(n0,mean = cens_mean,sd = sqrt(phi1_vec_train[cens_inds]))
        draw$ystar_train[, iter_min_burnin] <- ystar_store * tempsd + tempmean
      }

      if(has_numeric_censor){
        # uncondexptrain <- censored_value*probcens_train +  mutemp_y*(1- probcens_train ) + gamma1*dnorm(- mutemp_z - offsetz )
        if(draw_censored_predictions){
          # uncondexptrain <- censored_value*probcens_train +  mutemp_y_all*(1- probcens_train ) + gamma1*dnorm(- mutemp_z - offsetz )
          uncondexptrain <- censored_value*probcens_train + mutemp_y_all*(1- probcens_train ) +
            gamma1_vec_train*dnorm(- mutemp_z - offsetz)
          draw$uncond_exp_train[, iter_min_burnin] <- uncondexptrain*tempsd+tempmean
        }
        uncondexptrain_uncens <- censored_value*probcens_train[uncens_inds] +
          mutemp_y*(1- probcens_train[uncens_inds] ) +
          gamma1_vec_train[uncens_inds]*dnorm(- mutemp_z[uncens_inds] - offsetz )
        draw$uncond_exp_train_uncens[, iter_min_burnin] <- uncondexptrain_uncens*tempsd+tempmean
        uncondexptest <- censored_value*probcens_test +  mutemp_test_y*(1 - probcens_test ) +
          gamma1_vec_test*dnorm(- mutemp_test_z - offsetz)
        draw$uncond_exp_test[, iter_min_burnin] <- uncondexptest*tempsd+tempmean
        draw$ydraws_test[, iter_min_burnin] <- ifelse(zytest[,1] < 0, censored_value, zytest[,2] )*tempsd+tempmean
      }

      draw$var_count_y_store[iter_min_burnin,] <- var_count_y
      draw$var_count_z_store[iter_min_burnin,] <- var_count_z
      draw$var_count_gamma_store[iter_min_burnin,] <- var_count_gamma
      draw$var_count_phi_store[iter_min_burnin,] <- var_count_phi
      if(sparse){
        draw$alpha_s_y_store[iter_min_burnin] <- alpha_s_y
        draw$alpha_s_z_store[iter_min_burnin] <- alpha_s_z
        draw$alpha_s_gamma_store[iter_min_burnin] <- alpha_s_gamma
        draw$alpha_s_phi_store[iter_min_burnin] <- alpha_s_phi
        draw$s_prob_y_store[iter_min_burnin,] <- s_y
        draw$s_prob_z_store[iter_min_burnin,] <- s_z
        draw$s_prob_gamma_store[iter_min_burnin,] <- s_gamma
        draw$s_prob_phi_store[iter_min_burnin,] <- s_phi
      }
      if(tau_hyperprior){
        draw$tau_par[iter_min_burnin] <- tau
      }
    } # end if iter > burnin
    # pb$tick()
    if(iter %% print.opt == 0){
      print(paste("Gibbs Iteration", iter))
      # print(c(sigma2.alpha, sigma2.beta))
    }
  }#end iterations of Giibs sampler

  draw$preprocess <- list(
    y_mean = tempmean,
    y_sd = tempsd,
    # cens_inds = cens_inds,
    # uncens_inds = uncens_inds,
    offsetz = offsetz,
    censored_value_scaled = censored_value
  )
  return(draw)
}

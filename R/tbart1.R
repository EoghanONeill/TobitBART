

#' @title Type I Tobit Bayesian Additive Regression Trees implemented using MCMC
#'
#' @description Type I Tobit Bayesian Additive Regression Trees implemented using MCMC
#' @import dbarts
#' @import collapse
#' @import GIGrvg
#' @import MASS
#' @import dqrng
#' @import data.table
#' @import accelerometry
#' @import wrswoR
#' @import truncnorm
#' @import mvtnorm
#' @import censReg
#' @import fastncdf
#' @import SoftBart
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
#' @param tree_power Tree prior parameter for outcome model.
#' @param tree_base Tree prior parameter for outcome model.
#' @param node.prior (dbarts option) An expression of the form dbarts:::normal or dbarts:::normal(k) that sets the prior used on the averages within nodes.
#' @param resid.prior (dbarts option) An expression of the form dbarts:::chisq or dbarts:::chisq(df,quant) that sets the prior used on the residual/error variance
#' @param proposal.probs (dbarts option) Named numeric vector or NULL, optionally specifying the proposal rules and their probabilities. Elements should be "birth_death", "change", and "swap" to control tree change proposals, and "birth" to give the relative frequency of birth/death in the "birth_death" step.
#' @param sigmadbarts (dbarts option) A positive numeric estimate of the residual standard deviation. If NA, a linear model is used with all of the predictors to obtain one.
#' @param print.opt Print every print.opt number of Gibbs samples.
#' @param fast If equal to TRUE, then implements faster truncated normal draws and approximates normal pdf.
#' @param sparse If equal to TRUE, use Linero Dirichlet prior on splitting probabilities
#' @param alpha_a_y Linero alpha prior parameter for outcome equation splitting probabilities
#' @param alpha_b_y Linero alpha prior parameter for outcome equation splitting probabilities
#' @param alpha_split_prior If true, set hyperprior for Linero alpha parameter
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
#' \item{alpha_s_y_store}{For Dirichlet prior on splitting probabilities in outcome equation, vector of alpha hyperparameter draws for each iteration.}
#' \item{var_count_y_store}{Matrix of counts of splits on each variable in outcome observation. The number of rows is the number of potential splitting variables. The number of columns is the number of post-burn-in iterations.}
#' \item{s_prob_y_store}{Splitting probabilities for the outcome equation. The number of rows is the number of potential splitting variables. The number of columns is the number of post-burn-in iterations. }
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

tbart1 <- function(x.train,
                   x.test,
                   y,
                   n.iter=1000,
                   n.burnin=100,
                   below_cens = 0,
                   above_cens = Inf,
                   n.trees = 50L,
                   n.burn = 0L,
                   n.samples = 1L,
                   n.thin = 1L,
                   n.chains = 1,
                   n.threads = 1L,#guessNumCores(),
                   printEvery = 100L,
                   printCutoffs = 0L,
                   rngKind = "default",
                   rngNormalKind = "default",
                   rngSeed = NA_integer_,
                   updateState = TRUE,
                   tree_power = 2,
                   tree_base = 0.95,
                   node.prior = dbarts:::normal,
                   resid.prior = dbarts:::chisq,
                   proposal.probs = c(birth_death = 0.5, swap = 0.1, change = 0.4, birth = 0.5),
                   sigmadbarts = NA_real_,
                   print.opt = 100,
                   fast=TRUE,
                   censsigprior = FALSE,
                   lambda0 = NA,
                   sigest = NA,
                   nu0=3,
                   sigquant = 0.90,
                   nolinregforlambda = FALSE,
                   sparse = FALSE,
                   alpha_a_y = 0.5,
                   alpha_b_y = 1,
                   alpha_split_prior = TRUE){




  # if(is.vector(x.train) | is.factor(x.train)| is.data.frame(x.train)) x.train = as.matrix(x.train)
  # if(is.vector(x.test) | is.factor(x.test)| is.data.frame(x.test)) x.test = as.matrix(x.test)

  # if((!is.matrix(x.train))) stop("argument x.train must be a double matrix")
  # if((!is.matrix(x.test)) ) stop("argument x.test must be a double matrix")

  if(nrow(x.train) != length(y)) stop("number of rows in x.train must equal length of y.train")
  if((ncol(x.test)!=ncol(x.train))) stop("input x.test must have the same number of columns as x.train")


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

  if(!is.na(sigest)){
    if(sigest == "naive"){
      censsigest <- sd(z)
      censsigprior <- TRUE
    }
  }

  n <- length(y)
  n0 <- length(cens_inds)
  n1 <- length(uncens_inds)
  n_censbelow <- length(which(y <= below_cens))
  n_censabove <- length(which(y >= above_cens))

  ntest = nrow(x.test)

  p_y <- ncol(x.train)

  if(sparse){
    s_y <- rep(1 / p_y, p_y) # probability vector to be used during the growing process for DART feature weighting
    rho_y <- p_y # For DART

    if(alpha_split_prior){
      alpha_s_y <- p_y
    }else{
      alpha_s_y <- 1
    }
    alpha_scale_y <- p_y
  }



  if(censsigprior == TRUE){

    if(is.na(lambda0)) {
      if(is.na(sigest)) {
        if( (ncol(x.train) < n) & !nolinregforlambda ) {
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
          censsigest  <- exp(sum_est$estimate["logSigma", "Estimate"])



        } else {
          df0 <- data.frame(y)


          estResult <- censReg(y ~ 1,left = below_cens, right = above_cens, data = df0)


          sum_est <- summary( estResult )

          # print("sum_est = ")
          # print(sum_est)
          censsigest  <- exp(sum_est$estimate["logSigma", "Estimate"])


          # sigest = sd(y.train)
        }
      }
      # qchi = qchisq(1.0-sigquant,nu0)
      # lambda0 = (censsigest *censsigest *qchi)/nu0 #lambda parameter for sigma prior
    } else {
      censsigest =sqrt(lambda0)
    }

  }








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



  if(sparse){
    var_count_y <- rep(0, p_y)

    draw$alpha_s_y_store <- rep(NA, n.iter)
    draw$var_count_y_store <- matrix(0, ncol = p_y, nrow = n.iter)
    draw$s_prob_y_store <- matrix(0, ncol = p_y, nrow = n.iter)
  }






  control <- dbartsControl(updateState = updateState, verbose = FALSE,  keepTrainingFits = TRUE,
                           keepTrees = TRUE,
                           n.trees = n.trees,
                           n.burn = n.burn,
                           n.samples = n.samples,
                           n.thin = n.thin,
                           n.chains = n.chains,
                           n.threads = n.threads,
                           printEvery = printEvery,
                           printCutoffs = printCutoffs,
                           rngKind = rngKind,
                           rngNormalKind = rngNormalKind,
                           rngSeed = rngSeed)


  # print(colnames(Xmat.train))

  # print("begin dbarts")

  dftrain <- data.frame(y = z , x = x.train)

  if(nrow(x.test )==0){
    sampler <- dbarts(y ~ .,
                      data = dftrain, # as.data.frame(x.train),
                      #test = x.test,
                      control = control,
                      tree.prior = dbarts:::cgm(power = tree_power, base =  tree_base,  split.probs = rep(1 / p_y, p_y)),
                      node.prior = node.prior,
                      resid.prior = dbarts:::chisq(df = nu0,quant = sigquant),
                      proposal.probs = proposal.probs,
                      sigma = sigmadbarts
    )

  }else{
    dftest <- data.frame(y = NA , x = x.test)

    sampler <- dbarts(y ~ .,
                      data = dftrain, # as.data.frame(x.train),
                      test = dftest, # as.data.frame(x.test),
                      control = control,
                      tree.prior = dbarts:::cgm(power = tree_power, base =  tree_base,  split.probs = rep(1 / p_y, p_y)),
                      node.prior = node.prior,
                      resid.prior = dbarts:::chisq(df = nu0,quant = sigquant),
                      proposal.probs = proposal.probs,
                      sigma = sigmadbarts
    )

  }


  if(censsigprior == TRUE){

    sigest <- sampler$`.->data`@sigma

    qchi = qchisq(1.0-sigquant,nu0)
    lambda0 = (sigest*sigest*qchi)/nu0 #lambda parameter for sigma prior

    # check if this is the lambda0 value for dbarts?

    # now suppose a different sigest is obtained due to censoring
    # censsigest <- 2
    # keep same degrees of freedom, obtain new implied lambda
    qchi = qchisq(1.0-sigquant,nu0)
    censlambda0 = (censsigest*censsigest*qchi)/nu0 #lambda parameter for sigma prior

    # now find the sigquant value that would give cemslambda0
    # if used original sigest
    #i.e. want
    qchitarget <-  censlambda0*nu0/(sigest*sigest)

    newsigquant <- 1 - pchisq(q = qchitarget, df = nu0)

    # newqchi <- qchisq(1.0-newsigquant,nu0)

    # now edit sampler prior


    ### set the sampler again with the adjusted variance
    if(nrow(x.test )==0){
      sampler <- dbarts(y ~ .,
                        data = dftrain, # as.data.frame(x.train),
                        #test = x.test,
                        control = control,
                        tree.prior = dbarts:::cgm(power = tree_power, base =  tree_base,  split.probs = rep(1 / p_y, p_y)),
                        node.prior = node.prior,
                        resid.prior = dbarts:::chisq(df = nu0,quant = newsigquant),
                        proposal.probs = proposal.probs,
                        sigma = sigmadbarts
      )

    }else{
      dftest <- data.frame(y = NA , x = x.test)

      sampler <- dbarts(y ~ .,
                        data = dftrain, # as.data.frame(x.train),
                        test = dftest, # as.data.frame(x.test),
                        control = control,
                        tree.prior = dbarts:::cgm(power = tree_power, base =  tree_base,  split.probs = rep(1 / p_y, p_y)),
                        node.prior = node.prior,
                        resid.prior = dbarts:::chisq(df = nu0,quant = newsigquant),
                        proposal.probs = proposal.probs,
                        sigma = sigmadbarts
      )

    }
  }





  sampler$setResponse(y = z)
  # sampler$setSigma(sigma = 1)

  #sampler$setPredictor(x= Xmat.train$x, column = 1, forceUpdate = TRUE)

  if(sparse){
    tempmodel <- sampler$model
    tempmodel@tree.prior@splitProbabilities <- s_y
    sampler$setModel(newModel = tempmodel)
  }


  #mu = as.vector( alpha + X.mat %*% beta )
  sampler$sampleTreesFromPrior()
  samplestemp <- sampler$run()

  #mutemp <- samplestemp$train[,1]
  #suppose there are a number of samples

  # print("sigma = ")
  sigma <- samplestemp$sigma

  mu <- samplestemp$train[,1]
  mutest <- samplestemp$test[,1]

  if(sparse){
    tempcounts <- fcount(sampler$getTrees()$var)
    tempcounts <- tempcounts[tempcounts$x != -1, ]
    var_count_y <- rep(0, p_y)
    var_count_y[tempcounts$x] <- tempcounts$N
  }

  ystar <- rnorm(n,mean = mu, sd = sigma)
  ystartest <- rnorm(ntest,mean = mutest, sd = sigma)


  # if(fast){
  #   ystartestcens <-tnorm(ntest, a = below_cens, b = above_cens, mean = mutest, sd = sigma)
  #
  #
  # }else{
    ystartestcens <- rtruncnorm(ntest, a = below_cens, b = above_cens, mean = mutest, sd = sigma)

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
    draw$sigma[1] <- sigma

    draw$cond_exp_train[, 1] <- condexptrain
    draw$cond_exp_test[, 1] <- condexptest

    if(sparse){
      draw$alpha_s_y_store[1] <- alpha_s_y
      # draw$alpha_s_z_store[1] <- alpha_s_z
      draw$var_count_y_store[1,] <- var_count_y
      # draw$var_count_z_store[1,] <- var_count_z
      draw$s_prob_y_store[1,] <- s_y
      # draw$s_prob_z_store[1,] <- s_z

    }

}

  #loop through the Gibbs sampler iterations
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
    sampler$setResponse(y = z)
    # sampler$setSigma(sigma = 1)
    #sampler$setPredictor(x= Xmat.train$x, column = 1, forceUpdate = TRUE)

    if(sparse){
      tempmodel <- sampler$model
      tempmodel@tree.prior@splitProbabilities <- s_y
      sampler$setModel(newModel = tempmodel)
    }

    #mu = as.vector( alpha + X.mat %*% beta )
    samplestemp <- sampler$run()

    sigma <- samplestemp$sigma

    mu <- samplestemp$train[,1]
    mutest <- samplestemp$test[,1]

    if(sparse){
      tempcounts <- fcount(sampler$getTrees()$var)
      tempcounts <- tempcounts[tempcounts$x != -1, ]
      var_count_y[tempcounts$x] <- tempcounts$N
    }





    ########### splitting probability draws #############################


    if (sparse & (iter > floor(n.burnin * 0.5))) {
      # s_update_z <- update_s(var_count_z, p_z, alpha_s_z)
      # s_z <- s_update_z[[1]]

      s_update_y <- update_s(var_count_y, p_y, alpha_s_y)
      s_y <- s_update_y[[1]]

      if(alpha_split_prior){
        # alpha_s_z <- update_alpha(s_z, alpha_scale_z, alpha_a_z, alpha_b_z, p_z, s_update_z[[2]])
        alpha_s_y <- update_alpha(s_y, alpha_scale_y, alpha_a_y, alpha_b_y, p_y, s_update_y[[2]])
      }
    }







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






    if(iter>n.burnin){
      iter_min_burnin <- iter-n.burnin
      draw$Z.mat[,iter_min_burnin] = z
      draw$Z.matcens[,iter_min_burnin] = z[cens_inds]
      # draw$Z.matuncens[,iter_min_burnin] = z[uncens_inds]
      draw$Z.matcensbelow[,iter_min_burnin] = z[censbelow_inds]
      draw$Z.matcensabove[,iter_min_burnin] = z[censabove_inds]
      draw$mu[,iter_min_burnin] = mu
      draw$mucens[,iter_min_burnin] = mu[cens_inds]
      draw$muuncens[,iter_min_burnin] = mu[uncens_inds]
      draw$mucensbelow[,iter_min_burnin] = mu[censbelow_inds]
      draw$mucensabove[,iter_min_burnin] = mu[censabove_inds]
      draw$ystar[,iter_min_burnin] = ystar
      draw$ystarcens[,iter_min_burnin] = ystar[cens_inds]
      draw$ystaruncens[,iter_min_burnin] = ystar[uncens_inds]
      draw$ystarcensbelow[,iter_min_burnin] = ystar[censbelow_inds]
      draw$ystarcensabove[,iter_min_burnin] = ystar[censabove_inds]
      draw$test.mu[,iter_min_burnin] = mutest
      draw$test.y_nocensoring[,iter_min_burnin] = ystartest
      draw$test.y_withcensoring[,iter_min_burnin] = ystartestcens
      draw$test.probcensbelow[,iter_min_burnin] = probcensbelow
      draw$test.probcensabove[,iter_min_burnin] = probcensabove
      draw$sigma[iter_min_burnin] <- sigma

      draw$cond_exp_train[, iter_min_burnin] <- condexptrain
      draw$cond_exp_test[, iter_min_burnin] <- condexptest
      if(sparse){
        draw$alpha_s_y_store[iter_min_burnin] <- alpha_s_y
        # draw$alpha_s_z_store[iter_min_burnin] <- alpha_s_z
        draw$var_count_y_store[iter_min_burnin,] <- var_count_y
        # draw$var_count_z_store[iter_min_burnin,] <- var_count_z
        draw$s_prob_y_store[iter_min_burnin,] <- s_y
        # draw$s_prob_z_store[iter_min_burnin,] <- s_z
      }
    }

    if(iter %% print.opt == 0){
      print(paste("Gibbs Iteration", iter))
      # print(c(sigma2.alpha, sigma2.beta))
    }


  }#end iterations of Giibs sampler



  return(draw)



}

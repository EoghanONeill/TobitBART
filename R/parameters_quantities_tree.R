# -------------------------------------------------------------------------#
# Description: this script contains 2 functions that are used to generate  #
#              the predictions, update variance and compute the tree prior #
#              and the marginalised likelihood                             #
# -------------------------------------------------------------------------#

# 1. simulate_mu: generate the predicted values (mu's)
# 2. update_sigma2: updates the parameters sigma2
# 3. update_z: updates the latent variables z. This is required for MOTR-BART for classification.
# 4. get_tree_prior: returns the tree log prior score
# 5. tree_full_conditional: computes the marginalised likelihood for all nodes for a given tree
# 6. get_number_distinct_cov: counts the number of distinct covariates that are used in a tree to create the splitting rules
# Compute the full conditionals -------------------------------------------------

tree_full_conditional = function(tree, R, sigma2, sigma2_mu) {

  # Function to compute log full conditional distirbution for an individual tree
  # R is a vector of partial residuals

  # Need to calculate log complete conditional, involves a sum over terminal nodes

  # First find which rows are terminal nodes
  which_terminal = which(tree$tree_matrix[,'terminal'] == 1)

  # Get node sizes for each terminal node
  nj = tree$tree_matrix[which_terminal,'node_size']
  # nj = tree$tree_matrix[tree$tree_matrix[,'terminal'] == 1,'node_size']

  # nj <- nj[nj!=0]
  # Get sum of residuals and sum of residuals squared within each terminal node
  # sumRsq_j = aggregate(R, by = list(tree$node_indices), function(x) sum(x^2))[,2]
  # S_j = aggregate(R, by = list(tree$node_indices), sum)[,2]

  # sumRsq_j = fsum(R^2, tree$node_indices)
  # S_j = fsum(R, tree$node_indices)
  S_j = fsum(R,factor(tree$node_indices, levels = which_terminal ), fill = TRUE)

  if(length(S_j) != length(nj)){
    print("S_j = ")
    print(S_j)

    print("nj = ")
    print(nj)

    print("tree$node_indices = ")
    print(tree$node_indices)

    print("tree$tree_matrix = ")
    print(tree$tree_matrix)

    # print("n_j = ")
    # print(n_j)
  }

  # Now calculate the log posterior
  log_post = 0.5 * ( sum(log( sigma2 / (nj*sigma2_mu + sigma2))) +
              sum( (sigma2_mu* S_j^2) / (sigma2 * (nj*sigma2_mu + sigma2))))
  return(log_post)
}


tree_full_conditional_weighted = function(tree, R, #sigma2,
                                          sigma2_mu, weight_vec) {

  # Function to compute log full conditional distirbution for an individual tree
  # R is a vector of partial residuals

  # Need to calculate log complete conditional, involves a sum over terminal nodes

  # First find which rows are terminal nodes
  which_terminal = which(tree$tree_matrix[,'terminal'] == 1)

  # Get node sizes for each terminal node
  nj = tree$tree_matrix[which_terminal,'node_size']
  # nj = tree$tree_matrix[tree$tree_matrix[,'terminal'] == 1,'node_size']

  # nj <- nj[nj!=0]
  # Get sum of residuals and sum of residuals squared within each terminal node
  # sumRsq_j = aggregate(R, by = list(tree$node_indices), function(x) sum(x^2))[,2]
  # S_j = aggregate(R, by = list(tree$node_indices), sum)[,2]

  # sumRsq_j = fsum(R^2, tree$node_indices)
  # S_j = fsum(R, tree$node_indices)
  # S_j = fsum(R,factor(tree$node_indices, levels = which_terminal ), fill = TRUE)
  wbyS_j = fsum(weight_vec*R,factor(tree$node_indices, levels = which_terminal ), fill = TRUE)
  wsum_j = fsum(weight_vec,factor(tree$node_indices, levels = which_terminal ), fill = TRUE)


  # # sum_wbyRsq_j = fsum(weight_vec*R^2, tree$node_indices)
  # sum_wbyRsq_j = sum(weight_vec*R^2)


  if(length(wbyS_j) != length(nj)){
    print("S_j = ")
    print(S_j)

    print("nj = ")
    print(nj)

    print("tree$node_indices = ")
    print(tree$node_indices)

    print("tree$tree_matrix = ")
    print(tree$tree_matrix)

    # print("n_j = ")
    # print(n_j)
  }

  # # Now calculate the log posterior
  # log_post = - sum(log(1/sqrt(weight_vec))) - 0.5*sum(log(1 + sigma2_mu*wsum_j)) - 0.5*sum_wbyRsq_j +
  #   0.5*sum( ( wbyS_j^2) / ( (1/sigma2_mu)  +  wsum_j  ))
  #
  # log_post = 0.5* sum(log(weight_vec))  - 0.5*sum(log(1 + sigma2_mu*wsum_j)) - 0.5*sum_wbyRsq_j +
  #   0.5*sum( ( wbyS_j^2) / ( (1/sigma2_mu)  +  wsum_j  ))

  # removing terms that cancel out
  log_post =  0.5*sum( -log(1 + sigma2_mu*wsum_j)  +  ( wbyS_j^2) / ( (1/sigma2_mu)  +  wsum_j  ) )


  if(is.na(log_post)){
    print("wsum_j = ")
    print(wsum_j)

    print("wbyS_j = ")
    print(wbyS_j)

    print("sigma2_mu = ")
    print(sigma2_mu)

    print("tree$node_indices = ")
    print(tree$node_indices)

    print("tree$tree_matrix = ")
    print(tree$tree_matrix)

    print("log(1 + sigma2_mu*wsum_j) = ")
    print(log(1 + sigma2_mu*wsum_j))

    print("( wbyS_j^2) / ( (1/sigma2_mu)  +  wsum_j  ) = ")
    print(( wbyS_j^2) / ( (1/sigma2_mu)  +  wsum_j  ))

  }

  return(log_post)
}




tree_full_conditional_z_marg = function(tree, R, #sigma2,
                                        sigma2_mu,
                                        weightstemp,
                                        weightz,
                                        binmat_all_z, cens_inds, uncens_inds, BtB_z_u, BtB_z_c) {

  # Function to compute log full conditional distirbution for an individual tree
  # R is a vector of partial residuals

  # Need to calculate log complete conditional, involves a sum over terminal nodes

  # # First find which rows are terminal nodes
  # which_terminal = which(tree$tree_matrix[,'terminal'] == 1)
  #
  # # Get node sizes for each terminal node
  # nj = tree$tree_matrix[which_terminal,'node_size']
  # # nj = tree$tree_matrix[tree$tree_matrix[,'terminal'] == 1,'node_size']
  #
  # # nj <- nj[nj!=0]
  # # Get sum of residuals and sum of residuals squared within each terminal node
  # # sumRsq_j = aggregate(R, by = list(tree$node_indices), function(x) sum(x^2))[,2]
  # # S_j = aggregate(R, by = list(tree$node_indices), sum)[,2]
  #
  # # sumRsq_j = fsum(R^2, tree$node_indices)
  # # S_j = fsum(R, tree$node_indices)
  # # S_j = fsum(R,factor(tree$node_indices, levels = which_terminal ), fill = TRUE)
  #
  # # S_j = fsum(R,factor(tree$node_indices, levels = which_terminal ), w = weightstemp, fill = TRUE)
  # S_j = fsum(R*weightstemp,factor(tree$node_indices, levels = which_terminal ), fill = TRUE)

  # check if this is faster then binary matrix approach. Can avoid creating the binary matrix entirely
  # S_j <- rep(0,0) # initialize as numeric vector of length 0
  # for(i in 1:length(trees)){
  #   which_terminal = which(trees[[i]]$tree_matrix[,'terminal'] == 1)
  #   nj = trees[[i]]$tree_matrix[which_terminal,'node_size']
  #   S_j = c(S_j,fsum(R*weightstemp,factor(trees[[i]]$node_indices, levels = which_terminal ), fill = TRUE))
  # }
  S_j = crossprod(binmat_all_z, R*weightstemp)

  tempmat <- BtB_z_u*weightz + BtB_z_c + (1/sigma2_mu)*diag(ncol(BtB_z_u))

  # log_post <- 0.5*(- ncol(BtB_z_u)*log(sigma2_mu) + determinant(tempmat, logarithm = TRUE) +
  #                    t(S_j)%*%solve(tempmat)%*% S_j)

  U = chol ( tempmat)
  IR = backsolve (U , diag ( ncol(BtB_z_u) ))
  # btilde = crossprod ( t ( IR ))%*%( crossprod (X_node , r_node ) )
  # beta_hat = btilde + sqrt ( sigma2 )* IR %*% rnorm ( p )
  tSjtempmatinvSj = crossprod ( crossprod( IR , S_j ) )


  log_post <- 0.5*(- ncol(BtB_z_u)*log(sigma2_mu) + #determinant(tempmat, logarithm = TRUE) +
                     tSjtempmatinvSj) - sum(log(diag(U)))

  # Now calculate the log posterior
  # log_post = 0.5 * ( sum(log( sigma2 / (nj*sigma2_mu + sigma2))) +
  #                      sum( (sigma2_mu* S_j^2) / (sigma2 * (nj*sigma2_mu + sigma2))))
  return(log_post)
}




tree_full_conditional_y_marg_nogamma = function(tree, R, #sigma2,
                                                phi,
                                                sigma2_mu,
                                        binmat_all_y, BtB_y) {

  # Function to compute log full conditional distirbution for an individual tree
  # R is a vector of partial residuals

  S_j = crossprod(binmat_all_y, R)
  tempmat <- BtB_y*(1/phi)  + (1/sigma2_mu)*diag(ncol(BtB_y))
  U = chol ( tempmat)
  IR = backsolve (U , diag ( ncol(BtB_y) ))
  # btilde = crossprod ( t ( IR ))%*%( crossprod (X_node , r_node ) )
  # beta_hat = btilde + sqrt ( sigma2 )* IR %*% rnorm ( p )
  tSjtempmatinvSj = crossprod ( crossprod( IR , S_j*(1/phi) ) )

  log_post <- 0.5*(- ncol(BtB_y)*log(sigma2_mu) + #determinant(tempmat, logarithm = TRUE) +
                     tSjtempmatinvSj) - sum(log(diag(U)))

  # Now calculate the log posterior
  # log_post = 0.5 * ( sum(log( sigma2 / (nj*sigma2_mu + sigma2))) +
  #                      sum( (sigma2_mu* S_j^2) / (sigma2 * (nj*sigma2_mu + sigma2))))
  return(log_post)
}


tree_full_conditional_y_marg = function(trees, R, sigma2, priorgammavar, sigma2_mu, binmat_all_y_z, BtB_y, gamma0) {

  # Function to compute log full conditional distirbution for an individual tree
  # R is a vector of partial residuals

  # Need to calculate log complete conditional, involves a sum over terminal nodes

  # # First find which rows are terminal nodes
  # which_terminal = which(tree$tree_matrix[,'terminal'] == 1)
  #
  # # Get node sizes for each terminal node
  # nj = tree$tree_matrix[which_terminal,'node_size']
  # # nj = tree$tree_matrix[tree$tree_matrix[,'terminal'] == 1,'node_size']
  #
  # # nj <- nj[nj!=0]
  # # Get sum of residuals and sum of residuals squared within each terminal node
  # # sumRsq_j = aggregate(R, by = list(tree$node_indices), function(x) sum(x^2))[,2]
  # # S_j = aggregate(R, by = list(tree$node_indices), sum)[,2]
  #
  # # sumRsq_j = fsum(R^2, tree$node_indices)
  # # S_j = fsum(R, tree$node_indices)
  # S_j = fsum(R,factor(tree$node_indices, levels = which_terminal ), fill = TRUE)

  # S_j <- rep(0,0) # initialize as numeric vector of length 0
  # for(i in 1:length(trees)){
  #   which_terminal = which(trees[[i]]$tree_matrix[,'terminal'] == 1)
  #   nj = trees[[i]]$tree_matrix[which_terminal,'node_size']
  #   S_j = c(S_j,fsum(R,factor(trees[[i]]$node_indices, levels = which_terminal ), fill = TRUE))
  # }
  # S_j = c(S_j, crossprod(binmat_all_y_z[,ncol(binmat_all_y_z)], R) )
  S_j = crossprod(binmat_all_y_z, R)


  # there are two ways of calculating the marginal probability
  tempmat <- (1/sigma2)*BtB_y
  diag(tempmat) <- diag(tempmat) + c(rep(1/sigma2_mu, ncol(BtB_y) -1 ), 1/priorgammavar)

  U = chol ( tempmat)
  IR = backsolve (U , diag ( ncol(BtB_y) ))
  # btilde = crossprod ( t ( IR ))%*%( crossprod (X_node , r_node ) )
  # beta_hat = btilde + sqrt ( sigma2 )* IR %*% rnorm ( p )
  tempsj <- (1/sigma2)*S_j
  tempsj[length(tempsj)] <- tempsj[length(tempsj)] + gamma0/priorgammavar
  tSjtempmatinvSj = crossprod ( crossprod( IR , tempsj ) )

  log_post <- 0.5*(- (ncol(BtB_y)-1)*log(sigma2_mu) + #determinant(tempmat, logarithm = TRUE) +
                     tSjtempmatinvSj) - sum(log(diag(U)))



  # tempmat <- BtB_y
  # diag(tempmat) <- diag(tempmat) + sigma2*c(rep(1/sigma2_mu, ncol(BtB_y) -1 ), 1/priorgammavar)
  #
  # U = chol ( tempmat)
  # IR = backsolve (U , diag ( ncol(BtB_z_u) ))
  # # btilde = crossprod ( t ( IR ))%*%( crossprod (X_node , r_node ) )
  # # beta_hat = btilde + sqrt ( sigma2 )* IR %*% rnorm ( p )
  # tSjtempmatinvSj = crossprod ( crossprod( IR , S_j ) )
  #
  # log_post <- 0.5*(- ncol(BtB_y)*(log(sigma2_mu) +log(sigma2) ) + #determinant(tempmat, logarithm = TRUE) +
  #                    (1/sigma2)*tSjtempmatinvSj) - sum(log(diag(U)))



  # Now calculate the log posterior
  # log_post = 0.5 * ( sum(log( sigma2 / (nj*sigma2_mu + sigma2))) +
  #                      sum( (sigma2_mu* S_j^2) / (sigma2 * (nj*sigma2_mu + sigma2))))
  return(log_post)
}




tree_full_conditional_z_marg_savechol = function(tree, R, #sigma2,
                                                 sigma2_mu,
                                        weightstemp,
                                        weightz,
                                        binmat_all_z, cens_inds, uncens_inds, BtB_z_u, BtB_z_c) {

  # Function to compute log full conditional distirbution for an individual tree
  # R is a vector of partial residuals

  # Need to calculate log complete conditional, involves a sum over terminal nodes

  # # First find which rows are terminal nodes
  # which_terminal = which(tree$tree_matrix[,'terminal'] == 1)
  #
  # # Get node sizes for each terminal node
  # nj = tree$tree_matrix[which_terminal,'node_size']
  # # nj = tree$tree_matrix[tree$tree_matrix[,'terminal'] == 1,'node_size']
  #
  # # nj <- nj[nj!=0]
  # # Get sum of residuals and sum of residuals squared within each terminal node
  # # sumRsq_j = aggregate(R, by = list(tree$node_indices), function(x) sum(x^2))[,2]
  # # S_j = aggregate(R, by = list(tree$node_indices), sum)[,2]
  #
  # # sumRsq_j = fsum(R^2, tree$node_indices)
  # # S_j = fsum(R, tree$node_indices)
  # # S_j = fsum(R,factor(tree$node_indices, levels = which_terminal ), fill = TRUE)
  #
  # # S_j = fsum(R,factor(tree$node_indices, levels = which_terminal ), w = weightstemp, fill = TRUE)
  # S_j = fsum(R*weightstemp,factor(tree$node_indices, levels = which_terminal ), fill = TRUE)

  # check if this is faster then binary matrix approach. Can avoid creating the binary matrix entirely
  # S_j <- rep(0,0) # initialize as numeric vector of length 0
  # for(i in 1:length(trees)){
  #   which_terminal = which(trees[[i]]$tree_matrix[,'terminal'] == 1)
  #   nj = trees[[i]]$tree_matrix[which_terminal,'node_size']
  #   S_j = c(S_j,fsum(R*weightstemp,factor(trees[[i]]$node_indices, levels = which_terminal ), fill = TRUE))
  # }


  S_j = crossprod(binmat_all_z, R*weightstemp)

  tempmat <- BtB_z_u*weightz + BtB_z_c + (1/sigma2_mu)*diag(ncol(BtB_z_u))

  # log_post <- 0.5*(- ncol(BtB_z_u)*log(sigma2_mu) + determinant(tempmat, logarithm = TRUE) +
  #                    t(S_j)%*%solve(tempmat)%*% S_j)

  U = chol ( tempmat#, pivot = TRUE, tol = 0.0001
             )
  IR = backsolve (U , diag ( ncol(BtB_z_u) ))
  # btilde = crossprod ( t ( IR ))%*%( crossprod (X_node , r_node ) )
  # beta_hat = btilde + sqrt ( sigma2 )* IR %*% rnorm ( p )
  tSjtempmatinvSj = crossprod ( crossprod( IR , S_j ) )

  # print("dim(tSjtempmatinvSj) = ")
  # print(dim(tSjtempmatinvSj))

  log_post <- 0.5*(- ncol(BtB_z_u)*log(sigma2_mu) + #determinant(tempmat, logarithm = TRUE) +
                     tSjtempmatinvSj) - sum(log(diag(U)))

  ret_list <- list()
  ret_list[[1]] <- log_post
  ret_list[[2]] <- IR
  ret_list[[3]] <- S_j


  # Now calculate the log posterior
  # log_post = 0.5 * ( sum(log( sigma2 / (nj*sigma2_mu + sigma2))) +
  #                      sum( (sigma2_mu* S_j^2) / (sigma2 * (nj*sigma2_mu + sigma2))))
  return(ret_list)
}


tree_full_conditional_z_marg_lin = function(tree, R, #sigma2,
                                            sigma2_mu,
                                            weightstemp,
                                            weightz,
                                            binmat_all_z, cens_inds, uncens_inds, BtB_z_u, BtB_z_c,
                                            wmat_train, Amean_p, invAvar_p) {


  WBmat <- cbind(wmat_train, binmat_all_z)

  S_j = crossprod(WBmat, R*weightstemp)

  tempmat <- BtB_z_u*weightz + BtB_z_c + (1/sigma2_mu)*diag(ncol(BtB_z_u))

  offdiagblock <- crossprod(wmat_train[uncens_inds,,drop = FALSE], binmat_all_z[uncens_inds,,drop = FALSE])*weightz +
    crossprod(wmat_train[cens_inds,,drop = FALSE], binmat_all_z[cens_inds,,drop = FALSE])

  tempmat <- rbind(cbind(crossprod(wmat_train[uncens_inds,,drop = FALSE])*weightz +
                           crossprod(wmat_train[cens_inds,,drop = FALSE]) + invAvar_p ,
                         offdiagblock),
                   cbind(t(offdiagblock), tempmat))


  U = chol ( tempmat#, pivot = TRUE, tol = 0.0001
  )
  IR = backsolve (U , diag ( ncol(BtB_z_u) + ncol(wmat_train) ))
  # btilde = crossprod ( t ( IR ))%*%( crossprod (X_node , r_node ) )
  # beta_hat = btilde + sqrt ( sigma2 )* IR %*% rnorm ( p )
  tSjtempmatinvSj = crossprod ( crossprod( IR , S_j + c(invAvar_p %*% Amean_p, rep(0, ncol(BtB_z_u))) ) )


  log_post <- 0.5*(- ncol(BtB_z_u)*log(sigma2_mu) + #determinant(tempmat, logarithm = TRUE) +
                     tSjtempmatinvSj) - sum(log(diag(U)))

  # ret_list <- list()
  # ret_list[[1]] <- log_post
  # ret_list[[2]] <- IR
  # ret_list[[3]] <- S_j
  #

  # Now calculate the log posterior
  # log_post = 0.5 * ( sum(log( sigma2 / (nj*sigma2_mu + sigma2))) +
  #                      sum( (sigma2_mu* S_j^2) / (sigma2 * (nj*sigma2_mu + sigma2))))
  return(log_post)
}


tree_full_conditional_z_marg_lin_savechol = function(tree, R, #sigma2,
                                                 sigma2_mu,
                                                 weightstemp,
                                                 weightz,
                                                 binmat_all_z, cens_inds, uncens_inds, BtB_z_u, BtB_z_c,
                                            wmat_train, Amean_p, invAvar_p) {


  WBmat <- cbind(wmat_train, binmat_all_z)

  S_j = crossprod(WBmat, R*weightstemp)

  tempmat <- BtB_z_u*weightz + BtB_z_c + (1/sigma2_mu)*diag(ncol(BtB_z_u))

  offdiagblock <- crossprod(wmat_train[uncens_inds,,drop = FALSE], binmat_all_z[uncens_inds,,drop = FALSE])*weightz +
    crossprod(wmat_train[cens_inds,,drop = FALSE], binmat_all_z[cens_inds,,drop = FALSE])

  tempmat <- rbind(cbind(crossprod(wmat_train[uncens_inds,,drop = FALSE])*weightz +
          crossprod(wmat_train[cens_inds,,drop = FALSE]) + invAvar_p ,
          offdiagblock),
        cbind(t(offdiagblock), tempmat))


  U = chol ( tempmat#, pivot = TRUE, tol = 0.0001
  )
  IR = backsolve (U , diag ( ncol(BtB_z_u) + ncol(wmat_train) ))
  # btilde = crossprod ( t ( IR ))%*%( crossprod (X_node , r_node ) )
  # beta_hat = btilde + sqrt ( sigma2 )* IR %*% rnorm ( p )
  tSjtempmatinvSj = crossprod ( crossprod( IR , S_j + c(invAvar_p %*% Amean_p, rep(0, ncol(BtB_z_u))) ) )


  log_post <- 0.5*(- ncol(BtB_z_u)*log(sigma2_mu) + #determinant(tempmat, logarithm = TRUE) +
                     tSjtempmatinvSj) - sum(log(diag(U)))

  ret_list <- list()
  ret_list[[1]] <- log_post
  ret_list[[2]] <- IR
  ret_list[[3]] <- S_j


  # Now calculate the log posterior
  # log_post = 0.5 * ( sum(log( sigma2 / (nj*sigma2_mu + sigma2))) +
  #                      sum( (sigma2_mu* S_j^2) / (sigma2 * (nj*sigma2_mu + sigma2))))
  return(ret_list)
}



tree_full_conditional_y_marg_nogamma_savechol = function(tree, R, #sigma2,
                                                         phi,
                                                         sigma2_mu,
                                                 binmat_all_y, BtB_y) {

  # Function to compute log full conditional distirbution for an individual tree
  # R is a vector of partial residuals

  # print("line 474")
  S_j = crossprod(binmat_all_y, R)

  tempmat <- BtB_y*(1/phi) + (1/sigma2_mu)*diag(ncol(BtB_y))
  # print("line 476")

  U = chol ( tempmat#, pivot = TRUE, tol = 0.0001
  )
  IR = backsolve (U , diag ( ncol(BtB_y) ))
  # btilde = crossprod ( t ( IR ))%*%( crossprod (X_node , r_node ) )
  # beta_hat = btilde + sqrt ( sigma2 )* IR %*% rnorm ( p )

  # print("line 486")

  tSjtempmatinvSj = crossprod ( crossprod( IR , S_j*(1/phi) ) )

  # print("dim(tSjtempmatinvSj) = ")
  # print(dim(tSjtempmatinvSj))
  # print("line 492")

  log_post <- 0.5*(- ncol(BtB_y)*log(sigma2_mu) + #determinant(tempmat, logarithm = TRUE) +
                     tSjtempmatinvSj) - sum(log(diag(U)))

  ret_list <- list()
  ret_list[[1]] <- log_post
  ret_list[[2]] <- IR
  ret_list[[3]] <- S_j


  # Now calculate the log posterior
  # log_post = 0.5 * ( sum(log( sigma2 / (nj*sigma2_mu + sigma2))) +
  #                      sum( (sigma2_mu* S_j^2) / (sigma2 * (nj*sigma2_mu + sigma2))))
  return(ret_list)
}


tree_full_conditional_y_marg_savechol = function(trees, R, sigma2, priorgammavar, sigma2_mu, binmat_all_y_z, BtB_y,
                                                 gamma0) {

  # Function to compute log full conditional distirbution for an individual tree
  # R is a vector of partial residuals

  # Need to calculate log complete conditional, involves a sum over terminal nodes

  # # First find which rows are terminal nodes
  # which_terminal = which(tree$tree_matrix[,'terminal'] == 1)
  #
  # # Get node sizes for each terminal node
  # nj = tree$tree_matrix[which_terminal,'node_size']
  # # nj = tree$tree_matrix[tree$tree_matrix[,'terminal'] == 1,'node_size']
  #
  # # nj <- nj[nj!=0]
  # # Get sum of residuals and sum of residuals squared within each terminal node
  # # sumRsq_j = aggregate(R, by = list(tree$node_indices), function(x) sum(x^2))[,2]
  # # S_j = aggregate(R, by = list(tree$node_indices), sum)[,2]
  #
  # # sumRsq_j = fsum(R^2, tree$node_indices)
  # # S_j = fsum(R, tree$node_indices)
  # S_j = fsum(R,factor(tree$node_indices, levels = which_terminal ), fill = TRUE)

  # S_j <- rep(0,0) # initialize as numeric vector of length 0
  # for(i in 1:length(trees)){
  #   which_terminal = which(trees[[i]]$tree_matrix[,'terminal'] == 1)
  #   nj = trees[[i]]$tree_matrix[which_terminal,'node_size']
  #   S_j = c(S_j,fsum(R,factor(trees[[i]]$node_indices, levels = which_terminal ), fill = TRUE))
  # }
  # S_j = c(S_j, crossprod(binmat_all_y_z[,ncol(binmat_all_y_z)], R) )


  # print("dim(binmat_all_y_z) = ")
  # print(dim(binmat_all_y_z))
  #
  # print("length(R) = ")
  # print(length(R))

  S_j = crossprod(binmat_all_y_z, R)


  # print("dim(S_j) = ")
  # print(dim(S_j))

  # for(j in 1:ncol(BtB_y)){
  #   if(any(BtB_y[,j]!= t(BtB_y)[,j])){
  #     print("BtB_y[,j] = ")
  #     print(BtB_y[,j])
  #     print("t(BtB_y)[,j] = ")
  #     print(t(BtB_y)[,j])
  #     print("j = ")
  #     print(j )
  #     stop("any(BtB_y[,j]!= t(BtB_y)[,j])")
  #   }
  #
  # }

  # if(!(isSymmetric(BtB_y, check.attributes = FALSE))){
  #
  #   print("(BtB_y[,ncol(BtB_y)] = ")
  #   print((BtB_y[,ncol(BtB_y)]))
  #   print("(t(BtB_y)[,ncol(BtB_y)] = ")
  #   print((t(BtB_y)[,ncol(BtB_y)]))
  #
  #   print("dim(BtB_y) = ")
  #   print(dim(BtB_y))
  #   print("(BtB_y) = ")
  #   print((BtB_y))
  #
  #   print("which((BtB_y)!= t(BtB_y), arr.ind = TRUE) = ")
  #   print(which((BtB_y)!= t(BtB_y), arr.ind = TRUE))
  #
  #   tempmissinginds <- which((BtB_y)!= t(BtB_y), arr.ind = TRUE)
  #
  #   print("BtB_y[tempmissinginds] = ")
  #   print(BtB_y[tempmissinginds])
  #
  #   print("t(BtB_y)[tempmissinginds] = ")
  #   print(t(BtB_y)[tempmissinginds])
  #
  #   stop("BtB_y not symmetric")
  #
  # }

  # if(max(abs(crossprod(binmat_all_y_z)- BtB_y)) >0.001 ){
  #   print("(max(abs(crossprod(binmat_all_y_z)- BtB_y))) = ")
  #   print((max(abs(crossprod(binmat_all_y_z)- BtB_y))))
  #
  #   print("dim(BtB_y) = ")
  #   print(dim(BtB_y))
  #   print("(BtB_y) = ")
  #   print((BtB_y))
  #   print("(crossprod(binmat_all_y_z)) = ")
  #   print((crossprod(binmat_all_y_z)))
  #   print("(crossprod(binmat_all_y_z)- BtB_y) = ")
  #   print((crossprod(binmat_all_y_z)- BtB_y))
  #
  #   tempmax <- max(abs(crossprod(binmat_all_y_z)- BtB_y))
  #   maxind <- which(abs(crossprod(binmat_all_y_z)- BtB_y) ==tempmax, arr.ind = TRUE)
  #   print("maxind = ")
  #   print(maxind)
  #
  #   print("(crossprod(binmat_all_y_z)- BtB_y)[maxind[,1],] = ")
  #   print((crossprod(binmat_all_y_z)- BtB_y)[maxind[,1],])
  #
  #   print("(crossprod(binmat_all_y_z)- BtB_y)[,maxind[,2]] = ")
  #   print((crossprod(binmat_all_y_z)- BtB_y)[,maxind[,2]])
  #
  #   print("which.max(abs(crossprod(binmat_all_y_z)- BtB_y)) = ")
  #   print(which.max(abs(crossprod(binmat_all_y_z)- BtB_y)))
  #
  #   stop("max(abs(crossprod(binmat_all_y_z)- BtB_y)) >0.001 ")
  # }

  # there are two ways of calculating the marginal probability
  tempmat <- (1/sigma2)*BtB_y
  diag(tempmat) <- diag(tempmat) + c(rep(1/sigma2_mu, ncol(BtB_y) -1 ), 1/priorgammavar)

  # print("dim(tempmat) = ")
  # print(dim(tempmat))

  # print("1/(sigma2) = ")
  # print(1/(sigma2))
  # print("1/(priorgammavar) = ")
  # print(1/(priorgammavar))
  # print("1/(sigma2_mu) = ")
  # print(1/(sigma2_mu))

  U = chol ( tempmat#, pivot = TRUE, tol = 0.0001
             )
  IR = backsolve (U , diag ( ncol(BtB_y) ))
  # btilde = crossprod ( t ( IR ))%*%( crossprod (X_node , r_node ) )
  # beta_hat = btilde + sqrt ( sigma2 )* IR %*% rnorm ( p )
  tempsj <- (1/sigma2)*S_j
  tempsj[length(tempsj)] <- tempsj[length(tempsj)] + gamma0/priorgammavar
  tSjtempmatinvSj = crossprod ( crossprod( IR , tempsj ) )

  # print("dim(tSjtempmatinvSj) = ")
  # print(dim(tSjtempmatinvSj))



  log_post <- 0.5*(- (ncol(BtB_y)-1)*log(sigma2_mu) + #determinant(tempmat, logarithm = TRUE) +
                     tSjtempmatinvSj) - sum(log(diag(U)))



  # tempmat <- BtB_y
  # diag(tempmat) <- diag(tempmat) + sigma2*c(rep(1/sigma2_mu, ncol(BtB_y) -1 ), 1/priorgammavar)
  #
  # U = chol ( tempmat)
  # IR = backsolve (U , diag ( ncol(BtB_z_u) ))
  # # btilde = crossprod ( t ( IR ))%*%( crossprod (X_node , r_node ) )
  # # beta_hat = btilde + sqrt ( sigma2 )* IR %*% rnorm ( p )
  # tSjtempmatinvSj = crossprod ( crossprod( IR , S_j ) )
  #
  # log_post <- 0.5*(- ncol(BtB_y)*(log(sigma2_mu) +log(sigma2) ) + #determinant(tempmat, logarithm = TRUE) +
  #                    (1/sigma2)*tSjtempmatinvSj) - sum(log(diag(U)))

  ret_list <- list()
  ret_list[[1]] <- log_post
  ret_list[[2]] <- IR
  ret_list[[3]] <- S_j



  # Now calculate the log posterior
  # log_post = 0.5 * ( sum(log( sigma2 / (nj*sigma2_mu + sigma2))) +
  #                      sum( (sigma2_mu* S_j^2) / (sigma2 * (nj*sigma2_mu + sigma2))))
  return(ret_list)
}




tree_full_conditional_y_marg_nogamma_lin = function(tree, R, #sigma2,
                                                   phi,
                                                   sigma2_mu,
                                                   binmat_all_y, BtB_y,
                                                   Bmean_p, invBvar_p, Xmat_train) {

  # Function to compute log full conditional distirbution for an individual tree
  # R is a vector of partial residuals
  XBmat <- cbind(Xmat_train, binmat_all_y)

  # print("line 474")
  S_j = crossprod(XBmat, R)

  tempmat <- BtB_y*(1/phi) + (1/sigma2_mu)*diag(ncol(BtB_y))
  # print("line 476")

  offdiagblock <- crossprod(Xmat_train, binmat_all_y)*(1/phi)

  # print("dim(crossprod(Xmat_train)) = " )
  # print(dim(crossprod(Xmat_train)))
  # print("dim(invBvar_p) = " )
  # print(dim(invBvar_p))
  #
  # print("(crossprod(Xmat_train)) = " )
  # print((crossprod(Xmat_train)))
  # print("(invBvar_p) = " )
  # print((invBvar_p))
  #
  # print("solve(tempmat) = " )
  # print(solve(tempmat))

  tempmat <- rbind(cbind(crossprod(Xmat_train)*(1/phi) + invBvar_p ,
                         offdiagblock),
                   cbind(t(offdiagblock), tempmat))


  # print("(tempmat) = " )
  # print((tempmat))
  #
  #
  # print("solve(crossprod(Xmat_train) + invBvar_p) = " )
  # print(solve(crossprod(Xmat_train) + invBvar_p))


  U = chol ( tempmat#, pivot = TRUE, tol = 0.0001
  )
  IR = backsolve (U , diag ( ncol(XBmat) ))
  # btilde = crossprod ( t ( IR ))%*%( crossprod (X_node , r_node ) )
  # beta_hat = btilde + sqrt ( sigma2 )* IR %*% rnorm ( p )

  # print("line 486")

  tSjtempmatinvSj = crossprod ( crossprod( IR , S_j*(1/phi) ) )

  # print("dim(tSjtempmatinvSj) = ")
  # print(dim(tSjtempmatinvSj))
  # print("line 492")

  log_post <- 0.5*(- ncol(BtB_y)*log(sigma2_mu) + #determinant(tempmat, logarithm = TRUE) +
                     tSjtempmatinvSj) - sum(log(diag(U)))

  # ret_list <- list()
  # ret_list[[1]] <- log_post
  # ret_list[[2]] <- IR
  # ret_list[[3]] <- S_j
  #

  # Now calculate the log posterior
  # log_post = 0.5 * ( sum(log( sigma2 / (nj*sigma2_mu + sigma2))) +
  #                      sum( (sigma2_mu* S_j^2) / (sigma2 * (nj*sigma2_mu + sigma2))))
  return(log_post)
}


tree_full_conditional_y_marg_lin = function(trees, R, sigma2, priorgammavar, sigma2_mu, binmat_all_y_z, BtB_y,
                                                     Bmean_p, invBvar_p, Xmat_train, gamma0) {

  # Function to compute log full conditional distirbution for an individual tree
  # R is a vector of partial residuals

  # Need to calculate log complete conditional, involves a sum over terminal nodes
  XBmat <- cbind(Xmat_train, binmat_all_y_z)

  S_j = crossprod(XBmat, R)

  # there are two ways of calculating the marginal probability
  tempmat <- (1/sigma2)*BtB_y
  diag(tempmat) <- diag(tempmat) + c(rep(1/sigma2_mu, ncol(BtB_y) -1 ), 1/priorgammavar)

  offdiagblock <- crossprod(Xmat_train, binmat_all_y_z)*(1/sigma2)

  # print("dim(crossprod(Xmat_train)) = " )
  # print(dim(crossprod(Xmat_train)))
  # print("dim(invBvar_p) = " )
  # print(dim(invBvar_p))

  tempmat <- rbind(cbind(crossprod(Xmat_train)*(1/sigma2) + invBvar_p ,
                         offdiagblock),
                   cbind(t(offdiagblock), tempmat))
  U = chol ( tempmat#, pivot = TRUE, tol = 0.0001
  )
  IR = backsolve (U , diag ( ncol(XBmat) ))
  # btilde = crossprod ( t ( IR ))%*%( crossprod (X_node , r_node ) )
  # beta_hat = btilde + sqrt ( sigma2 )* IR %*% rnorm ( p )

  tempsj <- (1/sigma2)*S_j
  tempsj[1:ncol(Xmat_train)] <-  tempsj[1:ncol(Xmat_train)] + invBvar_p %*% Bmean_p
  tempsj[length(tempsj)] <- tempsj[length(tempsj)] + gamma0/priorgammavar
  tSjtempmatinvSj = crossprod ( crossprod( IR , tempsj) )

  log_post <- 0.5*(- (ncol(BtB_y)-1)*log(sigma2_mu) + #determinant(tempmat, logarithm = TRUE) +
                     tSjtempmatinvSj) - sum(log(diag(U)))


  # Now calculate the log posterior
  # log_post = 0.5 * ( sum(log( sigma2 / (nj*sigma2_mu + sigma2))) +
  #                      sum( (sigma2_mu* S_j^2) / (sigma2 * (nj*sigma2_mu + sigma2))))
  return(log_post)
}


tree_full_conditional_y_marg_nogamma_savechol_lin = function(tree, R, #sigma2,
                                                         phi,
                                                         sigma2_mu,
                                                         binmat_all_y, BtB_y,
                                                         Bmean_p, invBvar_p, Xmat_train) {

  # Function to compute log full conditional distirbution for an individual tree
  # R is a vector of partial residuals
  XBmat <- cbind(Xmat_train, binmat_all_y)

  # print("line 474")
  S_j = crossprod(XBmat, R)

  tempmat <- BtB_y*(1/phi) + (1/sigma2_mu)*diag(ncol(BtB_y))
  # print("line 476")

  offdiagblock <- crossprod(Xmat_train, binmat_all_y)*(1/phi)

  # print("dim(crossprod(Xmat_train)) = " )
  # print(dim(crossprod(Xmat_train)))
  # print("dim(invBvar_p) = " )
  # print(dim(invBvar_p))
  #
  # print("(crossprod(Xmat_train)) = " )
  # print((crossprod(Xmat_train)))
  # print("(invBvar_p) = " )
  # print((invBvar_p))
  #
  # print("solve(tempmat) = " )
  # print(solve(tempmat))

  tempmat <- rbind(cbind(crossprod(Xmat_train)*(1/phi) + invBvar_p ,
                         offdiagblock),
                   cbind(t(offdiagblock), tempmat))


  # print("(tempmat) = " )
  # print((tempmat))
  #
  #
  # print("solve(crossprod(Xmat_train) + invBvar_p) = " )
  # print(solve(crossprod(Xmat_train) + invBvar_p))


  U = chol ( tempmat#, pivot = TRUE, tol = 0.0001
  )
  IR = backsolve (U , diag ( ncol(XBmat) ))
  # btilde = crossprod ( t ( IR ))%*%( crossprod (X_node , r_node ) )
  # beta_hat = btilde + sqrt ( sigma2 )* IR %*% rnorm ( p )

  # print("line 486")

  tSjtempmatinvSj = crossprod ( crossprod( IR , S_j*(1/phi) ) )

  # print("dim(tSjtempmatinvSj) = ")
  # print(dim(tSjtempmatinvSj))
  # print("line 492")

  log_post <- 0.5*(- ncol(BtB_y)*log(sigma2_mu) + #determinant(tempmat, logarithm = TRUE) +
                     tSjtempmatinvSj) - sum(log(diag(U)))

  ret_list <- list()
  ret_list[[1]] <- log_post
  ret_list[[2]] <- IR
  ret_list[[3]] <- S_j


  # Now calculate the log posterior
  # log_post = 0.5 * ( sum(log( sigma2 / (nj*sigma2_mu + sigma2))) +
  #                      sum( (sigma2_mu* S_j^2) / (sigma2 * (nj*sigma2_mu + sigma2))))
  return(ret_list)
}


tree_full_conditional_y_marg_savechol_lin = function(trees, R, sigma2, priorgammavar, sigma2_mu, binmat_all_y_z, BtB_y,
                                                     Bmean_p, invBvar_p, Xmat_train, gamma0) {

  # Function to compute log full conditional distirbution for an individual tree
  # R is a vector of partial residuals

  # Need to calculate log complete conditional, involves a sum over terminal nodes
  XBmat <- cbind(Xmat_train, binmat_all_y_z)

  S_j = crossprod(XBmat, R)

  # there are two ways of calculating the marginal probability
  tempmat <- (1/sigma2)*BtB_y
  diag(tempmat) <- diag(tempmat) + c(rep(1/sigma2_mu, ncol(BtB_y) -1 ), 1/priorgammavar)

  offdiagblock <- crossprod(Xmat_train, binmat_all_y_z)*(1/sigma2)

  # print("dim(crossprod(Xmat_train)) = " )
  # print(dim(crossprod(Xmat_train)))
  # print("dim(invBvar_p) = " )
  # print(dim(invBvar_p))

  tempmat <- rbind(cbind(crossprod(Xmat_train)*(1/sigma2) + invBvar_p ,
                         offdiagblock),
                   cbind(t(offdiagblock), tempmat))
  U = chol ( tempmat#, pivot = TRUE, tol = 0.0001
  )
  IR = backsolve (U , diag ( ncol(XBmat) ))
  # btilde = crossprod ( t ( IR ))%*%( crossprod (X_node , r_node ) )
  # beta_hat = btilde + sqrt ( sigma2 )* IR %*% rnorm ( p )

  tempsj <- (1/sigma2)*S_j
  tempsj[1:ncol(Xmat_train)] <-  tempsj[1:ncol(Xmat_train)] + invBvar_p %*% Bmean_p
  tempsj[length(tempsj)] <- tempsj[length(tempsj)] + gamma0/priorgammavar
  tSjtempmatinvSj = crossprod ( crossprod( IR , tempsj) )

  log_post <- 0.5*(- (ncol(BtB_y)-1)*log(sigma2_mu) + #determinant(tempmat, logarithm = TRUE) +
                     tSjtempmatinvSj) - sum(log(diag(U)))

  ret_list <- list()
  ret_list[[1]] <- log_post
  ret_list[[2]] <- IR
  ret_list[[3]] <- S_j

  # Now calculate the log posterior
  # log_post = 0.5 * ( sum(log( sigma2 / (nj*sigma2_mu + sigma2))) +
  #                      sum( (sigma2_mu* S_j^2) / (sigma2 * (nj*sigma2_mu + sigma2))))
  return(ret_list)
}


# Simulate_par -------------------------------------------------------------

simulate_mu = function(tree, R, sigma2, sigma2_mu) {

  # Simulate mu values for a given tree

  # First find which rows are terminal nodes
  which_terminal = which(tree$tree_matrix[,'terminal'] == 1)

  # Get node sizes for each terminal node
  nj = tree$tree_matrix[which_terminal,'node_size']
  # nj <- nj[nj!=0]
  # Get sum of residuals in each terminal node
  # sumR = aggregate(R, by = list(tree$node_indices), sum)[,2]

  # sumR = fsum(R, tree$node_indices)

  sumR = fsum(R,factor(tree$node_indices, levels = which_terminal ), fill = TRUE)




  # Now calculate mu values
  mu = rnorm(length(nj) ,
             mean = (sumR / sigma2) / (nj/sigma2 + 1/sigma2_mu),
             sd = sqrt(1/(nj/sigma2 + 1/sigma2_mu)))

  # Wipe all the old mus out for other nodes
  tree$tree_matrix[,'mu'] = NA

  if(any(is.na(mu))){

    print("(sumR / sigma2) / (nj/sigma2 + 1/sigma2_mu) = ")
    print((sumR / sigma2) / (nj/sigma2 + 1/sigma2_mu))
    stop("NA in mu vector")

  }

  # Put in just the ones that are useful
  tree$tree_matrix[which_terminal,'mu'] = mu

  return(tree)
}


simulate_mu_weighted = function(tree, R, # sigma2,
                                sigma2_mu, weight_vec) {

  # Simulate mu values for a given tree

  # First find which rows are terminal nodes
  which_terminal = which(tree$tree_matrix[,'terminal'] == 1)

  # Get node sizes for each terminal node
  nj = tree$tree_matrix[which_terminal,'node_size']
  # nj <- nj[nj!=0]
  # Get sum of residuals in each terminal node
  # sumR = aggregate(R, by = list(tree$node_indices), sum)[,2]

  # sumR = fsum(R, tree$node_indices)

  # sumR = fsum(R,factor(tree$node_indices, levels = which_terminal ), fill = TRUE)
  sumRw = fsum(R*weight_vec,factor(tree$node_indices, levels = which_terminal ), fill = TRUE)
  sumw = fsum(weight_vec,factor(tree$node_indices, levels = which_terminal ), fill = TRUE)

  sigtilde_j <- 1/( (1/sigma2_mu)  +  sumw  )

  # # Now calculate mu values
  # mu = rnorm(length(nj) ,
  #            mean = (sumR / sigma2) / (nj/sigma2 + 1/sigma2_mu),
  #            sd = sqrt(1/(nj/sigma2 + 1/sigma2_mu)))

  mu = rnorm(length(nj) ,
             mean = sigtilde_j*sumRw ,
             sd = sqrt(sigtilde_j))

  # Wipe all the old mus out for other nodes
  tree$tree_matrix[,'mu'] = NA

  if(any(is.na(mu))){

    print("(sumR / sigma2) / (nj/sigma2 + 1/sigma2_mu) = ")
    print((sumR / sigma2) / (nj/sigma2 + 1/sigma2_mu))
    stop("NA in mu vector")

  }

  # Put in just the ones that are useful
  tree$tree_matrix[which_terminal,'mu'] = mu

  return(tree)
}


simulate_mu_all_y = function(trees, R, sigma2,
                             priorgammavar,
                             sigma2_mu_y, binmat_all_y_z, BztBz_y, firstcolindtrees_y, gamma0) {


  # S_j <- rep(0,0) # initialize as numeric vector of length 0
  # for(i in 1:length(trees)){
  #   which_terminal = which(trees[[i]]$tree_matrix[,'terminal'] == 1)
  #   nj = trees[[i]]$tree_matrix[which_terminal,'node_size']
  #   S_j = c(S_j,fsum(R,factor(trees[[i]]$node_indices, levels = which_terminal ), fill = TRUE))
  # }
  # S_j = c(S_j, crossprod(binmat_all_y_z[,ncol(binmat_all_y_z)], R) )
  S_j = crossprod(binmat_all_y_z, R)


  # there are two ways of calculating the marginal probability
  tempmat <- (1/sigma2)*BztBz_y
  diag(tempmat) <- diag(tempmat) + c(rep(1/sigma2_mu_y, ncol(BztBz_y) -1 ), 1/priorgammavar)

  U = chol ( tempmat#, pivot = TRUE, tol = 0.0001
             )
  IR = backsolve (U , diag ( ncol(BztBz_y) ))
  # btilde = crossprod ( t ( IR ))%*%( crossprod (X_node , r_node ) )
  # beta_hat = btilde + sqrt ( sigma2 )* IR %*% rnorm ( p )
  # tSjtempmatinvSj = crossprod ( crossprod( IR , (1/sigma2)*S_j ) )

  # tSjtempmatinvSj = crossprod ( crossprod( IR , (1/sigma2)*S_j ) )
  tempsj <- (1/sigma2)*S_j
  tempsj[length(tempsj)] <- tempsj[length(tempsj)] + gamma0/priorgammavar
  mugamma_mean = tcrossprod (  IR , crossprod( tempsj, IR ))
  mugamma_sample = mugamma_mean + IR %*% rnorm ( ncol(BztBz_y) )

  # perhaps it would be more efficient to store one mu vector throughout all the code.
  for(i in 1:length(trees)){
    which_terminal = which(trees[[i]]$tree_matrix[,'terminal'] == 1)
    trees[[i]]$tree_matrix[,'mu'] <- NA
    if(i < length(trees)){
      trees[[i]]$tree_matrix[which_terminal,'mu'] = mugamma_mean[firstcolindtrees_y[i]:(firstcolindtrees_y[i+1]-1)]
    }else{
      trees[[i]]$tree_matrix[which_terminal,'mu'] = mugamma_mean[firstcolindtrees_y[i]:(ncol(BztBz_y)-1)]
    }
  }

  ret_list <- list()
  ret_list[[1]] <- trees
  ret_list[[2]] <- mugamma_mean[1:(ncol(BztBz_y)-1)]
  ret_list[[3]] <- mugamma_sample[1:(ncol(BztBz_y)-1)]
  ret_list[[4]] <- mugamma_mean[ncol(BztBz_y)]
  ret_list[[5]] <- mugamma_sample[ncol(BztBz_y)]

  return(ret_list)
}




simulate_mu_all_y_fast = function(trees, R, sigma2,
                             priorgammavar,
                             sigma2_mu_y, binmat_all_y_z, BztBz_y, firstcolindtrees_y,
                             IR, S_j, gamma0 ) {


  # # S_j <- rep(0,0) # initialize as numeric vector of length 0
  # # for(i in 1:length(trees)){
  # #   which_terminal = which(trees[[i]]$tree_matrix[,'terminal'] == 1)
  # #   nj = trees[[i]]$tree_matrix[which_terminal,'node_size']
  # #   S_j = c(S_j,fsum(R,factor(trees[[i]]$node_indices, levels = which_terminal ), fill = TRUE))
  # # }
  # # S_j = c(S_j, crossprod(binmat_all_y_z[,ncol(binmat_all_y_z)], R) )
  # S_j = crossprod(binmat_all_y_z, R)
  #
  #
  # # there are two ways of calculating the marginal probability
  # tempmat <- (1/sigma2)*BztBz_y
  # diag(tempmat) <- diag(tempmat) + c(rep(1/sigma2_mu, ncol(BztBz_y) -1 ), 1/priorgammavar)
  #
  # U = chol ( tempmat)
  # IR = backsolve (U , diag ( ncol(BztBz_y) ))
  # # btilde = crossprod ( t ( IR ))%*%( crossprod (X_node , r_node ) )
  # # beta_hat = btilde + sqrt ( sigma2 )* IR %*% rnorm ( p )
  # # tSjtempmatinvSj = crossprod ( crossprod( IR , (1/sigma2)*S_j ) )
  #
  # # tSjtempmatinvSj = crossprod ( crossprod( IR , (1/sigma2)*S_j ) )
  tempsj <- (1/sigma2)*S_j
  tempsj[length(tempsj)] <- tempsj[length(tempsj)] + gamma0/priorgammavar
  mugamma_mean = tcrossprod (  IR , crossprod( tempsj, IR ))
  mugamma_sample = mugamma_mean + IR %*% rnorm ( ncol(BztBz_y) )

  # perhaps it would be more efficient to store one mu vector throughout all the code.
  for(i in 1:length(trees)){
    which_terminal = which(trees[[i]]$tree_matrix[,'terminal'] == 1)
    trees[[i]]$tree_matrix[,'mu'] <- NA
    if(i < length(trees)){
      trees[[i]]$tree_matrix[which_terminal,'mu'] = mugamma_sample[firstcolindtrees_y[i]:(firstcolindtrees_y[i+1]-1)]
    }else{
      trees[[i]]$tree_matrix[which_terminal,'mu'] = mugamma_sample[firstcolindtrees_y[i]:(ncol(BztBz_y)-1)]
    }
  }

  ret_list <- list()
  ret_list[[1]] <- trees
  ret_list[[2]] <- mugamma_mean[1:(ncol(BztBz_y)-1)]
  ret_list[[3]] <- mugamma_sample[1:(ncol(BztBz_y)-1)]
  ret_list[[4]] <- mugamma_mean[ncol(BztBz_y)]
  ret_list[[5]] <- mugamma_sample[ncol(BztBz_y)]

  return(ret_list)
}




simulate_mu_all_y_lin = function(trees, R, sigma2,
                                      priorgammavar,
                                      sigma2_mu_y, binmat_all_y_z, BztBz_y, firstcolindtrees_y,
                                      Xmat_train, Bmean_p, invBvar_p, gamma0 ) {


  XBmat <- cbind(Xmat_train, binmat_all_y_z)

  S_j = crossprod(XBmat, R)

  # there are two ways of calculating the marginal probability
  tempmat <- (1/sigma2)*BztBz_y
  diag(tempmat) <- diag(tempmat) + c(rep(1/sigma2_mu_y, ncol(BztBz_y) -1 ), 1/priorgammavar)

  offdiagblock <- crossprod(Xmat_train, binmat_all_y_z)*(1/sigma2)

  # print("dim(crossprod(Xmat_train)) = " )
  # print(dim(crossprod(Xmat_train)))
  # print("dim(invBvar_p) = " )
  # print(dim(invBvar_p))

  tempmat <- rbind(cbind(crossprod(Xmat_train)*(1/sigma2) + invBvar_p ,
                         offdiagblock),
                   cbind(t(offdiagblock), tempmat))
  U = chol ( tempmat#, pivot = TRUE, tol = 0.0001
  )
  IR = backsolve (U , diag ( ncol(XBmat) ))


  tempsj <- (1/sigma2)*S_j
  tempsj[1:ncol(Xmat_train)] <-  tempsj[1:ncol(Xmat_train)] + invBvar_p %*% Bmean_p
  tempsj[length(tempsj)] <- tempsj[length(tempsj)] + gamma0/priorgammavar

  mugamma_mean = tcrossprod (  IR , crossprod( tempsj, IR ))
  mugamma_sample = mugamma_mean + IR %*% rnorm ( ncol(IR) )

  p_lin <- ncol(Xmat_train)

  # perhaps it would be more efficient to store one mu vector throughout all the code.
  for(i in 1:length(trees)){
    which_terminal = which(trees[[i]]$tree_matrix[,'terminal'] == 1)
    trees[[i]]$tree_matrix[,'mu'] <- NA
    if(i < length(trees)){
      trees[[i]]$tree_matrix[which_terminal,'mu'] = mugamma_sample[p_lin + (firstcolindtrees_y[i]:(firstcolindtrees_y[i+1]-1))]
    }else{
      trees[[i]]$tree_matrix[which_terminal,'mu'] = mugamma_sample[p_lin + (firstcolindtrees_y[i]:(ncol(BztBz_y)-1) )]
    }
  }

  ret_list <- list()
  ret_list[[1]] <- trees
  ret_list[[2]] <- mugamma_mean[(p_lin+1):(p_lin+ncol(BztBz_y)-1)]
  ret_list[[3]] <- mugamma_sample[(p_lin+1):(p_lin+ncol(BztBz_y)-1)]
  ret_list[[4]] <- mugamma_mean[p_lin+ncol(BztBz_y)]
  ret_list[[5]] <- mugamma_sample[p_lin+ncol(BztBz_y)]
  ret_list[[6]] <- mugamma_sample[1:p_lin]

  return(ret_list)
}


simulate_mu_all_y_fast_lin = function(trees, R, sigma2,
                                  priorgammavar,
                                  sigma2_mu_y, binmat_all_y_z, BztBz_y, firstcolindtrees_y,
                                  IR, S_j,
                                  Xmat_train, Bmean_p, invBvar_p, gamma0 ) {

  tempsj <- (1/sigma2)*S_j
  tempsj[1:ncol(Xmat_train)] <-  tempsj[1:ncol(Xmat_train)] + invBvar_p %*% Bmean_p
  tempsj[length(tempsj)] <- tempsj[length(tempsj)] + gamma0/priorgammavar
  mugamma_mean = tcrossprod (  IR , crossprod( tempsj, IR ))
  mugamma_sample = mugamma_mean + IR %*% rnorm ( ncol(IR) )

  p_lin <- ncol(Xmat_train)

  # perhaps it would be more efficient to store one mu vector throughout all the code.
  for(i in 1:length(trees)){
    which_terminal = which(trees[[i]]$tree_matrix[,'terminal'] == 1)
    trees[[i]]$tree_matrix[,'mu'] <- NA
    if(i < length(trees)){
      trees[[i]]$tree_matrix[which_terminal,'mu'] = mugamma_sample[p_lin + (firstcolindtrees_y[i]:(firstcolindtrees_y[i+1]-1))]
    }else{
      trees[[i]]$tree_matrix[which_terminal,'mu'] = mugamma_sample[p_lin + (firstcolindtrees_y[i]:(ncol(BztBz_y)-1) )]
    }
  }

  ret_list <- list()
  ret_list[[1]] <- trees
  ret_list[[2]] <- mugamma_mean[(p_lin+1):(p_lin+ncol(BztBz_y)-1)]
  ret_list[[3]] <- mugamma_sample[(p_lin+1):(p_lin+ncol(BztBz_y)-1)]
  ret_list[[4]] <- mugamma_mean[p_lin+ncol(BztBz_y)]
  ret_list[[5]] <- mugamma_sample[p_lin+ncol(BztBz_y)]
  ret_list[[6]] <- mugamma_sample[1:p_lin]

  return(ret_list)
}



simulate_mu_all_y_nogamma = function(trees,
                                     R, # sigma2,
                                     phi,
                                     sigma2_mu,
                                     binmat_all_y,
                                     BtB_y,
                                     firstcolindtrees_y) {

  S_j = crossprod(binmat_all_y, R)

  tempmat <- BtB_y*(1/phi) + (1/sigma2_mu)*diag(ncol(BtB_y))

  U = chol ( tempmat#, pivot = TRUE, tol = 0.0001
  )
  IR = backsolve (U , diag ( ncol(BtB_y) ))
  # btilde = crossprod ( t ( IR ))%*%( crossprod (X_node , r_node ) )
  # beta_hat = btilde + sqrt ( sigma2 )* IR %*% rnorm ( p )
  # tSjtempmatinvSj = crossprod ( crossprod( IR , S_j ) )

  mumean = tcrossprod (  IR , crossprod( S_j*(1/phi), IR ))
  mu_sample = mumean + IR %*% rnorm ( ncol(BtB_y) )

  # perhaps it would be more efficient to store one mu vector throughout all the code.
  for(i in 1:length(trees)){
    which_terminal = which(trees[[i]]$tree_matrix[,'terminal'] == 1)
    trees[[i]]$tree_matrix[,'mu'] <- NA
    if(i < length(trees)){
      trees[[i]]$tree_matrix[which_terminal,'mu'] = mu_sample[firstcolindtrees_y[i]:(firstcolindtrees_y[i+1]-1)]
    }else{
      trees[[i]]$tree_matrix[which_terminal,'mu'] = mu_sample[firstcolindtrees_y[i]:(ncol(BtB_y))]
    }
  }

  ret_list <- list()
  ret_list[[1]] <- trees
  ret_list[[2]] <- mumean
  ret_list[[3]] <- mu_sample

  return(ret_list)
}





simulate_mu_all_y_nogamma_fast = function(trees,
                                      R, # sigma2,
                                      phi,
                                      sigma2_mu,
                                      binmat_all_y,
                                      BtB_y,
                                      firstcolindtrees_y,
                                      IR,
                                      S_j) {
  tempsj <- (1/phi)*S_j
  mumean = tcrossprod (  IR , crossprod( tempsj, IR ))
  mu_sample = mumean + IR %*% rnorm ( ncol(BtB_y) )

  # perhaps it would be more efficient to store one mu vector throughout all the code.
  for(i in 1:length(trees)){
    which_terminal = which(trees[[i]]$tree_matrix[,'terminal'] == 1)
    trees[[i]]$tree_matrix[,'mu'] <- NA
    if(i < length(trees)){
      trees[[i]]$tree_matrix[which_terminal,'mu'] = mu_sample[firstcolindtrees_y[i]:(firstcolindtrees_y[i+1]-1)]
    }else{
      trees[[i]]$tree_matrix[which_terminal,'mu'] = mu_sample[firstcolindtrees_y[i]:(ncol(BtB_y))]
    }
  }

  ret_list <- list()
  ret_list[[1]] <- trees
  ret_list[[2]] <- mumean
  ret_list[[3]] <- mu_sample

  return(ret_list)
}




simulate_mu_all_y_nogamma_lin = function(trees,
                                              R, # sigma2,
                                              phi,
                                              sigma2_mu,
                                              binmat_all_y,
                                              BtB_y,
                                              firstcolindtrees_y,
                                              Xmat_train, Bmean_p, invBvar_p) {


  XBmat <- cbind(Xmat_train, binmat_all_y)

  S_j = crossprod(XBmat, R)
  tempmat <- BtB_y*(1/phi) + (1/sigma2_mu)*diag(ncol(BtB_y))
  offdiagblock <- crossprod(Xmat_train, binmat_all_y)*(1/phi)
  tempmat <- rbind(cbind(crossprod(Xmat_train)*(1/phi) + invBvar_p ,
                         offdiagblock),
                   cbind(t(offdiagblock), tempmat))

  U = chol ( tempmat#, pivot = TRUE, tol = 0.0001
  )
  IR = backsolve (U , diag ( ncol(XBmat) ))

  tempsj <- (1/phi)*S_j
  tempsj[1:ncol(Xmat_train)] <-  tempsj[1:ncol(Xmat_train)] + invBvar_p %*% Bmean_p

  mumean = tcrossprod (  IR , crossprod( tempsj, IR ))
  mu_sample = mumean + IR %*% rnorm ( ncol(XBmat) )

  p_lin <- ncol(Xmat_train)

  # perhaps it would be more efficient to store one mu vector throughout all the code.
  for(i in 1:length(trees)){
    which_terminal = which(trees[[i]]$tree_matrix[,'terminal'] == 1)
    trees[[i]]$tree_matrix[,'mu'] <- NA
    if(i < length(trees)){
      trees[[i]]$tree_matrix[which_terminal,'mu'] = mu_sample[p_lin + (firstcolindtrees_y[i]:(firstcolindtrees_y[i+1]-1))]
    }else{
      trees[[i]]$tree_matrix[which_terminal,'mu'] = mu_sample[p_lin + (firstcolindtrees_y[i]:(ncol(BtB_y)) )]
    }
  }


  ret_list <- list()
  ret_list[[1]] <- trees
  ret_list[[2]] <- mumean
  ret_list[[3]] <- mu_sample[(p_lin+1):(length(mu_sample))]
  ret_list[[4]] <- mu_sample[1:p_lin]
  ret_list[[5]] <- mu_sample



  return(ret_list)
}





simulate_mu_all_y_nogamma_fast_lin = function(trees,
                                          R, # sigma2,
                                          phi,
                                          sigma2_mu,
                                          binmat_all_y,
                                          BtB_y,
                                          firstcolindtrees_y,
                                          IR,
                                          S_j,
                                          Xmat_train, Bmean_p, invBvar_p) {

  tempsj <- (1/phi)*S_j
  tempsj[1:ncol(Xmat_train)] <-  tempsj[1:ncol(Xmat_train)] + invBvar_p %*% Bmean_p

  mumean = tcrossprod (  IR , crossprod( tempsj, IR ))
  mu_sample = mumean + IR %*% rnorm ( ncol(IR) )

  p_lin <- ncol(Xmat_train)

  # perhaps it would be more efficient to store one mu vector throughout all the code.
  for(i in 1:length(trees)){
    which_terminal = which(trees[[i]]$tree_matrix[,'terminal'] == 1)
    trees[[i]]$tree_matrix[,'mu'] <- NA
    if(i < length(trees)){
      trees[[i]]$tree_matrix[which_terminal,'mu'] = mu_sample[p_lin + (firstcolindtrees_y[i]:(firstcolindtrees_y[i+1]-1))]
    }else{
      trees[[i]]$tree_matrix[which_terminal,'mu'] = mu_sample[p_lin + (firstcolindtrees_y[i]:(ncol(BtB_y)) )]
    }
  }


  ret_list <- list()
  ret_list[[1]] <- trees
  ret_list[[2]] <- mumean
  ret_list[[3]] <- mu_sample[(p_lin+1):(length(mu_sample))]
  ret_list[[4]] <- mu_sample[1:p_lin]
  ret_list[[5]] <- mu_sample



  return(ret_list)
}



simulate_mu_weighted_all_z = function(trees, R, # sigma2,
                                sigma2_mu, weight_vec,
                                weightz,
                                binmat_all_z, cens_inds, uncens_inds, BtB_z_u, BtB_z_c,
                                firstcolindtrees_z) {

  # check if this is faster then binary matrix approach. Can avoid creating the binary matrix entirely
  # S_j <- rep(0,0) # initialize as numeric vector of length 0
  # for(i in 1:length(trees)){
  #   which_terminal = which(trees[[i]]$tree_matrix[,'terminal'] == 1)
  #   nj = trees[[i]]$tree_matrix[which_terminal,'node_size']
  #   S_j = c(S_j,fsum(R*weightz,factor(trees[[i]]$node_indices, levels = which_terminal ), fill = TRUE))
  # }
  S_j = crossprod(binmat_all_z, R*weight_vec)

  tempmat <- BtB_z_u*weightz + BtB_z_c + (1/sigma2_mu)*diag(ncol(BtB_z_u))

  # log_post <- 0.5*(- ncol(BtB_z_u)*log(sigma2_mu) + determinant(tempmat, logarithm = TRUE) +
  #                    t(S_j)%*%solve(tempmat)%*% S_j)

  U = chol ( tempmat#, pivot = TRUE, tol = 0.0001
             )
  IR = backsolve (U , diag ( ncol(BtB_z_u) ))
  # btilde = crossprod ( t ( IR ))%*%( crossprod (X_node , r_node ) )
  # beta_hat = btilde + sqrt ( sigma2 )* IR %*% rnorm ( p )
  # tSjtempmatinvSj = crossprod ( crossprod( IR , S_j ) )

  mumean = tcrossprod (  IR , crossprod( S_j, IR ))
  mu_sample = mumean + IR %*% rnorm ( ncol(BtB_z_u) )

  # perhaps it would be more efficient to store one mu vector throughout all the code.
  for(i in 1:length(trees)){
    which_terminal = which(trees[[i]]$tree_matrix[,'terminal'] == 1)
    trees[[i]]$tree_matrix[,'mu'] <- NA
    if(i < length(trees)){
      trees[[i]]$tree_matrix[which_terminal,'mu'] = mu_sample[firstcolindtrees_z[i]:(firstcolindtrees_z[i+1]-1)]
    }else{
      trees[[i]]$tree_matrix[which_terminal,'mu'] = mu_sample[firstcolindtrees_z[i]:(ncol(BtB_z_u))]
    }
  }

  ret_list <- list()
  ret_list[[1]] <- trees
  ret_list[[2]] <- mumean
  ret_list[[3]] <- mu_sample

  return(ret_list)
}




simulate_mu_weighted_all_z_fast = function(trees, R, # sigma2,
                                      sigma2_mu, weight_vec,
                                      weightz,
                                      binmat_all_z, cens_inds, uncens_inds, BtB_z_u, BtB_z_c,
                                      firstcolindtrees_z, IR, S_j) {




  # check if this is faster then binary matrix approach. Can avoid creating the binary matrix entirely
  # S_j <- rep(0,0) # initialize as numeric vector of length 0
  # for(i in 1:length(trees)){
  #   which_terminal = which(trees[[i]]$tree_matrix[,'terminal'] == 1)
  #   nj = trees[[i]]$tree_matrix[which_terminal,'node_size']
  #   S_j = c(S_j,fsum(R*weightz,factor(trees[[i]]$node_indices, levels = which_terminal ), fill = TRUE))
  # }
  # S_j = crossprod(binmat_all_z, R*weight_vec)
  #
  # tempmat <- BtB_z_u*weightz + BtB_z_c + (1/sigma2_mu)*diag(ncol(BtB_z_u))
  #
  # # log_post <- 0.5*(- ncol(BtB_z_u)*log(sigma2_mu) + determinant(tempmat, logarithm = TRUE) +
  # #                    t(S_j)%*%solve(tempmat)%*% S_j)
  #
  # U = chol ( tempmat)
  # IR = backsolve (U , diag ( ncol(BtB_z_u) ))
  # btilde = crossprod ( t ( IR ))%*%( crossprod (X_node , r_node ) )
  # beta_hat = btilde + sqrt ( sigma2 )* IR %*% rnorm ( p )
  # tSjtempmatinvSj = crossprod ( crossprod( IR , S_j ) )

  mumean = tcrossprod (  IR , crossprod( S_j, IR ))
  mu_sample = mumean + IR %*% rnorm ( ncol(BtB_z_u) )

  # perhaps it would be more efficient to store one mu vector throughout all the code.
  for(i in 1:length(trees)){
    which_terminal = which(trees[[i]]$tree_matrix[,'terminal'] == 1)
    trees[[i]]$tree_matrix[,'mu'] <- NA
    if(i < length(trees)){
      trees[[i]]$tree_matrix[which_terminal,'mu'] = mu_sample[firstcolindtrees_z[i]:(firstcolindtrees_z[i+1]-1)]
    }else{
      trees[[i]]$tree_matrix[which_terminal,'mu'] = mu_sample[firstcolindtrees_z[i]:(ncol(BtB_z_u))]
    }
  }

  ret_list <- list()
  ret_list[[1]] <- trees
  ret_list[[2]] <- mumean
  ret_list[[3]] <- mu_sample

  return(ret_list)
}




simulate_mu_weighted_all_z_lin = function(trees, R, # sigma2,
                                               sigma2_mu, weight_vec,
                                               weightz,
                                               binmat_all_z, cens_inds, uncens_inds, BtB_z_u, BtB_z_c,
                                               firstcolindtrees_z, wmat_train, Amean_p, invAvar_p) {


  WBmat <- cbind(wmat_train, binmat_all_z)

  S_j = crossprod(WBmat, R*weight_vec) + c(invAvar_p %*% Amean_p, rep(0, ncol(BtB_z_u)))

  tempmat <- BtB_z_u*weightz + BtB_z_c + (1/sigma2_mu)*diag(ncol(BtB_z_u))

  offdiagblock <- crossprod(wmat_train[uncens_inds,,drop = FALSE], binmat_all_z[uncens_inds,,drop = FALSE])*weightz +
    crossprod(wmat_train[cens_inds,,drop = FALSE], binmat_all_z[cens_inds,,drop = FALSE])

  tempmat <- rbind(cbind(crossprod(wmat_train[uncens_inds,,drop = FALSE])*weightz +
                           crossprod(wmat_train[cens_inds,,drop = FALSE]) + invAvar_p ,
                         offdiagblock),
                   cbind(t(offdiagblock), tempmat))


  U = chol ( tempmat#, pivot = TRUE, tol = 0.0001
  )
  IR = backsolve (U , diag ( ncol(BtB_z_u) + ncol(wmat_train) ))

  mumean = tcrossprod (  IR , crossprod( S_j, IR ))
  mu_sample = mumean + IR %*% rnorm ( ncol(IR) )

  p_lin <- ncol(wmat_train)

  # perhaps it would be more efficient to store one mu vector throughout all the code.
  for(i in 1:length(trees)){
    which_terminal = which(trees[[i]]$tree_matrix[,'terminal'] == 1)
    trees[[i]]$tree_matrix[,'mu'] <- NA
    if(i < length(trees)){
      trees[[i]]$tree_matrix[which_terminal,'mu'] = mu_sample[p_lin + (firstcolindtrees_z[i]:(firstcolindtrees_z[i+1]-1))]
    }else{
      trees[[i]]$tree_matrix[which_terminal,'mu'] = mu_sample[p_lin + (firstcolindtrees_z[i]:(ncol(BtB_z_u)))]
    }
  }

  if(any(is.na(mu_sample))){
    stop("any(is.na(mu_sample))")
  }

  ret_list <- list()
  ret_list[[1]] <- trees
  ret_list[[2]] <- mumean
  ret_list[[3]] <- mu_sample[(p_lin+1):(length(mu_sample))]
  ret_list[[4]] <- mu_sample[1:p_lin]
  ret_list[[5]] <- mu_sample

  return(ret_list)
}

simulate_mu_weighted_all_z_fast_lin = function(trees, R, # sigma2,
                                      sigma2_mu, weight_vec,
                                      weightz,
                                      binmat_all_z, cens_inds, uncens_inds, BtB_z_u, BtB_z_c,
                                      firstcolindtrees_z, IR, S_j, Wmat_train, Amean_p, invAvar_p) {

  # check if this is faster then binary matrix approach. Can avoid creating the binary matrix entirely
  # S_j <- rep(0,0) # initialize as numeric vector of length 0
  # for(i in 1:length(trees)){
  #   which_terminal = which(trees[[i]]$tree_matrix[,'terminal'] == 1)
  #   nj = trees[[i]]$tree_matrix[which_terminal,'node_size']
  #   S_j = c(S_j,fsum(R*weightz,factor(trees[[i]]$node_indices, levels = which_terminal ), fill = TRUE))
  # }
  # S_j = crossprod(binmat_all_z, R*weight_vec)
  #
  # tempmat <- BtB_z_u*weightz + BtB_z_c + (1/sigma2_mu)*diag(ncol(BtB_z_u))
  #
  # # log_post <- 0.5*(- ncol(BtB_z_u)*log(sigma2_mu) + determinant(tempmat, logarithm = TRUE) +
  # #                    t(S_j)%*%solve(tempmat)%*% S_j)
  #
  # U = chol ( tempmat)
  # IR = backsolve (U , diag ( ncol(BtB_z_u) ))
  # btilde = crossprod ( t ( IR ))%*%( crossprod (X_node , r_node ) )
  # beta_hat = btilde + sqrt ( sigma2 )* IR %*% rnorm ( p )
  # tSjtempmatinvSj = crossprod ( crossprod( IR , S_j ) )

  mumean = tcrossprod (  IR , crossprod( S_j + c(invAvar_p %*% Amean_p, rep(0, ncol(BtB_z_u)))  , IR ))
  mu_sample = mumean + IR %*% rnorm ( ncol(IR) )

  p_lin <- ncol(Wmat_train)

  # perhaps it would be more efficient to store one mu vector throughout all the code.
  for(i in 1:length(trees)){
    which_terminal = which(trees[[i]]$tree_matrix[,'terminal'] == 1)
    trees[[i]]$tree_matrix[,'mu'] <- NA
    if(i < length(trees)){
      trees[[i]]$tree_matrix[which_terminal,'mu'] = mu_sample[p_lin + (firstcolindtrees_z[i]:(firstcolindtrees_z[i+1]-1))]
    }else{
      trees[[i]]$tree_matrix[which_terminal,'mu'] = mu_sample[p_lin + (firstcolindtrees_z[i]:(ncol(BtB_z_u)))]
    }
  }

  if(any(is.na(mu_sample))){
    stop("any(is.na(mu_sample))")
  }

  ret_list <- list()
  ret_list[[1]] <- trees
  ret_list[[2]] <- mumean
  ret_list[[3]] <- mu_sample[(p_lin+1):(length(mu_sample))]
  ret_list[[4]] <- mu_sample[1:p_lin]
  ret_list[[5]] <- mu_sample

  return(ret_list)
}
# Update sigma2 -------------------------------------------------------------

update_sigma2 <- function(S, n, nu, lambda){
  u = 1/rgamma(1, shape = (n + nu)/2, rate = (S + nu*lambda)/2)
  return(u)
}

# Update the latent variable z ---------------

update_z = function(y, prediction){

  ny0 = sum(y==0)
  ny1 = sum(y==1)
  z = rep(NA, length(y))

  z[y==0] = rtruncnorm(ny0, a = -Inf, b=0,   mean = prediction[y==0], 1)
  z[y==1] = rtruncnorm(ny1, a = 0   , b=Inf, mean = prediction[y==1], 1)

  return(z)
}

# Get tree priors ---------------------------------------------------------

  get_tree_prior = function(tree, alpha, beta) {

  # Need to work out the depth of the tree
  # First find the level of each node, then the depth is the maximum of the level
  level = rep(NA, nrow(tree$tree_matrix))
  level[1] = 0 # First row always level 0

  # Escpae quickly if tree is just a stump
  if(nrow(tree$tree_matrix) == 1) {
    return(log(1 - alpha)) # Tree depth is 0
  }

  for(i in 2:nrow(tree$tree_matrix)) {
    # Find the current parent
    curr_parent = as.numeric(tree$tree_matrix[i,'parent'])
    # This child must have a level one greater than it's current parent
    level[i] = level[curr_parent] + 1
  }

  # Only compute for the internal nodes
  internal_nodes = which(as.numeric(tree$tree_matrix[,'terminal']) == 0)
  log_prior = 0
  for(i in 1:length(internal_nodes)) {
    log_prior = log_prior + log(alpha) - beta * log(1 + level[internal_nodes[i]])
  }
  # Now add on terminal nodes
  terminal_nodes = which(as.numeric(tree$tree_matrix[,'terminal']) == 1)
  for(i in 1:length(terminal_nodes)) {
    log_prior = log_prior + log(1 - alpha * ((1 + level[terminal_nodes[i]])^(-beta)))
  }

  return(log_prior)

  }

# get_num_cov_prior <- function(tree, lambda_cov, nu_cov, penalise_num_cov){
#
#   # Select the rows that correspond to internal nodes
#   which_terminal = which(tree$tree_matrix[,'terminal'] == 0)
#   # Get the covariates that are used to define the splitting rules
#   num_distinct_cov = length(unique(tree$tree_matrix[which_terminal,'split_variable']))
#
#   if (penalise_num_cov == TRUE){
#     if (num_distinct_cov > 0){
#       # A zero-truncated double Poisson
#       log_prior_num_cov = ddoublepois(num_distinct_cov, lambda_cov, nu_cov, log = TRUE) -
#         log(1-ddoublepois(0,lambda_cov, nu_cov))
#     } else {
#       log_prior_num_cov = 0
#     }
#   } else {
#     log_prior_num_cov = 0 # no penalisation
#   }
#
#   return(log_prior_num_cov)
#
# }

  ### The code below I took from the Eoghan O'Neill's github

  # This functions calculates the number of terminal nodes
  get_nterminal = function(tree){

    indeces = which(tree$tree_matrix[,'terminal'] == '1') # determine which indices have a terminal node label
    b = as.numeric(length(indeces)) # take the length of these indices, to determine the number of terminal nodes/leaves
    return(b)
  }

  # This function calculates the number of parents with two terminal nodes/ second generation internal nodes as formulated in bartMachine
  get_w = function(tree){
    indeces = which(tree$tree_matrix[,'terminal'] == '1') #determine which indices have a terminal node label
    # determine the parent for each terminal node and sum the number of duplicated parents
    w = as.numeric(sum(duplicated(tree$tree_matrix[indeces,'parent'])))
    return(w)
  }

  # These functions calculate the grow and prune ratios respectively according to the Bart Machine/ soft BART papers
  ratio_grow = function(curr_tree, new_tree){
    grow_ratio = get_nterminal(curr_tree)/(get_w(new_tree)) # (get_w(new_tree)+1)

    return(as.numeric(grow_ratio))
  }

  ratio_prune = function(curr_tree, new_tree){
    prune_ratio = get_w(curr_tree)/(get_nterminal(curr_tree)-1) #(get_nterminal(curr_tree)-1)

    return(as.numeric(prune_ratio))
  }


  # The alpha MH function calculates the alpha of the Metropolis-Hastings step based on the type of transformation
  # input: transformation type, current tree, proposed/new tree and the respective likelihoods
  alpha_mh = function(l_new,l_old, curr_trees,new_trees, type){
    if(type == 'grow'){
      a = exp(l_new - l_old)*ratio_grow(curr_trees, new_trees)

    } else if(type == 'prune'){

      a = exp(l_new - l_old)*ratio_prune(curr_trees, new_trees)

    } else{
      a = exp(l_new - l_old)

    }
    return(a)
  }


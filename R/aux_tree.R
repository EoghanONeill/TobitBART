# -------------------------------------------------------------------------#
# Description: this script contains auxiliar functions needed to update    #
# the trees with details and to map the predicted values to each obs       #
# -------------------------------------------------------------------------#

# 1. fill_tree_details: takes a tree matrix and returns the number of obs in each node in it and the indices of each observation in each terminal node
# 2. get_predictions: gets the predicted values from a current set of trees
# 3. get_children: it's a function that takes a node and, if the node is terminal, returns the node. If not, returns the children and calls the function again on the children
# 4. resample: an auxiliar function
# 5. get_ancestors: get the ancestors of all terminal nodes in a tree
# 6. update_s: full conditional of the vector of splitting probability.
# 7. get_number_distinct_cov: given a tree, it returns the number of distinct covariates used to create its structure

# Fill_tree_details -------------------------------------------------------

fill_tree_details = function(curr_tree, X) {

  # Collect right bits of tree
  tree_matrix = curr_tree$tree_matrix

  # Create a new tree matrix to overwrite
  new_tree_matrix = tree_matrix

  # Start with dummy node indices
  node_indices = rep(1, nrow(X))

  # For all but the top row, find the number of observations falling into each one
  if(nrow(tree_matrix) > 1){
    for(i in 2:nrow(tree_matrix)) {
      # Get the parent
      curr_parent = as.numeric(tree_matrix[i,'parent'])

      # Find the split variable and value of the parent
      split_var = as.numeric(tree_matrix[curr_parent,'split_variable'])
      split_val = as.numeric(tree_matrix[curr_parent, 'split_value'])

      # Find whether it's a left or right terminal node
      left_or_right = ifelse(tree_matrix[curr_parent,'child_left'] == i,
                             'left', 'right')
      if(left_or_right == 'left') {
        # If left use less than condition
        new_tree_matrix[i,'node_size'] <- sum(X[node_indices == curr_parent,split_var] < split_val)
        node_indices[node_indices == curr_parent][X[node_indices == curr_parent,split_var] < split_val] <- i
      } else {
        # If right use greater than condition
        new_tree_matrix[i,'node_size'] <- sum(X[node_indices == curr_parent,split_var] >= split_val)
        node_indices[node_indices == curr_parent][X[node_indices == curr_parent,split_var] >= split_val] <- i
      }
    } # End of loop through table
  }



  # node_indices2 = rep(1, nrow(X))
  # leaf_found <- rep(FALSE,nrow(X))
  # for(i in 1:nrow(tree_matrix)) {
  #   split_var = as.numeric(tree_matrix[i,'split_variable'])
  #   split_val = as.numeric(tree_matrix[i, 'split_value'])
  #
  #   leaf_found <- (leaf_found | rep(tree_matrix[i, 'terminal'], nrow(X)))
  #
  #   child_indices <- 2*node_indices2 + 1*(X[,split_var] > split_val)
  #   node_indices2 <- ifelse(leaf_found, node_indices2, child_indices)
  # }
  #


  return(list(tree_matrix = new_tree_matrix,
              node_indices = node_indices))

} # End of function

# Get predictions ---------------------------------------------------------

get_predictions = function(trees, X, single_tree = FALSE) {

  # Stop nesting problems in case of multiple trees
  if(is.null(names(trees)) & (length(trees) == 1)) trees = trees[[1]]

  # Normally trees will be a list of lists but just in case
  if(single_tree) {
    # Deal with just a single tree
    if(nrow(trees$tree_matrix) == 1) {
      predictions = rep(trees$tree_matrix[1, 'mu'], nrow(X))
    } else {
      # Loop through the node indices to get predictions
      predictions = rep(NA, nrow(X))
      # unique_node_indices = unique(trees$node_indices)
      terminal_rows <- which(trees$tree_matrix[, "terminal"] == 1)

      # Get the node indices for the current X matrix
      curr_X_node_indices = fill_tree_details(trees, X)$node_indices
      # Now loop through all node indices to fill in details
      for(i in 1:length(terminal_rows)) {
        predictions[curr_X_node_indices == terminal_rows[i]] =
          trees$tree_matrix[terminal_rows[i], 'mu']
      }
    }
    # More here to deal with more complicated trees - i.e. multiple trees
  } else {
    # Do a recursive call to the function
    partial_trees = trees
    partial_trees[[1]] = NULL # Blank out that element of the list
    predictions = get_predictions(trees[[1]], X, single_tree = TRUE)  +
      get_predictions(partial_trees, X,
                      single_tree = length(partial_trees) == 1)
    #single_tree = !is.null(names(partial_trees)))
    # The above only sets single_tree to if the names of the object is not null (i.e. is a list of lists)
  }

  return(unname(as.numeric(predictions)))
}

get_predictions_soft_slow = function(trees, X, tau_vec) {
  predictions <- rep(NA, nrow(X))
  terminal_rows <- which(trees[[1]]$tree_matrix[, "terminal"] == 1)
  if(nrow( trees[[1]]$tree_matrix)==1){
    predictions <- rep(trees[[1]]$tree_matrix[1, 'mu'], nrow(X))
  }else{
    predictions <- phi_app(X,get_branch(trees[[1]]),tau_vec[1]) %*% trees[[1]]$tree_matrix[terminal_rows, 'mu']
  }
  # print("get_predictions_soft_slow line 117")
  if(length(trees) > 1){
    for(i in 2:length(trees)){
      terminal_rows <- which(trees[[i]]$tree_matrix[, "terminal"] == 1)
      if(nrow( trees[[i]]$tree_matrix)==1){
        predictions <- predictions + rep(trees[[i]]$tree_matrix[1, 'mu'], nrow(X))
      }else{
        predictions <- predictions + phi_app(X,get_branch(trees[[i]]),tau_vec[i]) %*% trees[[i]]$tree_matrix[terminal_rows, 'mu']
      }
      # predictions <- predictions + phi_app(X,get_branch(trees[[i]]),tau_vec[i]) %*% trees[[i]]$tree_matrix[terminal_rows, 'mu']
    }
  }
  # print("get_predictions_soft_slow line 124")
  return(as.numeric(predictions))
}

get_predictions_soft_slow_onetree = function(tree, X, tau) {
  terminal_rows <- which(tree$tree_matrix[, "terminal"] == 1)
  # print("get_predictions_soft_slow_onetree line 130")
  if(nrow( trees$tree_matrix)==1){
    predictions <- rep(trees$tree_matrix[1, 'mu'], nrow(X))
  }else{
    predictions <- phi_app(X,get_branch(tree),tau) %*% tree$tree_matrix[terminal_rows, 'mu']
  }
  # print("get_predictions_soft_slow_onetree line 132")
  return(as.numeric(predictions))
}


get_predictions_HBART = function(trees, X, single_tree = FALSE) {

  # product of terminal node parameters
  # equal to exponential of sum of logs
  # predictions saves the observation-specific sums of logs
  # then return the exponential

  # perhaps it is not very efficient to take many logarithms
  # therefore the tree matrix contains logs of the s samples
  # in this case, can just use the original get_predictions function
  # and take the exponential of the output

  # Stop nesting problems in case of multiple trees
  if(is.null(names(trees)) & (length(trees) == 1)) trees = trees[[1]]

  # Normally trees will be a list of lists but just in case
  if(single_tree) {
    # Deal with just a single tree
    if(nrow(trees$tree_matrix) == 1) {
      predictions = rep(trees$tree_matrix[1, 'mu'], nrow(X))
    } else {
      # Loop through the node indices to get predictions
      predictions = rep(NA, nrow(X))
      # unique_node_indices = unique(trees$node_indices)
      terminal_rows <- which(trees$tree_matrix[, "terminal"] == 1)
      # Get the node indices for the current X matrix
      curr_X_node_indices = fill_tree_details(trees, X)$node_indices
      # Now loop through all node indices to fill in details
      for(i in 1:length(terminal_rows)) {
        predictions[curr_X_node_indices == terminal_rows[i]] =
          trees$tree_matrix[terminal_rows[i], 'mu']
      }
    }
    # More here to deal with more complicated trees - i.e. multiple trees
  } else {
    # Do a recursive call to the function
    partial_trees = trees
    partial_trees[[1]] = NULL # Blank out that element of the list
    predictions = get_predictions(trees[[1]], X, single_tree = TRUE)  +
      get_predictions(partial_trees, X,
                      single_tree = length(partial_trees) == 1)
    #single_tree = !is.null(names(partial_trees)))
    # The above only sets single_tree to if the names of the object is not null (i.e. is a list of lists)
  }

  return(exp(predictions))
}

# get_children ------------------------------------------------------------

get_children = function(tree_mat, parent) {
  # Create a holder for the children
  all_children = NULL
  if(as.numeric(tree_mat[parent,'terminal']) == 1) {
    # If the node is terminal return the list so far
    return(c(all_children, parent))
  } else {
    # If not get the current children
    curr_child_left = as.numeric(tree_mat[parent, 'child_left'])
    curr_child_right = as.numeric(tree_mat[parent, 'child_right'])
    # Return the children and also the children of the children recursively
    return(c(all_children,
             get_children(tree_mat,curr_child_left),
             get_children(tree_mat,curr_child_right)))
  }
}

# Sample function ----------------------------------------------------------

resample <- function(x, ...) x[sample.int(length(x), size=1), ...]

update_s <- function(var_count, p, alpha_s) {
  # s_ = rdirichlet(1, as.vector((alpha_s / p ) + var_count))

  # // Get shape vector
  # shape_up = alpha_s / p
  # print("var_count = ")
  # print(var_count)
  # print("alpha_s = ")
  # print(alpha_s)
  # print("p = ")
  # print(p)
  #
  # print("alpha_s / p = ")
  # print(alpha_s / p)

  shape_up = as.vector((alpha_s / p ) + var_count)

  # print("shape_up = ")
  # print(shape_up)

  # // Sample unnormalized s on the log scale
  templogs = rep(NA, p)
  for(i in 1:p) {
    templogs[i] = SoftBart:::rlgam(shape = shape_up[i])
  }

  if(any(templogs== -Inf)){
    print("alpha_s = ")
    print(alpha_s)
    print("var_count = ")
    print(var_count)
    print("templogs = ")
    print(templogs)
    stop('templogs == -Inf')
  }

  # // Normalize s on the log scale, then exponentiate
  # templogs = templogs - log_sum_exp(hypers.logs);
  max_log = max(templogs)
  templogs2 = templogs - (max_log + log(sum(exp( templogs  -  max_log ))))


  s_ = exp(templogs2)

  # if(any(s_==0)){
  #   print("templogs2 = ")
  #   print(templogs2)
  #   print("templogs = ")
  #   print(templogs)
  #   print("alpha_s = ")
  #   print(alpha_s)
  #   print("var_count = ")
  #   print(var_count)
  #   print("s_ = ")
  #   print(s_)
  #   stop('s_ == 0')
  # }

  ret_list <- list()
  ret_list[[1]] <- s_
  ret_list[[2]] <- mean(templogs2)


  return(ret_list)
}

get_number_distinct_cov <- function(tree){

  # Select the rows that correspond to internal nodes
  which_terminal = which(tree$tree_matrix[,'terminal'] == 0)
  # Get the covariates that are used to define the splitting rules
  num_distinct_cov = length(unique(tree$tree_matrix[which_terminal,'split_variable']))

  return(num_distinct_cov)
}

sample_move = function(curr_tree, i, nburn, trans_prob){

  if (nrow(curr_tree$tree_matrix) == 1 ||
      i < max(floor(0.1*nburn), 10)) {
    type = 'grow'
  } else {
    type = sample(c('grow', 'prune', 'change'),  1, prob = trans_prob)
  }
  return(type)
}



update_alpha <- function(s, alpha_scale, alpha_a, alpha_b, p, mean_log_s) {

  # create inputs for likelihood

  # log_s <- log(s)
  # mean_log_s <- mean(log_s)
  # p <- length(s)
  # alpha_scale   # denoted by lambda_a in JRSSB paper

  rho_grid <- (1:1000)/1001

  alpha_grid <- alpha_scale * rho_grid / (1 - rho_grid )

  logliks <- alpha_grid * mean_log_s +
    lgamma(alpha_grid) -
    p*lgamma(alpha_grid/p) + # (alpha_a - 1)*log(rho_grid) + (alpha_b-1)*log(1- rho_grid)
    dbeta(x = rho_grid, shape1 = alpha_a, shape2 = alpha_b, ncp = 0, log = TRUE)


  # logliks <- log(ddirichlet( t(matrix(s, p, 1000))  , t(matrix( rep(alpha_grid/p,p) , p , 1000)  ) ) ) +
  #   (alpha_a - 1)*log(rho_grid) + (alpha_b-1)*log(1- rho_grid)
  # # dbeta(x = rho_grid, shape1 = alpha_a, shape2 = alpha_b, ncp = 0, log = TRUE)

  # logliks <- rep(NA, 1000)
  # for(i in 1:1000){
  #   logliks[i] <- log(ddirichlet(s  , rep(alpha_grid[i]/p,p) ) ) +
  #     (alpha_a - 1)*log(rho_grid[i]) + (alpha_b-1)*log(1- rho_grid[i])
  # }

  max_ll <- max(logliks)
  logsumexps <- max_ll + log(sum(exp( logliks  -  max_ll )))

  # print("logsumexps = ")
  # print(logsumexps)

  logliks <- exp(logliks - logsumexps)

  if(any(is.na(logliks))){
    print("logliks = ")
    print(logliks)

    print("logsumexps = ")
    print(logsumexps)

    print("mean_log_s = ")
    print(mean_log_s)

    print("lgamma(alpha_grid) = ")
    print(lgamma(alpha_grid))

    print("p*lgamma(alpha_grid/p) = ")
    print(p*lgamma(alpha_grid/p))

    print("(alpha_a - 1)*log(rho_grid) + (alpha_b-1)*log(1- rho_grid) = ")
    print((alpha_a - 1)*log(rho_grid) + (alpha_b-1)*log(1- rho_grid))

    print("max_ll = ")
    print(max_ll)

    # print("s = ")
    # print(s)


  }

  # print("logliks = ")
  # print(logliks)

  rho_ind <- sample.int(1000,size = 1, prob = logliks)


  return(alpha_grid[rho_ind])
}


update_sigma_mu_par <- function(trees, curr_sigmu2) {

  num_trees <- length(trees)
  mu_vec <- c()
  for(m in 1:length(trees)){
    mu_vec <- c(mu_vec,
                trees[[m]]$tree_matrix[, 'mu'])

  }

  mu_vec <- na.omit(mu_vec)

  # note Linero and Yang's sigma_mu_squared corresponds to
  # num_trees times sigma_mu_squared as defined in this package
  curr_s_mu <- sqrt(curr_sigmu2)

  prop_s_mu_minus2 <- rgamma(n = 1,
                             shape = 1 + length(mu_vec)/2,
                             rate = sum(mu_vec^2)/2)

  prop_s_mu <- sqrt(1/prop_s_mu_minus2)

  log_acceptprob <- dcauchy(prop_s_mu, 0, 0.25/sqrt(num_trees), log = TRUE) -
    dcauchy(curr_s_mu, 0, 0.25/sqrt(num_trees), log = TRUE) + 3*(log(prop_s_mu) - log(curr_s_mu))
  # acceptprob <- (dcauchy(prop_s_mu, 0, 0.25/sqrt(num_trees))/dcauchy(curr_s_mu, 0, 0.25/sqrt(num_trees)))*
  #   (prop_s_mu/curr_s_mu)^3

  # cauchy_scale <- 0.25/sqrt(length(trees))
  # cauchy_scale_sq <- 0.0625/(length(trees))
  # orig_dens <- 1/(pi*cauchy_scale* (1 + length(trees)/(curr_sigmu2*cauchy_scale_sq  )))



  # SSE <- sum(mu_vec^2)
  # n_mu <- length(mu_vec)
  # shape <- 0.5*n_mu + 1
  # scale <- 2/SSE
  # sigma_prop <- rgamma(n = 1,
  #                      shape = shape,
  #                      scale = scale)^(-0.5)
  # tau_prop <- sigma_prop^(-2)
  # tau_old <- curr_sigmu2^(-1)
  # loglik_rat <- cauchy_jacobian(tau_prop, ) - cauchy_jacobian(tau_old, )

  # acceptprob <- exp(log_acceptprob)
  # if(is.na(acceptprob)| is.null(acceptprob)){stop("Line 343. is.na(acceptprob)| is.null(acceptprob)")}
  if (length(log_acceptprob) != 1L || is.na(log_acceptprob)) {
    stop("Invalid log acceptance probability in update_sigma_mu_par().")
  }
  if (log(runif(1)) < min(0, log_acceptprob)) {
    new_s_mu <- prop_s_mu
  }else{
    new_s_mu <- curr_s_mu # sqrt(curr_sigmu2)
  }

  return(new_s_mu^2)
}



update_sigma_mu_par_Cauchy <- function(trees, curr_sigmu2, sigma_mu_hat) {

  num_trees <- length(trees)
  mu_vec <- c()
  for(m in 1:length(trees)){
    mu_vec <- c(mu_vec,
                trees[[m]]$tree_matrix[, 'mu'])

  }

  mu_vec <- na.omit(mu_vec)

  # SSE <- sum(mu_vec^2)
  # n_mu <- length(mu_vec)
  # shape <- 0.5*n_mu + 1
  # scale <- 2/SSE
  # sigma_prop <- rgamma(n = 1,
  #                      shape = shape,
  #                      scale = scale)^(-0.5)
  # tau_prop <- sigma_prop^(-2)
  # tau_old <- curr_sigmu2^(-1)
  # loglik_rat <- cauchy_jacobian(tau_prop, sigma_mu_hat) - cauchy_jacobian(tau_old, sigma_mu_hat)
  #
  # acceptprob <- exp(loglik_rat)
  # if(is.na(acceptprob)| is.null(acceptprob)){stop("Line 380. is.na(acceptprob)| is.null(acceptprob)")}
  #
  # # print("loglik_rat = ")
  # # print(loglik_rat)
  # # print("sigma_prop = ")
  # # print(sigma_prop)
  # # print("sqrt(curr_sigmu2) = ")
  # # print(sqrt(curr_sigmu2))
  # if(runif(1) < acceptprob){
  #   # print("accept sigma_mu")
  #   new_s_mu <- sigma_prop
  # }else{
  #   # print("reject sigma_mu")
  #   new_s_mu <- sqrt(curr_sigmu2)
  # }

  new_s_mu <- UpdateSigmaMu1(mu_vec, sigma_mu_hat, sqrt(curr_sigmu2) )

  return(new_s_mu^2)
}

cauchy_jacobian <- function(tau,sigma_hat) {
  sigma = tau^(-0.5)
  # give_log = 1;

  out = dcauchy(sigma, 0.0, sigma_hat, log = TRUE);
  out = out - log(2) - (3.0 / 2.0) * log(tau);
  return(out)
}
update_sigma_mu_par_IG <- function(trees, curr_sigmu2) {

  num_trees <- length(trees)
  mu_vec <- c()
  for(m in 1:length(trees)){
    mu_vec <- c(mu_vec,
                trees[[m]]$tree_matrix[, 'mu'])

  }

  mu_vec <- na.omit(mu_vec)

  # note Linero and Yang's sigma_mu_squared corresponds to
  # num_trees times sigma_mu_squared as defined in this package
  # curr_s_mu <- sqrt(curr_sigmu2)

  prop_s_mu_minus2 <- rgamma(n = 1,
                             shape = 1 + length(mu_vec)/2,
                             rate = 1 + sum(mu_vec^2)/2)

  prop_s_mu <- sqrt(1/prop_s_mu_minus2)

  return(prop_s_mu^2)
}


get_gen2 <- function(tree) {
  if (nrow(tree$tree_matrix) == 1) {
    w <- 0
  } else {
    # indeces <- which(tree$tree_matrix[, "terminal"] == 1)
    # Determine the parent for each terminal node and find the duplicated parents
    # w <- as.numeric(sum(duplicated(tree$tree_matrix[indeces,'parent'])))
    # parents <- tree$tree_matrix[indeces, "parent"]
    parents <- tree$tree_matrix[tree$tree_matrix[, "terminal"] == 1, "parent"]
    w <- parents[duplicated(parents)]
  }
  return(w)
}

count_tree_splits <- function(trees,p) {
  out <- integer(p)
  for (tree in trees) {
    vars <- tree$tree_matrix[tree$tree_matrix[, "terminal"] == 0,"split_variable"]
    vars <- as.integer(vars[!is.na(vars)])
    if (length(vars) > 0L) {out <- out +tabulate(vars, nbins = p)}
  }
  out
}


tree_col_range <- function(j,firstcolindtrees,n_trees,n_total_cols
) {
  first <- firstcolindtrees[j]
  last <- if (j < n_trees) {
    firstcolindtrees[j + 1L] - 1L
  } else {
    n_total_cols
  }
  first:last
}

check_soft_partition <- function(B,firstcolindtrees,n_trees,tolerance = 1e-10) {
  if (any(!is.finite(B))) {stop("Soft terminal matrix contains nonfinite values.")}
  if (any(B < -tolerance) ||any(B > 1 + tolerance)) {stop("Soft terminal probabilities lie outside [0, 1].")}
  for (j in seq_len(n_trees)) {
    cols <- tree_col_range(j,firstcolindtrees,n_trees,ncol(B))
    row_mass <- rowSums(B[, cols, drop = FALSE])
    if (max(abs(row_mass - 1)) >tolerance) {stop(sprintf(paste0("Terminal probabilities for tree %d do not sum to one; max error %.3e."),j,max(abs(row_mass - 1))))}
  }
  invisible(TRUE)
}

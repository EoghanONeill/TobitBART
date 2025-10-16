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
      unique_node_indices = unique(trees$node_indices)
      # Get the node indices for the current X matrix
      curr_X_node_indices = fill_tree_details(trees, X)$node_indices
      # Now loop through all node indices to fill in details
      for(i in 1:length(unique_node_indices)) {
        predictions[curr_X_node_indices == unique_node_indices[i]] =
          trees$tree_matrix[unique_node_indices[i], 'mu']
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

  return(predictions)
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

  acceptprob <- (dcauchy(prop_s_mu, 0, 0.25/sqrt(num_trees))/dcauchy(curr_s_mu, 0, 0.25/sqrt(num_trees)))*
    (prop_s_mu/curr_s_mu)^3

  if(runif(1) < acceptprob){
    new_s_mu <- prop_s_mu
  }else{
    new_s_mu <- curr_s_mu
  }

  return(new_s_mu^2)
}


update_sigma_mu_par_norm <- function(trees, curr_sigmu2) {

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


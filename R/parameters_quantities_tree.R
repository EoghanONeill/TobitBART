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




tree_full_conditional_z_marg = function(tree, R, sigma2, sigma2_mu,
                                        weightz,
                                        binmat_all_z, cens_inds, uncens_inds) {

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



tree_full_conditional_y_marg = function(tree, R, sigma2, sigma2_mu) {

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


# Simulate_par -------------------------------------------------------------

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


#' @export
predict_mybart = function(object, newdata,
                         type = c('all', 'median', 'mean')) {

  # Create holder for predicted values
  n_newX = dim(newdata)[1]
  n_its = object$npost
  y_hat_mat = matrix(NA, nrow = n_its,
                     ncol = nrow(newdata))

  # Now loop through iterations and get predictions
  for (i in 1:n_its) {
    # Get current set of trees
    curr_trees = object$trees[[i]]

    # Use get_predictions function to get predictions
    y_hat_mat[i,] = get_predictions(curr_trees,
                                    newdata,
                                    single_tree = length(curr_trees) == 1)
  }

  # Sort out what to return
  out = switch(type,
               all = object$y_mean + object$y_sd * y_hat_mat,
               mean = object$y_mean + object$y_sd * apply(y_hat_mat,2,'mean'),
               median = object$y_mean + object$y_sd * apply(y_hat_mat,2,'median'))

  return(out)

} # end of predict function
#' @export
marginal_predict_mybart = function(object, var_marg, newdata,
                               type = c('all', 'median', 'mean')) {

  # Create holder for predicted values
  n_newX = dim(newdata)[1]
  n_its = object$npost
  ntrees = object$ntrees
  y_hat_mat = matrix(0, nrow = n_its,
                     ncol = nrow(newdata))

  # Get which covariates are used by each tree in each MCMC iteration
  vars_trees = matrix(NA, nrow=n_its, ncol=ntrees)
  for (i in 1:n_its){
    for (j in 1:ntrees){
      aux = object$trees[[i]][[j]]$tree_matrix[,'split_variable']
      if(length(aux) > 1){
        vars_trees[i,j] = paste(unique(sort(aux[!is.na(aux)])), collapse = ',')
      }
    }
  }

  # Now loop through iterations and get predictions
  for (i in 1:n_its) {
    marginal_trees = which(vars_trees[i,] == var_marg)
    # Sometimes the trees do not contain the variable we're interested in
    if (length(marginal_trees) > 0){
      # Get current set of trees
      curr_trees = object$trees[[i]][marginal_trees]

      # Use get_predictions function to get predictions
      y_hat_mat[i,] = get_predictions(curr_trees,
                                      newdata,
                                      single_tree = length(curr_trees) == 1)

    }
  }

  # Sort out what to return
  out = list(vars_trees = vars_trees,
             prediction = switch(type,
                                 all = object$y_mean + object$y_sd * y_hat_mat,
                                 mean = object$y_mean + object$y_sd * apply(y_hat_mat,2,'mean'),
                                 median = object$y_mean + object$y_sd * apply(y_hat_mat,2,'median')))

  return(out)

} # end of predict function


########################################################################################################
# Predictions for classification
########################################################################################################

#' @export
predict_mybart_class = function(object,
                                   newdata,
                                   type = c('all', 'median', 'mean')) {

  # Create holder for predicted values
  n_newX = dim(newdata)[1]
  n_its = object$npost
  y_hat_mat = matrix(NA, nrow = n_its,
                     ncol = nrow(newdata))

  # Now loop through iterations and get predictions
  for (i in 1:n_its) {
    # Get current set of trees
    curr_trees = object$trees[[i]]

    # Use get_predictions function to get predictions
    y_hat_mat[i,] = get_predictions(curr_trees,
                                    newdata,
                                    single_tree = length(curr_trees) == 1)
  }

  # Sort out what to return
  out = switch(type,
               all = y_hat_mat,
               mean = apply(pnorm(y_hat_mat),2,'mean'),
               median = apply(pnorm(y_hat_mat),2,'median'))

  return(out)

} # end of predict function


#' @export
marginal_predict_mybart_class = function(object, var_marg, newdata,
                                   type = c('all', 'median', 'mean')) {

  # Create holder for predicted values
  n_newX = dim(newdata)[1]
  n_its = object$npost
  ntrees = object$ntrees
  y_hat_mat = matrix(0, nrow = n_its,
                     ncol = nrow(newdata))

  # Get which covariates are used by each tree in each MCMC iteration
  vars_trees = matrix(NA, nrow=n_its, ncol=ntrees)
  for (i in 1:n_its){
    for (j in 1:ntrees){
      aux = object$trees[[i]][[j]]$tree_matrix[,'split_variable']
      if(length(aux) > 1){
        vars_trees[i,j] = paste(unique(sort(aux[!is.na(aux)])), collapse = ',')
      }
    }
  }

  # Now loop through iterations and get predictions
  for (i in 1:n_its) {
    marginal_trees = which(vars_trees[i,] == var_marg)
    # Sometimes the trees do not contain the variable we're interested in
    if (length(marginal_trees) > 0){
      # Get current set of trees
      curr_trees = object$trees[[i]][marginal_trees]

      # Use get_predictions function to get predictions
      y_hat_mat[i,] = get_predictions(curr_trees,
                                      newdata,
                                      single_tree = length(curr_trees) == 1)

    }
  }

  # Sort out what to return
  out = list(vars_trees = vars_trees,
             prediction = switch(type,
                                 all    = pnorm(y_hat_mat),
                                 mean   = apply(pnorm(y_hat_mat),2,'mean'),
                                 median = apply(pnorm(y_hat_mat),2,'median')))

  return(out)

} # end of predict function

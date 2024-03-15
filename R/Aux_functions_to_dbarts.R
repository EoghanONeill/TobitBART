# Thanks to Dr. Vincent Dorie for kindly helping out with this!

dbarts_rebuild_tree <- function(tree.flat) {
  if (tree.flat$var[1L] == -1) return(tree.flat$value[1L])

  left <- dbarts_rebuild_tree(tree.flat[-1L,])
  if (!is.list(left)) {
    n.left <- 1L
  } else {
    n.left <- left$n
    left$n <- NULL
  }
  right <- dbarts_rebuild_tree(tree.flat[seq.int(2L + n.left, nrow(tree.flat)),])
  if (!is.list(right)) {
    n.right <- 1L
  } else {
    n.right <- right$n
    right$n <- NULL
  }

  list(var = tree.flat$var[1L], value = tree.flat$value[1L],
       left = left, right = right, n = 1L + n.left + n.right)
}

# Uses a list of lists tree to partition x and returns the
# vector of predictions.
dbarts_get_predictions <- function(tree, x) {
  if (!is.list(tree)) return(rep_len(tree, nrow(x)))

  goes_left <- x[, tree$var] <= tree$value
  predictions <- numeric(nrow(x))

  predictions[ goes_left] <- dbarts_get_predictions(tree$left,  x[ goes_left, , drop = FALSE])
  predictions[!goes_left] <- dbarts_get_predictions(tree$right, x[!goes_left, , drop = FALSE])

  predictions
}

# Uses the flatted trees to get predictions and returns them as a 2-d list,
# n_chains x n_samples.

#' @export
dbarts_marginal_prediction <- function(object, x, filter_variable){

  trees <- object$fit$getTrees()

  samples_list <- by(trees[c("tree", "var", "value")],
                     trees[c("chain", "sample")],
                     function (trees.flat) {
                       tree_predictions <- by(trees.flat[c("var", "value")], trees.flat$tree,
                                              function(tree.flat)
                                              {
                                                # If tree doesn't contain filter variable, return 0.
                                                if (all(tree.flat$var != filter_variable))
                                                  return(numeric(nrow(x)))

                                                tree <- dbarts_rebuild_tree(tree.flat)

                                                dbarts_get_predictions(tree, x)
                                              })
                       tree_predictions <- matrix(unlist(tree_predictions), nrow = nrow(x))

                       rowSums(tree_predictions)
                     })

  # Convert list
  samples <- array(unlist(samples_list),
                   c(nrow(x), dim(samples_list)[1L], dim(samples_list)[2L]))

  # permute to match stored values in bart object
  samples <- aperm(samples, c(2L, 3L, 1L))

  # undo dbarts' internal scaling
  samples <- diff(range(object$y)) * (samples + 0.5) + min(object$y)

  # Take the mean of the predicted values over the chains
  samples = apply(apply(samples, c(1,3), mean),2 ,mean)

  return(samples)
}

# f <- function(x) {
#   10 * sin(pi * x[,1] * x[,2]) + 20 * (x[,3] - 0.5)^2 +
#     10 * x[,4] + 5 * x[,5]
# }
#
# set.seed(99)
# sigma <- 1.0
# n     <- 100
#
# x  <- matrix(runif(n * 10), n, 10)
# Ey <- f(x)
# y  <- rnorm(n, Ey, sigma)
#
# set.seed(99)
# bartFit <- bart2(x, y, verbose = FALSE, keepTrees = TRUE)
# Set this to the variable you want to only require that trees contain. By
# setting it to -1L, we can double check that the method returns the correct
# result by including all trees.
# filter_variable <- -1L
# filter_variable <- 4
# marg_pred = dbarts_marginal_prediction(bartFit, x, filter_variable)

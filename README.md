
# TobitBART

<!-- badges: start -->
<!-- badges: end -->

The goal of TobitBART is to provide implementations of type I Tobit models with Bayesian Additive Regression Trees instead of linear combinations of covariates. Sums-of-trees are sampled using the dbarts package.

## Installation

You can install the development version of TobitBART like so:

``` r
library(devtools)
install_github("EoghanONeill/TobitBART")
```

## Example

This is a basic example:

``` r
library(TobitBART)
## basic example code

#example taken from https://stats.idre.ucla.edu/r/dae/tobit-models/

dat <- read.csv("https://stats.idre.ucla.edu/stat/data/tobit.csv")

train_inds <- sample(1:200,190)
test_inds <- (1:200)[-train_inds]

ytrain <- dat$apt[train_inds]
ytest <- dat$apt[test_inds]

xtrain <- cbind(dat$read, dat$math)[train_inds,]
xtest <- cbind(dat$read, dat$math)[test_inds,]

tobart_res <- tbart1(xtrain,xtest,ytrain,
                     below_cens = -Inf,
                     above_cens = 800,
                     n.iter = 400,
                     n.burnin = 100)


```


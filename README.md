
# TobitBART

<!-- badges: start -->
<!-- badges: end -->

The goal of TobitBART is to provide implementations of type 1 and type 2 Tobit models with Bayesian Additive Regression Trees (Chipman et al. 2010) instead of linear combinations of covariates. Sums-of-trees are sampled using the ``dbarts`` package.

The Type 1 Tobit implementaiton is based on Chib (1992). The Type 2 Tobit implementaiton is based on Omori (2007), and van Hasselt (2011).


The `tbart1` function runs Type 1 TOBART.

The `tbart2c` function runs Type 2 TOBART with bivariate normal errors in the selection and outcome equations. [Not tested yet]

The `tbart2np` function runs nonparametric Type 2 TOBART. The errors in the selection and outcome equations are jointly distributed as a Dirichlet Process mixture of bivariate normal distributions. [Not tested yet]




Chib, S. (1992). Bayes inference in the Tobit censored regression model. Journal of Econometrics, 51(1-2), 79-99.

Omori, Y. (2007). Efficient Gibbs sampler for Bayesian analysis of a sample selection model. Statistics & probability letters, 77(12), 1300-1311.

Van Hasselt, M. (2011). Bayesian inference in a sample selection model. Journal of Econometrics, 165(2), 221-232.




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


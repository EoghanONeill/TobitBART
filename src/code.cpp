
# include <RcppArmadillo.h>
# include <cmath>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::mat phi_app(const arma::mat& X_stand, const arma::mat& anc, const double& tau) {

  arma::vec ancsplits = anc.col(2);
  arma::vec anclefts = anc.col(3);
  arma::uvec leftinds = arma::find(anclefts);
  arma::vec ancterms = arma::unique(anc.col(0));
  arma::vec anctermcol = anc.col(0);
  arma::uvec ancvars = arma::conv_to<arma::uvec>::from(anc.col(4) - 1) ;

  arma::mat phimat(X_stand.n_rows, ancterms.n_elem);

  // Rcpp::Rcout << "Line 15. leftinds = " << leftinds  << ". \n" ;


  for(unsigned int rowind=0; rowind < X_stand.n_rows; rowind++){


    // Rcpp::Rcout << "Line 15. rowind = " << rowind  << ". \n" ;


    arma::vec xrow = (X_stand.row(rowind)).t();

    // Rcpp::Rcout << "Line 18. rowind = " << rowind  << ". \n" ;

    arma::vec xvars = xrow(ancvars);

    arma::vec input_temp = (xvars - ancsplits)/tau;
    arma::vec psi_temp = 1/(1+ arma::exp(- input_temp));


    // Rcpp::Rcout << "Line 31. rowind = " << rowind  << ". \n" ;


    //test that this gives the right numbers
    arma::vec probvec = 1 - psi_temp;


    probvec.elem(leftinds) = psi_temp.elem(leftinds);


    // Obtain products for each terminal node

    // Rcpp::Rcout << "Line 45. rowind = " << rowind  << ". \n" ;



    arma::vec retvec(ancterms.n_elem);
    // Rcpp::Rcout << "Line 51. rowind = " << rowind  << ". \n" ;

    for(unsigned int termind=0; termind < ancterms.n_elem; termind++){
      double term_temp = ancterms(termind);

      arma::uvec tempinds = arma::find(anctermcol == term_temp);

      // Rcpp::Rcout << "Line 62. tempinds = " << tempinds  << ". \n" ;
      // Rcpp::Rcout << "Line 63. term_temp = " << term_temp  << ". \n" ;
      //
      // Rcpp::Rcout << "Line 65. termind = " << termind  << ". \n" ;

      arma::vec termprobs = probvec.elem(tempinds);

      double temp_prod = arma::prod(termprobs);

      retvec(termind) = temp_prod;
    }

    // Rcpp::Rcout << "Line 65. rowind = " << rowind  << ". \n" ;


    phimat.row(rowind) = retvec.t();

  }

  return( phimat);
}




double cauchy_jacobian(double tau, double sigma_hat) {
  double sigma = std::pow(tau, -0.5);
  int give_log = 1;

  double out = R::dcauchy(sigma, 0.0, sigma_hat, give_log);
  out = out - M_LN2 - 3.0 / 2.0 * std::log(tau);

  return out;

}
//
// double update_sigma(const arma::vec& r,
//                     double sigma_hat, double sigma_old) {
//
//   double SSE = arma::as_scalar(r.t()* r) ; //dot(r,r) ;
//   double n = r.size() ;
//
//   double shape = 0.5 * n + 1.0;
//   double scale = 2.0 / SSE;
//   double sigma_prop = pow(R::rgamma(shape, scale), -0.5);
//
//   double tau_prop = pow(sigma_prop, -2.0);
//   double tau_old = pow(sigma_old, -2.0);
//
//   double loglik_rat = cauchy_jacobian(tau_prop, sigma_hat) -
//     cauchy_jacobian(tau_old, sigma_hat);
//
//   return log(unif_rand()) < loglik_rat ? sigma_prop : sigma_old;
//
// }

double update_sigma(arma::vec r, double sigma_hat, double sigma_old) {

  double SSE = arma::as_scalar(r.t() *r ) ;
  double n = (double) r.n_elem ;

  double shape = 0.5 * n + 1.0;
  double scale = 2.0 / SSE;
  double sigma_prop = std::pow(R::rgamma(shape, scale), -0.5);

  double tau_prop = std::pow(sigma_prop, -2.0);
  double tau_old = std::pow(sigma_old, -2.0);

  double loglik_rat = cauchy_jacobian(tau_prop, sigma_hat) -
    cauchy_jacobian(tau_old, sigma_hat);

  if(log(R::unif_rand()) < loglik_rat ){
    return(sigma_prop);
  }else{
    return(sigma_old);
  }
  // return log(unif_rand()) < loglik_rat ? sigma_prop : sigma_old;

}

// double UpdateSigma(arma::vec r,double sigma_hat, double sigma) {
//   sigma = update_sigma(r, sigma_hat, sigma);
//   return(sigma);
// }

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
double UpdateSigmaMu1(arma::vec means, double sigma_mu_hat, double sigma_mu) {
  sigma_mu = update_sigma(means, sigma_mu_hat, sigma_mu);
  return(sigma_mu);
}

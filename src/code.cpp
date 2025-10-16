
# include <RcppArmadillo.h>
# include <cmath>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::mat phi_app(arma::mat X_stand, arma::mat anc, double tau) {

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

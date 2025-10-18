import numpy as np
from sklearn import linear_model
import warnings

def sparse_coefficients(n_features, ind, coef, threshold):
    """
    Remove sparse coefficients below a specified threshold
    """
    c = np.zeros(n_features)
    c[ind] = coef
    big_ind = np.abs(c) >= threshold
    
    # set coefficients which are < threshold to 0
    c[~big_ind] = 0
    
    return c, big_ind

def regress(x, y, x_short, x_long, alpha, 
            fit_intercept=False, solver='auto', max_iter=100000):
    """
    Perform regression
    """
    regr = linear_model.Lasso(alpha=alpha, 
                              fit_intercept=fit_intercept, 
                              max_iter=max_iter)
    # regr = linear_model.Ridge(alpha=alpha, 
    #                           fit_intercept=fit_intercept, 
    #                           solver=solver, 
    #                           max_iter=max_iter)
    regr.fit(x, y)

    adot_estimate_short = regr.predict(x_short)
    adot_estimate_long  = regr.predict(x_long)

    return regr.coef_, regr, adot_estimate_short, adot_estimate_long

def no_change(history):
    """
    Check if the coefficient mask has changed after thresholding
    """
    this_coef = history[-1].flatten()
    
    if len(history) > 1:
        last_coef = history[-2].flatten()
    else:
        last_coef = np.zeros_like(this_coef)
    
    # returns True if all items in bool(i) == bool(j) are True
    return all(bool(i) == bool(j) for i, j in zip(this_coef, last_coef))


def sparse_regression(alpha, threshold, max_iter, 
                      a_poly_train, adot_train, adot_train_long, a_poly_test_short, a_poly_test_long, 
                      n_features, n_targets, verbose=False):
    """
    SPARSE REGRESSION
    output: adot_estimate --> estimated adot after regression fitting and threshold elimination
            coef          --> coefficients of each library function { shape: (n_targets, n_features) }
            ind           --> consits of "True" and "False" corresponding to active status of that feature
    """
    regrlist = [None] * n_targets
    coef = np.array([[0.0] * n_features] * n_targets)  # coefficiets
    ind  = np.array([[True] * n_features] * n_targets) # feature status: True --> active feature; False --> inactive feature
    # n_features_selected = np.sum(ind)                  # number of features before elimination
    adot_estimate_short = adot_train.copy()
    adot_estimate_long = adot_train_long.copy()

    """ Sparse Iteration Sequence """
    for k in range(max_iter):
        for i in range(n_targets):
            coef_i, regr, adot_i_short, adot_i_long = regress(a_poly_train[:, ind[i]], adot_train[:, i], 
                                                              a_poly_test_short[:, ind[i]], a_poly_test_long[:, ind[i]], 
                                                              alpha=alpha)

            # threshold elimination of sparse coefficients
            coef_i, ind_i = sparse_coefficients(n_features, 
                                                ind[i], 
                                                coef_i, 
                                                threshold=threshold) 

            """ Check if all features have been eliminated """
            if all(ind_i == False):
                # do not update values as all values have been eliminated
                if k == 0:
                    print(f"Warning: Sparse parameter is too large for threshold={threshold:.4f} or an unsuitable alpha={alpha:.2f} is used "
                                  f"==> All relevant features for mode={i+1} have been eliminated.")
                    print("--- Program TERMINATED! ---")
                    exit()
                else:
                    continue
            else:
                # store updated coefficients and feature_status
                coef[i] = coef_i
                ind[i]  = ind_i
                adot_estimate_short[:,i] = adot_i_short
                adot_estimate_long[:,i] = adot_i_long
                regrlist[i] = regr
                    
        # store history of coefficients
        if k == 0:
            history = [coef]
        else:
            history.append(coef)

        """ check if there is a change in feature activity """
        # if no change --> stop iterating (no more features to sparsify)
        if no_change(history):
            # could not (further) select important features
            break
    
    if verbose:
        print(f"Convergence in {k} steps.")

    return adot_estimate_short, adot_estimate_long, coef, ind, regrlist

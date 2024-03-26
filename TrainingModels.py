from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso
import numpy as np


def poly_train(t_train, a_train, regress_type, degree=2, 
               interaction_only=False, include_bias=True, 
               fit_intercept=True, max_iter=1000, alpha=1.0):

    t_train = np.tile(t_train, (1, a_train.shape[1]))
    
    poly = PolynomialFeatures(degree=degree, 
                              interaction_only=interaction_only, 
                              include_bias=include_bias)
    
    t_poly_train = poly.fit_transform(t_train)  # transform and fit time series
    
    poly.fit(a_train)                           # fit training data
    feature_names = poly.get_feature_names_out()
    
    """ Regression Fitting """
    # Linear
    if regress_type == "Linear":
        regr = LinearRegression(fit_intercept=fit_intercept)
        regr.fit(t_poly_train, a_train)         # fit polynomial data

        coeff = regr.coef_
        t_poly_test = poly.fit_transform(t_train)       # transform time series
        a_pred = regr.predict(t_poly_test)
        score = regr.score(t_poly_train, a_train)

    # Lasso
    elif regress_type == "Lasso":
        clf = Lasso(alpha=alpha, max_iter=max_iter)
        clf.fit(t_poly_train, a_train)

        coeff = clf.coef_

        t_poly_test = poly.transform(t_train)       # transform time series
        a_pred = clf.predict(t_poly_test)
        score = clf.score(t_poly_train, a_train)

    return feature_names, coeff, a_pred, score
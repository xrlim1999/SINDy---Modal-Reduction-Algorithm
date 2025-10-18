import sys
sys.path.append(".")
from utils.preprocessing_func import import_data, delete
from utils.plot_func import plot_1D, plot_compare, plot_compare_multiple, plot_score, plot_compare_multiple_xdotx
from utils.time_integrate import time_integration
from utils.STLSQ import sparse_regression

import numpy as np
from matplotlib import pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import r2_score


""" ======================================================================================================= """

"""
Pre-processing
"""

""" import data """
m, totalmodes = 367, 9

a = dict(); adot = dict()

a["velroot"]   = import_data("./csvdata/a_velocity.csv", totalmodes, m)
a["Uroot"]     = import_data("./csvdata/a_U.csv", totalmodes, m)
a["Vroot"]     = import_data("./csvdata/a_V.csv", totalmodes, m)
a["omegaroot"] = import_data("./csvdata/a_omega.csv", totalmodes, m)

adot["velroot"]   = import_data("./csvdata/adot_velocity.csv", totalmodes, m)
adot["Uroot"]     = import_data("./csvdata/adot_U.csv", totalmodes, m)
adot["Vroot"]     = import_data("./csvdata/adot_V.csv", totalmodes, m)
adot["omegaroot"] = import_data("./csvdata/adot_omega.csv", totalmodes, m)

"""
cleanup data --> delete abnormal data
"""
colnums = [0]
a["velroot"]   = delete(a["velroot"], colnums=colnums)
a["Uroot"]     = delete(a["Uroot"], colnums=colnums)
a["Vroot"]     = delete(a["Vroot"], colnums=colnums)
a["omegaroot"] = delete(a["omegaroot"], colnums=colnums)

adot["velroot"]   = delete(adot["velroot"], colnums=colnums)
adot["Uroot"]     = delete(adot["Uroot"], colnums=colnums)
adot["Vroot"]     = delete(adot["Vroot"], colnums=colnums)
adot["omegaroot"] = delete(adot["omegaroot"], colnums=colnums)

"""
truncate training data
"""
n_targets = 2  # number of training data

a["veltrain_long"]   = a["velroot"][:, 0:n_targets]
a["Utrain_long"]     = a["Uroot"][:, 0:n_targets]
a["Vtrain_long"]     = a["Vroot"][:, 0:n_targets]
a["omegatrain_long"] = a["omegaroot"][:, 0:n_targets]

adot["veltrain_long"]   = adot["velroot"][:, 0:n_targets]
adot["Utrain_long"]     = adot["Uroot"][:, 0:n_targets]
adot["Vtrain_long"]     = adot["Vroot"][:, 0:n_targets]
adot["omegatrain_long"] = adot["omegaroot"][:, 0:n_targets]

"""
define time series
"""
# totaltime = 40000 - 15000
timescale = 1 / 0.00146122449;  # time scale of simulation
totalT = (m-1) * 13 / timescale
t_train = np.linspace(0, totalT, num=a["velroot"].shape[0]).reshape(-1,1)

"""
save a copy of long time dataset
"""
a["veltrain_long"]   = a["veltrain_long"].copy()
a["Utrain_long"]     = a["Utrain_long"].copy()
a["Vtrain_long"]     = a["Vtrain_long"].copy()
a["omegatrain_long"] = a["omegatrain_long"].copy()

adot["veltrain_long"]   = adot["veltrain_long"].copy()
adot["Utrain_long"]     = adot["Utrain_long"].copy()
adot["Vtrain_long"]     = adot["Vtrain_long"].copy()
adot["omegatrain_long"] = adot["omegatrain_long"].copy()

t_train_long = t_train.copy()

"""
truncate data to 1 period
"""
# period time interval
trunc_t = int(np.ceil(4.93 * timescale / 13))      # number of iterations for truncation to 1 period
start_shift = int(2.5 * np.ceil(timescale / 13))   # iteration number of starttime
end_shift = start_shift + trunc_t                  # iteration number of endtime

# periodic data
t_train     = t_train[start_shift : end_shift, 0].reshape(-1,1) - t_train[0]

a["veltrain_period"]   = a["veltrain_long"][start_shift : end_shift, :]
a["Utrain_period"]     = a["Utrain_long"][start_shift : end_shift, :]
a["Vtrain_period"]     = a["Vtrain_long"][start_shift : end_shift, :]
a["omegatrain_period"] = a["omegatrain_long"][start_shift : end_shift, :]

adot["veltrain_period"]   = adot["veltrain_long"][start_shift : end_shift, :]
adot["Utrain_period"]     = adot["Utrain_long"][start_shift : end_shift, :]
adot["Vtrain_period"]     = adot["Vtrain_long"][start_shift : end_shift, :]
adot["omegatrain_period"] = adot["omegatrain_long"][start_shift : end_shift, :]


""" ======================================================================================================= """


"""
Polynomial Transformation
"""
# Polynomial Transformation
degree = 2                   # order of polynomial equation
interaction_only = False     # if False --> include individual a terms
include_bias = True          # if True  --> include constant terms

poly = PolynomialFeatures(degree=degree, 
                          interaction_only=interaction_only, 
                          include_bias=include_bias)

# create test dataset from a copy of training dataset
a["veltest_long"] = a["veltrain_long"]

# create dictionary for polytrain datasets
apoly = dict()

# fit and transfrom training dataset into polynomial matrix
apoly["veltrain"] = poly.fit_transform(a["veltrain_period"])

# transfrom short and long test dataset into polynomial matrix
apoly["veltest_period"] = poly.fit_transform(a["veltrain_period"])
apoly["veltest_long"]   = poly.fit_transform(a["veltrain_long"])

# Feature Names
feature_names = poly.get_feature_names_out() # extract feature names
n_features = len(feature_names)              # number of features


""" ======================================================================================================= """


"""
Sparse-Regression
output: a_estimate --> estimated a(t) using the SINDy model
"""

""" define SINDy parameters """
# 1. Penalty multiplier (alpha)
# alpha = np.linspace(0, 19, 300)
alpha = 0.2

# 2. Sparsity threshold (lambda)
# threshold = np.linspace(0.001, 0.1, 100)
threshold = 0.00000

# 3. Number of iterations to perform
max_iter = 10000

""" record RMSE """
# R2 score
score_adot = []
score_a = []

""" number of active terms (not eliminated) """
# number of active terms
activeterms = []


"""
Perform sparse-regression
"""
adot["velest_period"], adot["velest_long"], coef, ind, regrlist = sparse_regression(alpha, threshold, max_iter, 
                                                                       apoly["veltrain"], adot["veltrain_period"], adot["veltrain_long"], 
                                                                       apoly["veltest_period"], apoly["veltest_long"], 
                                                                       n_features, n_targets, verbose=False)


""" ======================================================================================================= """


"""
Time-Integration

output: a_estimate    --> estimated a(t) using the SINDy model
        adot_estimate --> estimated da/dt using the SINDy model
"""

"""
Time-integrate from initial value problem using the trained model
"""
regr_type = "RK45"

# periodic data
a["velest_period"], adot["velest_period"] = time_integration(a["veltrain_period"], adot["velest_period"], t_train_long, 
                                            poly=poly, regrlist=regrlist, ind=ind, method=regr_type)

# long-time data
a["velest_long"], adot["velest_long"] = time_integration(a["veltrain_long"], adot["velest_long"], t_train_long, 
                                        poly=poly, regrlist=regrlist, ind=ind, method=regr_type)


""" ======================================================================================================= """


"""
PLOTS
"""
# dataset indexes to plot
indexes = np.linspace(0, n_targets-1, num=n_targets).astype(int)

""" da/dt(t) against a(t) dataset """
# period
fig1, axs = plot_compare_multiple_xdotx(a["veltrain_period"], adot["veltrain_period"], adot["velest_period"], 
                                        indexes=indexes, xlabel="a", ylabel="da/dt", 
                                        extparams=" for $\u03B1$ = " + str(alpha) + " threshold = " + str(threshold) + "\n"
                                            + " with " + str(n_targets) + " latent variables and polynomial of order " + str(degree))
# long
fig2, axs = plot_compare_multiple_xdotx(a["veltrain_long"], adot["veltrain_long"], adot["velest_long"], 
                                        indexes=indexes, xlabel="a", ylabel="da/dt", 
                                        extparams=" for $\u03B1$ = " + str(alpha) + " threshold = " + str(threshold) + "\n"
                                            + " with " + str(n_targets) + " latent variables and polynomial of order " + str(degree))

""" a(t) against time dataset """
# period
fig3, axs = plot_compare_multiple(t_train, a["veltrain_period"], a["velest_period"], 
                                  indexes=indexes, xlabel="time (tU/L)", ylabel="a(t)", 
                                  extparams=" for $\u03B1$ = " + str(alpha) + " threshold = " + str(threshold) + "\n"
                                            + " with " + str(n_targets) + " latent variables and polynomial of order " + str(degree))
# long
fig4, axs = plot_compare_multiple(t_train_long, a["veltrain_long"], a["velest_long"], 
                                        indexes=indexes, xlabel="time (tU/L)", ylabel="a(t)", 
                                        extparams=" for $\u03B1$ = " + str(alpha) + " threshold = " + str(threshold) + "\n"
                                                    + " with " + str(n_targets) + " latent variables and polynomial of order " + str(degree))

plt.show()

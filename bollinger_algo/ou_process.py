#region imports
from AlgorithmImports import *
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import math
import random
#endregion

# Implementation based on QuantConnect Tutoring Project
def neg_log_likelihood(params, x):
    mu, theta, sigma = params
    sigma_sq = sigma ** 2
    n = x.shape[0]
    delta_t = 1 / (n-1)
    sigma_sq_est = sigma_sq*(1-np.exp(-2*mu*delta_t))/(2*mu)
    part = 0
    for i in range(1,n):
        sub = x[i]-x[i-1]*np.exp(-mu*delta_t)-theta*(1-np.exp(-mu*delta_t))
        part += (sub)**2
    ll = -0.5*np.log(2*np.pi)-np.log(sigma_sq_est)-(1/(2*n*sigma_sq_est))*part
    return -ll

# Actuall MLE Function
def MLE(X):
#     bounds = ((None, None), (1e-5, None), (1e-5, None))  # theta ∈ ℝ, mu > 0, sigma > 0
#                                                            # we need 1e-10 b/c scipy bounds are inclusive of 0, 
#                                                            # and sigma = 0 causes division by 0 error
    theta_init = np.mean(X)
    mu_init = 100
    sigma_init = np.std(X)
    params0 = (mu_init, theta_init, sigma_init)
    result = minimize(neg_log_likelihood, params0, args=(X,))
    mu, theta, sigma = result.x
    max_log_likelihood = -result.fun  # undo negation from __compute_log_likelihood
    # .x gets the optimized parameters, .fun gets the optimized value
    return theta, mu, sigma, max_log_likelihood

# Calculate portfolio value based on stock weight
def compute_portfolio_values(ts_A, ts_B, alloc_B):
    ts_A = ts_A.copy()  # defensive programming
    ts_B = ts_B.copy()
    
    ts_A = ts_A / ts_A[0]
    ts_B = ts_B / ts_B[0]
    
    result = ts_A - alloc_B * ts_B
    return result


# Find out optimal coefficient - beta
def arg_max_B_alloc(ts_A, ts_B):   
    theta = mu = sigma = alloc_B = 0
    max_log_likelihood = 0

    def compute_coefficients(x):
        portfolio_values = compute_portfolio_values(ts_A, ts_B, x)
        return MLE(portfolio_values)

    vectorized = np.vectorize(compute_coefficients)
    linspace = np.linspace(.01, 1, 50)
    res = vectorized(linspace)
    index = res[3].argmax()
    return res[0][index], res[1][index], res[2][index], linspace[index]

def probabilistic_forecasting(theta, mu, sigma, beta, forecasting_length = 5, epoch = 10000, c_level = 0.95):
    all_ou = []
    for i in tqdm.tqdm(range(epoch)):
        T = forecasting_length
        N = forecasting_length
        x_ou = [None for i in range(N)]
        x_ou[0] = x3[-1]
        dt = 1/(T-1)
        # theta, mu, sigma, b_alloc
        for i in range(N-1):
            x_ou[i+1] = x_ou[i] + theta*(mu - x_ou[i])*dt + sigma*norm.rvs() #loc = mean_x3, scale = std_x3
        all_ou.append(x_ou)
    # Compute the mean and standard deviation of each time step across the forecasts
    mean_forecasts = np.mean(all_ou, axis=0)
    std_forecasts = np.std(all_ou, axis=0)

    # Compute the upper and lower confidence intervals using the standard deviation
    ci_coef = -norm.ppf((1-c_level)/2)
    lower_ci = mean_forecasts - ci_coef * std_forecasts
    upper_ci = mean_forecasts + ci_coef * std_forecasts
    return mean_forecasts, upper_ci, lower_ci



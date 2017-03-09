# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.optimize import brute

class fit:
    def __init__(func_fit,func_predict,fit_dict,fit_result):
        pass

    
def objective_l2(func_predict,true_values,*args,**kwargs):
    """
    Returns the L2 (least squares) fit of the predicted values to the true,
    normalizd by (true_values)**2

    Args:
        func_predict: should take *args,**kwargs, return a list like true_values
        true_values: ground truth
        *args,**kwargs: for func_predict
    Returns:
        normalized L2 
    """
    predicted_values = func_predict(*args,**kwargs)
    values = np.abs(predicted_values-true_values)**2
    to_ret =  sum(values)/sum(true_values**2)
    return to_ret

def brute_optimize(func_to_call,true_values,brute_dict=dict()):
    """
    given a function to call, gets the brute-optimized parameter values
    
    Args:
        func_to_call: what to call, should just take in the fit parameters
        true_values: what the true results should be
        brute_dict: options for scipy.optimize.brute
    Returns:
        output of scipy.optimize
    """
    objective = lambda *args: objective_l2(func_to_call,true_values,*args)
    return brute(objective,disp=False,**brute_dict)

def brute_fit(func_to_call,true_values,func_predict=None,**kwargs):
    """
    given a function for fiting and a function for predicting, calls 
    brute_optimize and returns a fit object
    
    Args:
        func_to_call: what to call, should just take in the fit parameters
        true_values: what the true results should be
        **kwargs: see brute_optimize
    Returns:
        output of brute_optimize, wrapped to a fit object
    """
    brute_result = brute_optimize(func_to_call,true_values,**kwargs)
    return fit(func_to_call,func_predict=func_predict,fit_dit=kwargs,
               fit_result=brute_result)
    

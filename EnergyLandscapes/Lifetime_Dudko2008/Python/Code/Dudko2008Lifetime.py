# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

MACHINE_EPS = np.finfo(float).eps
from FitUtil import fit_base

def dudko_log_model(force,tau0,x_tx,DeltaG_tx,kbT,v):
    """
    Logarithm of Equation 5, intended for use in fitting, from 
    Dudko, Olga K., Gerhard Hummer, and Attila Szabo. 
    "Theory, Analysis, and Interpretation of Single-Molecule Force Spectroscopy
    Experiments."
    Proceedings of the National Academy of Sciences 105, no. 41      

    Args: 
        force: F in ibid, array of forces
        kbT: (k*T), where T is the temperature and k is boltzmann's constant
        v: XXX TODO 
        tau0: tau_0 in ibid,lifetime in the abscence of force and DeltaG=0
        x_tx: distance to transition state in ibid
        DeltaG: apparent free-eneergy of activation in the abscence of an
        external force
    returns:
        log of equation 5, ibid
    """
    # use a placeholder variable for the term that appears twice
    term =  1-(v*force*x_tx/DeltaG_tx)
    term = np.maximum(term,MACHINE_EPS)
    beta = 1./kbT
    toRet = np.log(tau0) + \
            (1-(1/v)) * np.log(term) + \
            -beta * DeltaG_tx * (1- (term)**(1/v) )
    return toRet

def dudko_model(*args,**kwargs):
    """
    Gives the actual values for the Dudko model; see DudkoLogModel

    Args:  
        *args,**kwargs: see DudkoLogModel, passed directly to it 
    Returns:
        equation 5 from cite in DudkoLogModel
    """
    return np.exp(dudko_log_model(*args,**kwargs))

def dudko_fit(force,rates,v=1/2,kbT=4.1e-21,fit_dict=dict()):
    kw = dict(v=v,kbT=kbT)
    func_to_call = lambda *args,**kwargs : dudko_log_model(force,*args,**kw)
    func_to_predict = dudko_model
    # fit to the logarithm of the rates.
    return fit_base.brute_fit(func_to_call=func_to_call,
                              true_values=np.log(rates),
                              func_predict=func_to_predict,
                              fit_dict=fit_dict,fixed_kwargs=kw)

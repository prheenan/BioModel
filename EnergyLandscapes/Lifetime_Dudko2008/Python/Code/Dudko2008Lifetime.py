# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

import FitUtils.Python.FitMain as FitMain
from FitUtils.Python.FitClasses import Initialization,BoundsObj,FitInfo,\
    GetBoundsDict
from FitUtils.Python.FitMain import Fit
from Dudko_Helper import DudkoParamValues
MACHINE_EPS = np.finfo(float).eps

def DudkoLogModel(force,tau0,v,x_tx,DeltaG_tx,kbT):
    """
    Logarithm of Equation 5, intended for use in fitting, from 
    Dudko, Olga K., Gerhard Hummer, and Attila Szabo. 
    "Theory, Analysis, and Interpretation of Single-Molecule Force Spectroscopy
    Experiments."
    Proceedings of the National Academy of Sciences 105, no. 41      


    Args: 
        force: F in ibid, array of forces
        tau0: tau_0 in ibid,lifetime in the abscence of force and DeltaG=0
        x_tx: distance to transition state in ibid
        DeltaG: apparent free-eneergy of activation in the abscence of an
        external force

        kbT: (k*T), where T is the temperature and k is boltzmann's constant
    """
    # use a placeholder variable for the term that appears twice
    term =  1-(v*force*x_tx/DeltaG_tx)
    term = np.maximum(term,MACHINE_EPS)
    beta = 1./kbT
    toRet = np.log(tau0) + \
            (1-(1/v)) * np.log(term) + \
            -beta * DeltaG_tx * (1- (term)**(1/v) )
    return toRet

def DudkoModel(*args,**kwargs):
    """
    Gives the actual values for the Dudko model; see DudkoLogModel

    Args:  
        *args,**kwargs: see DudkoLogModel, passed directly to it 
    Returns:
        equation 5 from cite in DudkoLogModel
    """
    return np.exp(DudkoLogModel(*args,**kwargs))

def DudkoFit(Force,Rates,Values,Vary=None,
             Bounds=None,Initial=None):
    if (Vary is None):
        Vary = dict(tau0=True,
                    v=False,
                    x_tx=True,
                    DeltaG_tx=True,
                    kbT=False)
    if (Bounds is None):
        kbT = Values['kbT']
        # XXX move to bounds guessing / reasonable bounds
        maxF = max(Force)
        minE = kbT/10
        maxE = kbT*20
        RawBounds = dict(tau0=[0,1e2],
                         v=[0,1],
                         x_tx=[minE/maxF,maxE/maxF],
                         DeltaG_tx=[minE,maxE],
                         kbT=[0,np.inf])
        Bounds = GetBoundsDict(**RawBounds)
    if (Initial is None):
        Initial = Initialization(Type=Initialization.BRUTE,Ns=20,finish=None)
    Model = DudkoLogModel
    mVals = DudkoParamValues(Vary=Vary,Bounds=Bounds,Values=Values)
    Options = FitInfo(FunctionToCall=Model,ParamVals=mVals,
                      Initialization=Initial,FunctionToPredict=DudkoModel)
    toRet =  FitMain.Fit(Force,np.log(Rates),Options)
    return toRet

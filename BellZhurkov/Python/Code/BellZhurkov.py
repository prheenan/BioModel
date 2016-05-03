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
from Bell_Helper import BellParamValues

def BellZhurkovLogModel(Force,beta,k0,DeltaG,DeltaX):
    """
    Logarithm of the Bell-Zhurkov model
    
    XXX fill in with woodside description

    Args:
        Force the force (x value)
        beta: 1/(k*T)
        k0: the extrapolted folding rate at zero force (XXX)
        DeltaG : The energy associated with the model at zero force
        DeltaX: the barrier distance
    """
    FoldRate = np.log(k0) -beta * ( DeltaG - Force*DeltaX)
    return FoldRate

def BellZhurkovModel(*args,**kwargs):
    return np.exp(BellZhurkovLogModel(*args,**kwargs))

def GenBellZurkovFit(Force,Rates,values,vary=None,
                     bounds=None,Initial=None):
    Vary = lambda x: x is None
    if (vary is None):
        vary = dict(beta=False,
                    k0=False,
                    DeltaG=True,
                    DeltaX=True)
    if (bounds is None):
        bounds = GetBoundsDict(beta=[0,np.inf],
                               k0=[0,np.inf],
                               DeltaG=[-np.inf,np.inf],
                               DeltaX=[-np.inf,np.inf])
    if (Initial is None):
        Initial = Initialization(Type=Initialization.HOP,disp=False,
                                 stepsize=1e-9)
    Model = BellZhurkovLogModel
    mVals = BellParamValues(Vary=vary,Bounds=bounds,Values=values)
    Options = FitInfo(FunctionToCall=Model,ParamVals=mVals,
                      Initialization=Initial,FunctionToPredict=BellZhurkovModel)
    toRet =  FitMain.Fit(Force,np.log(Rates),Options)
    return toRet



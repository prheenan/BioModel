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
from FJC_Helper import FJCValues

def Coth(x):
    """
    Returns the hyperbolic cotangent of x
    """
    return 1./np.tanh(x)

def FJCModel(force,L0,b_kuhn,S_modulus,kbT):
    """
    Freely jointed chain model, which gets the extension as a function of force.
    From the last equation in :  

    Smith, Steven B., Yujia Cui, and Carlos Bustamente. 
    "Overstretching B-DNA: The Elastic Response of Individual Double-Stranded 
    and Single-Stranded DNA Molecules."
    Science 271, no. 5250 (February 9, 1996): 795.

    Args:
        force: force, in units of kbT/b_kuhn
        L0: the contour length, in units of b_kuhn (or equivalently, extension)
        b_kuhn: the kuhn length, units of extension
        S_modulus: the stretch modulus, units of force
        kbT : the boltmann constant times temperature. Room temp: 4.1 pN . nm
    """
    gamma = (force * b_kuhn) / (kbT)
    return L0 * (Coth(gamma) - 1/gamma) * (1+force/S_modulus)

def FreelyJointedChainFit(Extensions,Force,Values,Vary=None,
                          Bounds=None,Initial=None):
    if (Vary is None):
        Vary = dict(L0=True,
                    b_kuhn=False,
                    S_modulus=False,
                    kbT=False)
    if (Bounds is None):
        Bounds = GetBoundsDict(L0=[0,np.inf],
                               b_kuhn=[0,np.inf],
                               S_modulus=[0,np.inf],
                               kbT=[0,np.inf])
    if (Initial is None):
        Initial = Initialization(Type=Initialization.GUESS,disp=False,
                                 stepsize=1e-9)
    Model = FJCModel
    mVals = FJCValues(Vary=Vary,Bounds=Bounds,Values=Values)
    Options = FitInfo(FunctionToCall=Model,ParamVals=mVals,
                      Initialization=Initial)
    toRet =  FitMain.Fit(Force,Extensions,Options)
    return toRet



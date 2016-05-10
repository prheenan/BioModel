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


# cutoff foudn empircally for mathemtica for the LangevinSmall function
cutoff_for_ppm = 0.2

def Langevin(x,cutoff=cutoff_for_ppm):
    """
    Returns the Langevin funciton of x  (see 
    en.wikipedia.org/wiki/Brillouin_and_Langevin_functions#Langevin_function )

    Does this in two steps: simply uses the formula for x>cutoff, but for
    x < cutoff, uses LangevinSmall, which has proven (small) error properties
    at x<cutoff

    Args:
        x: values we want the langevin function of. 
        cutoff: where to start using the small-x approximation. Recommend
        not changing this
    """
    toRet = np.zeros(x.size)
    # fill in the small-x portions of the array
    idxSmall = np.where(x <= cutoff)
    toRet[idxSmall] = LangevinSmall(x[idxSmall])
    # for the large-x portions of the array, just use the forumla
    idxLarge = np.where(x > cutoff)
    xLarge = x[idxLarge]
    toRet[idxLarge] = 1./np.tanh(xLarge) - 1./xLarge
    # and... small and large parts are done
    return toRet

def LangevinSmall(x):
    """
    Approximate the Langevin function at small x, up to order n in a taylor 
    expansion in x.

    This is good to a part in a million relative error for x<0.2

    The coefficients come from 'FreelyJointedChain/Mathematica/FJC_Docs.nb' 
    (and/or wikipedia 
    en.wikipedia.org/wiki/Brillouin_and_Langevin_functions#Langevin_function)

    Necessary because 1/x is numerically ill-defined for very small x

    Args:
        x: value to feed into Langevin function
    Returns:
        the small-x approximation to the langevin function 
    """
    coeffs = [2/93555,-1/4725,0,2/945,0,-1/45,0,1/3,0]
    return np.polyval(coeffs,x)

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
    return L0 * Langevin(gamma) * (1+force/S_modulus)

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



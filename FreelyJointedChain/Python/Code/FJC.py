# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys
from FitUtil import fit_base
from scipy.interpolate import interp1d


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

def fjc_extension(F,L0,Lp,kbT,K0):
    """
    Freely jointed chain model, which gets the extension as a function of force.
    From the last equation in :  

    Smith, Steven B., Yujia Cui, and Carlos Bustamente. 
    "Overstretching B-DNA: The Elastic Response of Individual Double-Stranded 
    and Single-Stranded DNA Molecules."
    Science 271, no. 5250 (February 9, 1996): 795.
    
    also:
    
    Wang, M. D., Yin, H., Landick, R., Gelles, J. & Block, S. M. 
    Stretching DNA with optical tweezers. Biophys J 72, 1335-1346 (1997)

    Args:
        force: force, in units of kbT/b_kuhn
        L0: the contour length, in units of b_kuhn (or equivalently, extension)
        Lp: the persistance length, units of extension
        K0: the stretch modulus, units of force
        kbT : the boltmann constant times temperature. Room temp: 4.1 pN . nm
    """
    gamma = (2*F * Lp) / (kbT)
    return L0 * Langevin(gamma) * (1+F/K0)
    
def fjc_predicted_force_at_ext(separation,force,*args,**kwargs):
    ext_modelled,force_modelled = fjc_model_ext_and_force(force,*args,**kwargs)
    # interpolate back onto the grid we care about 
    interp = interp1d(ext_modelled,force_modelled,bounds_error=False,
                      fill_value="extrapolate")
    predicted_force = interp(separation)
    return predicted_force

    
def fjc_model_force(F,*args,**kwargs):
    force_range = [min(F),max(F)]
    return np.linspace(*force_range,num=F.size)
    
def fjc_model_ext_and_force(F,*args,**kwargs):
    force_modelled = fjc_model_force(F,*args,**kwargs)
    ext_modelled = fjc_extension(force_modelled,*args,**kwargs)
    return ext_modelled,force_modelled

def fit_fjc_contour(separation,force,brute_dict,**kwargs):
    func = lambda *args: \
        fjc_predicted_force_at_ext(separation,force,*args,**kwargs)
    x0 = fit_base.brute_optimize(func,force,
                                 brute_dict=brute_dict)
    model_x, model_y = fjc_model_ext_and_force(force,*x0,**kwargs)
    min_sep = min(separation)
    idx = np.where(model_x >= min_sep)
    return x0,model_x[idx],model_y[idx]



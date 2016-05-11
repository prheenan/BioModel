# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

from FitUtils.Python import FitClasses

class DudkoParamValues(FitClasses.ParamValues):
    """
    Class to record parameter values given to a fit or gotten from the same
    """
    def __init__(self,**kwargs):
        """
        Args:
            **kwargs: see FitClasses.ParamValues
        """
        super(DudkoParamValues,self).__init__(**kwargs)
    def InitParams(self):
        """
        Initiliaze parameters...
        """
        Params = ["tau0",
                  "v",
                  "x_tx",
                  "DeltaG_tx",
                  "kbT"]
        return Params
                

def GetDudkoIntegral(probabilities,forces,loads):
    """
    Returns equation 10 from Dudko2008

    Args:
        probabilities: array, element i is the probabilities for a rupture in 
        force bin forces[i]
     
        forces: array, element [i] is force bin i
        loads: array, list of loading factors for each bin
    Returns:
        tuple of <tau,forces> where each is a list representing the non-zero
        lifetimes obtained from equation 10.
    """
    DeltaF = np.median(np.diff(forces))
    h = probabilities/DeltaF
    WhereValid = np.where(h > 0)
    h = h[WhereValid]
    tauAtK = lambda k: (DeltaF/loads[k]) * (h[k]/2 + sum(h[k+1:]))/(h[k])
    tau = np.array([tauAtK(k) for k in range(len(h))])
    forces = forces[WhereValid] + DeltaF/2
    return tau,forces
    
def GetTimeIntegral(probabilities,forces,loads):
    """
    Getting the lifetimes by equation 2, from Dudko2008. I do *not* use
    the method they suggest (10), since we can do the numerical integral a 
    little better by using trapezoids

    Args:
        probabilities: array of probabilites; element [i] is probablity 
        to rupture at force forces[i]. y axis of (e.g) Dudko2008 Figure 1a

        forces: array of rupture 'forces' or similiar (e.g. x axis of ibid
        is voltage). Element [i] corresponds to rupture associated with probs[i]

        Load: array of 'loading rate' (units of forces/time), e.g. label
        of ibid 
    """
    lifetimes = [np.trapz(y=probabilities[i:]/(probabilities[i]*loads[i]),
                          x=forces[i:])
                 for i in range(probabilities.size)]
    return lifetimes

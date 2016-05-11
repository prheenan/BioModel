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

# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

from FitUtils.Python import FitClasses

class BellParamValues(FitClasses.ParamValues):
    """
    Class to record parameter values given to a fit or gotten from the same
    """
    def __init__(self,**kwargs):
        """
        Args:
            **kwargs: see FitClasses.ParamValues
        """
        super(BellParamValues,self).__init__(**kwargs)
    def InitParams(self):
        """
        Initiliaze parameters...
        """
        Params = ["beta",
                  "k0",
                  "DeltaG",
                  "DeltaX",]
        return Params
    def Scale(self,x,y):
        """
        Scales the variables to x and y (Force and rate) limits
        """
        return dict(k0=1,
                    DeltaX=1,
                    DeltaG=1,
                    beta=1)
                


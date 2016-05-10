# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

from FitUtils.Python import FitClasses

class FJCValues(FitClasses.ParamValues):
    """
    Class to record parameter values given to a fit or gotten from the same
    """
    def __init__(self,**kwargs):
        """
        Args:
            **kwargs: see FitClasses.ParamValues
        """
        super(FJCValues,self).__init__(**kwargs)
    def InitParams(self):
        """
        Initiliaze parameters...
        """
        Params = ["L0",
                  "b_kuhn",
                  "S_modulus",
                  "kbT"]
        return Params
    def Scale(self,x,y):
        """
        Scales the variables to x and y limits
        """
        return dict(L0=x,
                    b_kuhn=x,
                    S_modulus=y,
                    kbT=x*y)
                


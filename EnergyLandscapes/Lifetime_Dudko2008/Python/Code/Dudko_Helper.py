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
                


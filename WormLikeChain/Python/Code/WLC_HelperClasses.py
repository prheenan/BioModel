# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys


from collections import OrderedDict
from FitUtil.FitUtils.Python import FitClasses

class WLC_DEF:
    """
    Class defining defaults for inputs. 

    From "Estimating the Persistence Length of a Worm-Like Chain Molecule ..."
    C. Bouchiat, M.D. Wang, et al.
    Biophysical Journal Volume 76, Issue 1, January 1999, Pages 409-413

    See Wang, 1997.
    """
    # Default Dictionary
    ValueDictionary = OrderedDict(L0 = 1317.52e-9, # meters
                                  Lp = 40.6e-9, # meters
                                  K0 = 1318e-12, # Newtons
                                  kbT = 4.11e-21,
                                  F=0) # 4.1 pN * nm = 4.1e-21 N*m
    # write down the default bounds; just positive values parameter
    BoundsDictionary = OrderedDict(L0=FitClasses.BoundsObj(0,np.inf),
                                   Lp=FitClasses.BoundsObj(0,np.inf),
                                   K0=FitClasses.BoundsObj(0,np.inf),
                                   kbT=FitClasses.BoundsObj(0,np.inf),
                                   F=FitClasses.BoundsObj(0,np.inf))
    # default varying dictionary
    VaryDictionary =  OrderedDict(L0=True,
                                  Lp=False,
                                  K0=False,
                                  kbT=False,
                                  F=False)

class WLC_MODELS:
    """
    Class definining valid models.
    """
    EXTENSIBLE_WANG_1997 = 0
    INEXTENSIBLE_BOUICHAT_1999 = 1
    EXTENSIBLE_BY_INVERSE_WANG_1997 = 2


class WlcParamValues(FitClasses.ParamValues):
    """
    Class to record parameter values given to a fit or gotten from the same
    """
    def __init__(self,**kwargs):
        """
        Args:
             **kwargs: see FitClasses.ParamValues
        """
        super(WlcParamValues,self).__init__(**kwargs)
    def InitParams(self):
        """
        Initiliaze parameters names, as written in functions
        """
        Params = ["L0",
                  "Lp",
                  "K0",
                  "kbT",
                  "F"]
        return Params
    def Scale(self,xScale,ForceScale):
        """
        Scales the data to an x and y scale given by xScale and ForceScale.
        
        In other words, x -> x/xScale, etc

        Args:
            xScale: What to divide the distance parts by
            yScale: What to divide the force parts by
        """
        # lengths are distances
        Scales = dict(L0=xScale,
                      Lp=xScale,
                      K0=ForceScale,
                      kbT=ForceScale*xScale,
                      F=ForceScale)
        return Scales

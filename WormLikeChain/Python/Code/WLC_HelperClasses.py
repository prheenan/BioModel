# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys


from collections import OrderedDict
from FitUtils.Python import FitClasses
MACHINE_EPSILON = np.finfo(float).eps


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
                                  kbT = 4.11e-21) # 4.1 pN * nm = 4.1e-21 N*m
    # write down the default bounds; just positive values parameter
    BoundsDictionary = OrderedDict(L0=FitClasses.BoundsObj(0,np.inf),
                                   Lp=FitClasses.BoundsObj(0,np.inf),
                                   K0=FitClasses.BoundsObj(0,np.inf),
                                   kbT=FitClasses.BoundsObj(0,np.inf))
    # default varying dictionary
    VaryDictionary =  OrderedDict(L0=True,
                                  Lp=False,
                                  K0=False,
                                  kbT=False)

class WLC_MODELS:
    """
    Class definining valid models.
    """
    EXTENSIBLE_WANG_1997 = 0
    INEXTENSIBLE_BOUICHAT_1999 = 1


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
        Initiliaze parameters...
        """
        Params = [FitClasses.Param(Name="L0"),
                  FitClasses.Param(Name="Lp"),
                  FitClasses.Param(Name="K0"),
                  FitClasses.Param(Name="kbT")]
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
                      kbT=ForceScale*xScale)
        return Scales

def BouchiatPolyCoeffs():
    """
    Gives the polynomial correction coefficients

    See 
    "Estimating the Persistence Length of a Worm-Like Chain Molecule ..."
    C. Bouchiat, M.D. Wang, et al.
    Biophysical Journal Volume 76, Issue 1, January 1999, Pages 409-413
web.mit.edu/cortiz/www/3.052/3.052CourseReader/38_BouchiatBiophysicalJ1999.pdf

    Returns:
       list of coefficients; element [i] is the coefficient of term x^(i) in the
       correction listed above
    """
    return [0,
            0,
            -.5164228,
            -2.737418,
            16.07497,
            -38.87607,
            39.49949,
            -14.17718]

def GetReasonableBounds(ext,force,
                        c_L0_lower=0.8,c_L0_upper=1.1,
                        c_Lp_lower=0.0,c_Lp_upper=0.1,
                        c_K0_lower=10,c_K0_upper=1e4):
    """
    Returns a reasonable (ordered) dictionary of bounds, given extensions and 
    force

    Args:
        ext: the extesions we are interested in
        force: the force we are interestd in 
        c_<xx>_lower: lower bound, in terms of the max of ext/force (depending
        on which constant, K0 is Force, Lp and L0 are length)

        c_<xx>_upper: upper bound, in terms of the max of ext/force (depending
        on which constant, K0 is Force, Lp and L0 are length)
    Returns:
        Dictionary of <Parameter Name : Bounds> Pairs
    """
    MaxX = max(ext)
    MaxForce = max(force)
    TupleL0 = np.array([c_L0_lower,c_L0_upper]) * MaxX
    TupleLp = np.array([c_Lp_lower,c_Lp_upper]) * MaxX
    TupleK0 = np.array([c_K0_lower,c_K0_upper]) * MaxForce
    return FitClasses.GetBoundsDict(L0=TupleL0,
                                    Lp=TupleLp,
                                    K0=TupleK0,
                                    # we typically dont fit temperature,
                                    # really no way to know.
                                    kbT=[0,np.inf])

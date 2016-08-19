# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

import warnings
import copy
import FitUtil.FitUtils.Python.FitUtil as FitUtil

from collections import OrderedDict
from FitUtil.FitUtils.Python.FitMain import Fit
from FitUtil.FitUtils.Python.FitClasses import\
    Initialization,BoundsObj,FitInfo,FitOpt


from WLC_ComplexValued_Fit import InvertedWlcForce
from WLC_HelperClasses import WlcParamValues,WLC_MODELS,WLC_DEF
from WLC_Utils import BouchiatPolyCoeffs,\
    GetReasonableBounds,MACHINE_EPSILON
from WLC_Utils import WlcExtensible_Helper,WlcNonExtensible,WlcPolyCorrect

from WLC_ComplexValued_Fit import InvertedWlcForce

def InitializeParamVals(model,toVary,Force,Values=None,Bounds=None,
                        InitialObj=None):
    """
    Adapter to go to the general fitting method
    """
    function = GetFunctionFromModel(model)
    # a little extra work (TODO: kludgey) for inverse wang, since
    # it needs the force too...
    if (Values is None):
        Values = WLC_DEF.ValueDictionary
    if (Bounds is None):
        Bounds = WLC_DEF.BoundsDictionary
    if (InitialObj is None):
        InitialObj = Initialization()
    Values["F"] = Force
    mVals = WlcParamValues(Vary=toVary,Bounds=Bounds,Values=Values)
    return FitInfo(FunctionToCall=function,ParamVals=mVals,
                   Initialization=InitialObj,
                   FitOptions=FitOpt(Normalize=False))

def NonExtensibleWlcFit(ext,force,VaryL0=True,VaryLp=False,**kwargs):
    """
    Non-extensible version of the WLC fit. By default, varies the contour length
    to get the fit. Uses Bouichat, 1999 (see aboce) , by default

    Args:
        ext,force : see WlcFit
        VaryL0,VaryLp : see WlcParamsToVary, boolean if we should vary
        **kwargs: passed directly to InitializeParamVals 
        (eg: initial guesses,bounds)
    Returns:
        see WlcFit
    """
    toVary = dict(L0=VaryL0,Lp=VaryLp,K0=False,kbT=False)
    mInfo = InitializeParamVals(WLC_MODELS.INEXTENSIBLE_BOUICHAT_1999,
                                Force=force,
                                toVary=toVary,**kwargs)
    # call the fitter
    return Fit(ext,force,mInfo)

def GetFunctionFromModel(model):
    """
    Given a model, gets the funciton to call

    Args:
        model: valid element from WLC_MODELS
    """
    
    ModelToFunc = \
        dict( [(WLC_MODELS.INEXTENSIBLE_BOUICHAT_1999,WlcNonExtensible),
               (WLC_MODELS.EXTENSIBLE_BY_INVERSE_WANG_1997,InvertedWlcForce),
               (WLC_MODELS.EXTENSIBLE_ODJIK_1995,InvertedWlcForce)
           ])
    return ModelToFunc[model]

def ExtensibleWlcFit(ext,force,VaryL0=True,VaryLp=False,VaryK0=False,
                     **kwargs):
    """
    extensible version of the WLC fit. By default, varies the contour length
    to get the fit. Uses Bouichat, 1999 (see aboce) , by default

    Args:
        ext,force : see WlcFit
        VaryL0,VaryLp : see WlcParamsToVary, boolean if we should vary
        **kwargs: passed directly to InitializeParamVals 
        (eg: initial guesses,bounds)
    Returns:
        see WlcFit
    """
    toVary = dict(L0=VaryL0,Lp=VaryLp,K0=VaryK0,kbT=False)
    mInfo = InitializeParamVals(WLC_MODELS.EXTENSIBLE_BY_INVERSE_WANG_1997,
                                Force=force,
                                toVary=toVary,**kwargs)
    return Fit(ext,force,mInfo)

def BoundedWlcFit(ext,force,VaryL0=True,VaryLp=False,VaryK0=False,
                  Ns=100,Bounds=None,finish=None,**kwargs):
    """
    Uses a Brute-force method to get a coase-grained grid on the data
    before using a fine-grained local minimizer to find the best solution.

    Args:
        ext,force, Vary<XX>: see ExtensibleWlcFit
        Ns: size of the grid on a side. For example, if a 2-D problem (2
        parameters are being varied), and Ns=50, then the grid is a 50x50 
        search over the bounds

        Bounds: A boundsobject, formatted like 
        WLC_HelperClass.GetReasonableBounds. If none, we create one
        based on the data and kwargs

        **kwargs: passed directly to the Bounds object, if we create one
    """
    if (Bounds is None):
        Bounds = GetReasonableBounds(ext,force,**kwargs)
    InitialObj = Initialization(Type=Initialization.BRUTE,Ns=Ns,finish=finish)
    toVary = dict(L0=VaryL0,Lp=VaryLp,K0=VaryK0,kbT=False)
    mInfo = InitializeParamVals(WLC_MODELS.EXTENSIBLE_BY_INVERSE_WANG_1997,
                                Force=force,
                                Bounds=Bounds,
                                toVary=toVary,InitialObj=InitialObj)
    return Fit(ext,force,mInfo)
                  
        

# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

from FitUtil.FitUtils.Python.FitMain import Fit
from WLC_HelperClasses import GetReasonableBounds
from FitUtil.FitUtils.Python.FitClasses import\
    Initialization,BoundsObj,FitInfo,FitOpt
import WLC_Fit
from WLC_HelperClasses import WlcParamValues,BouchiatPolyCoeffs,\
    GetReasonableBounds
from WLC_HelperClasses import WLC_MODELS,WLC_DEF,MACHINE_EPSILON



def SafeCast(x):
    if (type(x) is not int) and (type(x) is not float):
        return np.array(list(x)).astype(np.complex128)
    else:
        return complex(x)

def Power(x,y):
    return np.power(SafeCast(x),SafeCast(y))

def Complex(r,i):
    return r + (1j * i)

def Sqrt(x):
    return np.sqrt(SafeCast(x))

def ExtensionPerForce(kbT,Lp,L0,K0,F):
    """
    This function takes in the WLC parameters and the force, and returns the 
    predicted extension. It does *not* account for the seventh-order Bouchiat
    Prediction, but should be a solid guess. 
    """
    ToRet = L0*(F/K0 - (-9*kbT - 4*F*Lp)/(12.*kbT) + 
     ((1 + Complex(0,1)*Sqrt(3))*(-9*Power(kbT,2) + 24*F*kbT*Lp - 
          16*Power(F,2)*Power(Lp,2)))/
      (24.*kbT*Power(-243*Power(kbT,3) + 108*F*Power(kbT,2)*Lp - 
          144*Power(F,2)*kbT*Power(Lp,2) + 64*Power(F,3)*Power(Lp,3) + 
          12*Sqrt(3)*Sqrt(135*Power(kbT,6) - 108*F*Power(kbT,5)*Lp + 
             144*Power(F,2)*Power(kbT,4)*Power(Lp,2) - 
             64*Power(F,3)*Power(kbT,3)*Power(Lp,3)),0.3333333333333333)) - 
     ((1 - Complex(0,1)*Sqrt(3))*Power(-243*Power(kbT,3) + 108*F*Power(kbT,2)*Lp - 
          144*Power(F,2)*kbT*Power(Lp,2) + 64*Power(F,3)*Power(Lp,3) + 
          12*Sqrt(3)*Sqrt(135*Power(kbT,6) - 108*F*Power(kbT,5)*Lp + 
             144*Power(F,2)*Power(kbT,4)*Power(Lp,2) - 
             64*Power(F,3)*Power(kbT,3)*Power(Lp,3)),0.3333333333333333))/(24.*kbT))
    # can have complex extension; we just want the real part. 
    return np.real(ToRet)

def FitExtensionByForce(Force,kbT,Lp,L0,K0,ext):
    print("aqui")
    ext = ExtensionPerForce(kbT,Lp,L0,K0,Force)
    return ext

def GriddedWLCFitByInverse(ext,force,VaryL0=True,VaryLp=False,VaryK0=False,
                           Ns=100,Bounds=None,finish=None,**kwargs):
    """
    Uses a Brute-force method to get a coase-grained grid on the data
    before using a fine-grained local minimizer to find the best solution.

    Args:
        Args: see WLC_Fit.BoundedWlcFit
    """
    if (Bounds is None):
        Bounds = GetReasonableBounds(ext,force,**kwargs)
    InitialObj = Initialization(Type=Initialization.BRUTE,Ns=Ns,finish=finish)
    toVary = dict(L0=VaryL0,Lp=VaryLp,K0=VaryK0,kbT=False)
    mInfo = WLC_Fit.InitializeParamVals(WLC_MODELS.EXTENSIBLE_WANG_1997,
                                        Bounds=Bounds,
                                        toVary=toVary,InitialObj=InitialObj)
    mInfo.FunctionToCall = FitExtensionByForce
    # note we switch the values (ordering) of x and y in this funciton call,
    # since we are doing an implicit fit 
    ExtensionFit = Fit(force,ext,mInfo)
    PredictedExt = ExtensionFit.Predict(force)
    plt.plot(PredictedExt,force)
    plt.plot(ext,force)
    plt.show()
    


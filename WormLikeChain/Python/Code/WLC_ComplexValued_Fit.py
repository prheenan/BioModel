# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys


from FitUtil.FitUtils.Python.FitClasses import\
    Initialization,BoundsObj,FitInfo,FitOpt
from WLC_Utils import WlcExtensible_Helper,WlcNonExtensible,WlcPolyCorrect,\
    GetReasonableBounds
from scipy.interpolate import interp1d


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
    Prediction, but should be a solid guess to start with. 
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
    # mathematically, can have complex extension; we just want the real part.
    # this turns out to converge well for what we care about
    return np.real(ToRet)

def SeventhOrderForceAndExtGrid(ext,kbT,Lp,L0,K0,F):
    """
    Given extension data, parameters, and a force, creates a WLC-based 
    grid, including Bouchiat polynomials. This is essentially the (smooth)
    fit to the data.

    Args:
        
    """
    # grid the force uniformly (x vs F is essentially a 1-1 map)
    N = F.size
    UpSample = 10
    ForceGrid = np.linspace(start=min(F),stop=max(F),num=N*UpSample)
    # get the extension predictions on the uniform grid (essentially
    # this is a detailed 1-1 map of how to go between force and ext)
    ExtPred = ExtensionPerForce(kbT,Lp,L0,K0,ForceGrid)
    # use the grided extensions and forces as a prediction for the
    # seventh-order Bouchiat corrections. This should be quite 'smooth'
    # and devoid of any noise or singularities. 
    ForceGrid = WlcExtensible_Helper(ExtPred,kbT,Lp,L0,K0,ForceGrid)
    # return the extension and force on the *grid*. These are smooth, and
    # (for fitting purposes) should be interpolated back to the original
    # extension / force data
    return ExtPred,ForceGrid

def IntepolateForceFromGriddedToDataExtensions(ExtGrid,ForceGrid,ExtActual,
                                               bounds_error=False,
                                               kind="linear",
                                               fill_value="extrapolate"):
    """
    interpolate the force from the predicted extension grid to the actual
    extensions -- which is what we care about

    Note: by default this linearly extrapolates for edge cases.
    Considering the WLC is linear at the start (hooke) and the end 
    (stretching backbone), this is probably 
    ok, but the user should be sure this behavior is desired

    Args:
        ExtGrid,ForceGrid: two arrays of the same length; we interpolate
        along the extension grid. Probably the outputs of 
        SeventhOrderForceAndExtGrid

        ExtActual: whatever grid we want 
    """
    IntepolationMap = interp1d(ExtGrid,ForceGrid,bounds_error=bounds_error,
                               kind=kind,fill_value=fill_value)
    # do the actual mapping 
    return IntepolationMap(ExtActual)


def InvertedWlcForce(ext,F,kbT,Lp,L0,K0):
    """
    Function to fit F vs ext using an ext(F). This allows us to get a 
    good initial guess for F(ext). The force is gridded, giving 
    a smooth interpolation. 

    Args:
        ext: the extension data, size N
        F: the force data, size N 
        others: See WLC_Fit.WlcNonExtensible
    """
    ExtPred,ForceGrid = SeventhOrderForceAndExtGrid(ext,kbT,Lp,L0,K0,F)
    return IntepolateForceFromGriddedToDataExtensions(ExtPred,ForceGrid,ext)

    

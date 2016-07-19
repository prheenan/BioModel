# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

MACHINE_EPSILON = np.finfo(float).eps

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


def WlcPolyCorrect(kbT,Lp,lRaw):
    """
    From "Estimating the Persistence Length of a Worm-Like Chain Molecule ..."
    C. Bouchiat, M.D. Wang, et al.
    Biophysical Journal Volume 76, Issue 1, January 1999, Pages 409-413

web.mit.edu/cortiz/www/3.052/3.052CourseReader/38_BouchiatBiophysicalJ1999.pdf

    Args:
        kbT : the thermal energy in units of [ForceOutput]/Lp
        Lp  : the persisence length, sensible units of length
        lRaw   : is either extension/Contour=z/L0 Length (inextensible) or   
        z/L0 - F/K0, where f is the force and K0 is the bulk modulus. See 
        Bouchiat, 1999 equation 13
    Returns:
        Model-predicted value for the force
    """
    # parameters taken from paper cited above
    l = lRaw.copy()
    a0=0 
    a1=0
    a2=-.5164228
    a3=-2.737418
    a4=16.07497
    a5=-38.87607
    a6=39.49949
    a7=-14.17718
    #http://docs.scipy.org/doc/numpy/reference/generated/numpy.polyval.html
    #If p is of length N, this function returns the value:a
    # p[0]*x**(N-1) + p[1]*x**(N-2) + ... + p[N-2]*x + p[N-1]
    # note: a0 and a1 are zero, including them for easy of use of polyval.
    # see especially equation 13. Note we reverse the Bouchiat Coefficients...
    polyValCoeffs = BouchiatPolyCoeffs()[::-1]
    denom = (1-l)**2
    inner = 1/(4*denom) -1/4 + l + np.polyval(polyValCoeffs,l)
    toRet = (kbT/Lp) * inner
    return toRet

def WlcNonExtensible(ext,kbT,Lp,L0,*args,**kwargs):
    """
    Gets the non-extensible model for WLC polymers, given and extension

    Args:
        kbT : see WlcPolyCorrect
        Lp: see WlcPolyCorrect
        L0 : contour length, units of ext
        ext: the extension, same units as L0
        *args: extra arguments, ignored (so we can all use the same call sig)
    Returns:
        see WlcPolyCorrect
    """
    return WlcPolyCorrect(kbT,Lp,ext/L0)

def WlcExtensible_Helper(ext,kbT,Lp,L0,K0,ForceGuess):
    """
    Fits to the (recursively defined) extensible model. 

    Args: 
        kbT,Lp,L0,ext : see WlcPolyCorrect
        K0: bulk modulus, units of Force
        ForceGuess: the 'guess' for the force
    Returns:
        see WlcPolyCorrect
    """
    # get the non-extensible model
    xNorm = ext/L0
    yNorm = ForceGuess/K0
    # I follow convention of Wang 1997, in terms of 'l' or 'z'
    # which is defined as L/L0-F/K0
    l = xNorm-yNorm
    return WlcPolyCorrect(kbT,Lp,l)

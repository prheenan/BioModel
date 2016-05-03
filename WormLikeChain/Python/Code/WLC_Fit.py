# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

import warnings
import copy
import FitUtils.Python.FitUtil as FitUtil

from collections import OrderedDict
from FitUtils.Python.FitClasses import Initialization,BoundsObj,FitInfo,\
    FitOptions
from FitUtils.Python.FitMain import Fit


from WLC_HelperClasses import WlcParamValues,BouchiatPolyCoeffs,\
    GetReasonableBounds
from WLC_HelperClasses import WLC_MODELS,WLC_DEF,MACHINE_EPSILON


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
    Fits to the (recursively defined) extensible model. Note this will need to
    be called several times to converge properly

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
    l = xNorm-yNorm
    return WlcPolyCorrect(kbT,Lp,l)

def DebugExtensibleConvergence(extOrig,yOrig,extNow,yNow,ext,
                               extrapX=None,extrapY=None):
    """
    Useful for plotting the convergence of the extensible model

    Args:
        extOrig; original extension used for non-extensible WLC
        yOrig; original force, from non-extensible WLC
        extNow: the (presummmably longer) extension used with the extensible 
        model

        yNow: the (presummmably longer) force used with the extensible 
        model
      
        ext: the actual extension, in its full glory (not truncated)
        extrapX: if present, the extrapolated x values
        extrapY: if present, the extrapolated y values
    """
    plt.plot(extOrig,yOrig,'k--',linewidth=2.5,label="Non-Extensible")
    plt.plot(extNow,yNow,'b-.',label="Extensible")
    plt.xlabel("Extension (au)")
    plt.ylabel("Force (au)")
    # make a sensible range for the plotting
    minV = min(ext)
    maxV = max(ext)
    rangeV = maxV-minV
    fudge = rangeV/100
    xRange = [minV-fudge,maxV+fudge]
    plt.xlim(xRange)
    # do the same...
    minY = min(min(yOrig),min(yNow))
    maxY = max(max(yOrig),max(yNow))
    rangeY = maxY-minY
    fudgeY = rangeY/100
    if ((extrapX is not None) and (extrapY is not None)):
        plt.plot(extrapX,extrapY,'g-',linewidth=3,label="Extrapolated")
    plt.ylim([-fudgeY,maxY+fudgeY])
    plt.legend(loc='upper left')
    plt.show()


def ExtrapolateExtensible(nToAdd,ext,extOrig,yOrig,xToFit,y,func,degree,
                          DebugConvergence=False,**kwargs):
    n = ext.size
    maxIdx = y.size
    nLeft = (n-maxIdx+1)
    factor = int(np.ceil(nLeft/nToAdd))
    for i in range(1,factor+1):
        # for the last iteration, we may want to extend just a little.
        # this avoids re-calculating, which can cause problems
        diff = n-y.size
        if (diff < 2*nToAdd):
            # then we are between 1 and 2 persistence lengths, at the
            # very end. things are likely very linear, so just 'double'
            # (worst case)
            nToAdd = diff
        # fit a taylor series to the last nToAdd points
        xExtrap = xToFit[-nToAdd:]
        yExtrap = y[-nToAdd:]
        idxInfinite = np.where(~np.isfinite(yExtrap))[0]
        if (idxInfinite.size):
            # record infinities all along
            toRet = np.empty(n)
            toRet[:y.size] = y
            toRet[y.size:] = np.inf
            y = toRet
            break
        taylor = FitUtil.TaylorSeries(xExtrap,yExtrap,deg=degree)
        # get the new x and y
        sliceV = slice(0,maxIdx+nToAdd*i,1)
        xToFit = ext[sliceV]
        prev = np.zeros(xToFit.size)
        # extrapolate the previous fit out just a smidge
        newX = xToFit[-nToAdd:]
        fitted =  np.polyval(taylor,newX)
        # add in the old solution
        prev[:y.size] = y
        # tack on the new
        prev[-nToAdd:] = fitted
        # fit everything to the extensible model again.
        y = func(xToFit,ForceGuess=prev,**kwargs)
        if (DebugConvergence):
            DebugExtensibleConvergence(extOrig,yOrig,xToFit,y,ext,newX,fitted)
        if (y.size == n):
            break
    return y

def WlcExtensible(ext,kbT,Lp,L0,K0,ForceGuess=None,Debug=False,
                  DebugConvergence=False,**kwargs):
    """
    Fits to the (recursively defined) extensible model. 

    Args: 
        kbT,Lp,L0,ext,K0,ForceGuess:  See WlcExtensible_Helper. Note if 
        ForceGuess: if None, then we use the non-extensible model to 'bootstrap'
        ourselves
    
        Debug: if true, then we plot the 'final' WLC plot
        DebugConvergence: if true, then we plot the convergence 'as we go',
        in addition to the final (ie: Can just set DebugConvergence, does Debug 
        too)
    Returns:
        see WlcPolyCorrect
    """
    if (ForceGuess is None):
        n = ext.size
        # maxFractionOfL0: determines the maximum fraction of L0 we fit to
        # non-extensible before switching to extensible. should always be
        # able to fit at least 40% of a contour length (very low forces,
        # less than (1 kbT/Lp)), since this is the region where WLC is linear
        # and hookean (see Docs)
        maxFractionOfL0 = max(0.4,1-10*(Lp/L0))
        highestX = maxFractionOfL0 * L0
        maxX = max(ext)
        # degree used for (possible) extensible fitting
        degree = 2
        # check where we stop fitting the non-extensible
        if (maxX > highestX):
            maxIdx = np.argmin(np.abs(highestX-ext))
        else:
            maxIdx = n
        maxIdx = max(degree,maxIdx)
        sliceV = slice(0,maxIdx,1)
        xToFit= ext[sliceV]
        y = WlcNonExtensible(xToFit,kbT,Lp,L0)
        # make copies of the non extensible model, this is completely
        # for debugging purposes.
        yOrig = y.copy()
        extOrig = xToFit.copy()
        # make the extensible version, shouldnt change (much)
        y = WlcExtensible_Helper(xToFit,kbT,Lp,L0,K0,y)
        if (DebugConvergence):
            DebugExtensibleConvergence(extOrig,yOrig,xToFit,y,ext)
        if (y.size < n):
            # extrapolate the y back
            pastMaxExt = maxX-highestX
            # determine the number of points per persistence length
            # I assume the separation points are more or less evenly
            # spaced
            deltaX = np.median(np.diff(ext))
            pointsPerLp = (Lp/deltaX)
            # Get the total number of points per extrapolation iteration.
            # We want each extrapolation to consist of the points
            extrapolationsPerLp = 5
            pointsPerExtrapolation = pointsPerLp/extrapolationsPerLp
            # we use a second order polynomial to fit, so we want to
            # make sure we have enough points for the fit itself
            nToAdd = max(2*degree,int(np.ceil(pointsPerExtrapolation)))
            y = ExtrapolateExtensible(nToAdd,ext,extOrig,yOrig,
                                      xToFit,y,WlcExtensible_Helper,
                                      DebugConvergence=DebugConvergence,
                                      degree=degree,kbT=kbT,Lp=Lp,L0=L0,K0=K0)
            # POST: y is extended
        toRet = y
        if (Debug or DebugConvergence):
            DebugExtensibleConvergence(extOrig,yOrig,ext,toRet,ext)
    else:
        # already have a guess, go with that
        toRet = WlcExtensible_Helper(ext,kbT,Lp,L0,K0,ForceGuess)
    return toRet

def Power(x,y):
    return x**y

def ContourOrModulusGradient(kbT,Lp,l,x,L0,coeffs,sign):
    """
    Calculates the gradient of the WLC force WRT the contour length
    or the bulk modulus, depending on the variables it is given

    Note that if a non-extensible model is desired, use the contour length
    inputs and set F=0 (see below)

    Args:
        kbT: always kbT
        Lp: always Lp
        l: always x/L0 - F/K0
        x: for {Contour length,Modulus} this is {extension,Force}
        L0: for {Contour length,Modulus} this is {L0,K0}
        coeffs: the in-order Bouchiat coefficients, assumed formatted like 
        output of BouchiatPolyCoeffs

        sign: for {Contour length,Modulus} this is {+1,-1}
    """
    # get the coefficient sum (see mathematica). We use L0 as the default
    # (hence the minus sign)
    n=7
    coeffSum = sum([(-i*(l**(i-1))*x*coeffs[i]/L0**2) for i in range(2,n+1)])
    cbp = kbT/Lp
    grad = sign*cbp*(-(x/Power(L0,2)) - x/(2.*Power(1 - l,3)*Power(L0,2)) + \
                     coeffSum)
    return np.reshape(grad,(grad.size,1))


def L0Gradient(params,ext,y,_,VaryNames,FixedDictionary):
    """
    XXX in development; the gradient of the contour length (useful for, e.g.
    Jacobian calculations)

    Args:
        params: the values of the parameters to set
        ext: the extension values
        force: the force values
        VaryNames: the names of the varying parameters.
        See WlcOptions.GetFullDictionary

        FixedDictionary: the fixed dictionary. See WlcOptions.GetFullDictionary
    """
    # Get the fixed and variable stuff -- probably need to pass in something
    # in kwargs to suss this out
    # get all the arguments using the fixed and varible names
    AllArgs = GetFullDictionary(VaryNames,FixedDictionary,*params)
    # Use the X and the variables to get the extensible force
    F = WlcExtensible(ext,**AllArgs)
    # Use the variables to calculate the gradients
    ParamsByName = WlcParamValues(**AllArgs)
    # Get the named parameters
    L0,Lp,K0,kbT = ParamsByName.GetParamValsInOrder()
    # Get the different values used...
    l = ext/L0 - F/K0
    # constant infront of the force
    cbp = kbT/Lp
    # get all the bouchiat coefficients
    coeffs = BouchiatPolyCoeffs()
    # get the gradient
    l = ext/L0-F/K0
    x = ext
    sign = 1 # for contour length
    grad = ContourOrModulusGradient(kbT,Lp,l,x,L0,coeffs,sign)
    return grad


def InitializeParamVals(model,toVary,Values=None,Bounds=None,
                        InitialObj=None):
    """
    Adapter to go to the general fitting method
    """
    function = GetFunctionFromModel(model)
    if (Values is None):
        Values = WLC_DEF.ValueDictionary
    if (Bounds is None):
        Bounds = WLC_DEF.BoundsDictionary
    if (InitialObj is None):
        InitialObj = Initialization()
    mVals = WlcParamValues(Vary=toVary,Bounds=Bounds,Values=Values)
    return FitInfo(FunctionToCall=function,ParamVals=mVals,
                   Initialization=InitialObj,FitOpt=FitOpt(Normalize=True))

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
        dict( [(WLC_MODELS.EXTENSIBLE_WANG_1997,WlcExtensible),
               (WLC_MODELS.INEXTENSIBLE_BOUICHAT_1999,WlcNonExtensible)])
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
    mInfo = InitializeParamVals(WLC_MODELS.EXTENSIBLE_WANG_1997,
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
    mInfo = InitializeParamVals(WLC_MODELS.EXTENSIBLE_WANG_1997,Bounds=Bounds,
                                toVary=toVary,InitialObj=InitialObj)
    return Fit(ext,force,mInfo)
                  
        

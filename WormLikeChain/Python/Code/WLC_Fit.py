# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

import copy
import FitUtils.Python.FitUtil as FitUtil

from scipy.interpolate import InterpolatedUnivariateSpline as spline
from WLC_HelperClasses import WlcParamValues,WlcParamsToVary,WlcFitInfo,\
    FitReturnInfo,BouchiatPolyCoeffs,GetFunctionCall,GetFullDictionary
from WLC_HelperClasses import WLC_MODELS,WLC_DEF

def WlcPolyCorrect(kbT,Lp,l):
    """
    From "Estimating the Persistence Length of a Worm-Like Chain Molecule ..."
    C. Bouchiat, M.D. Wang, et al.
    Biophysical Journal Volume 76, Issue 1, January 1999, Pages 409-413

web.mit.edu/cortiz/www/3.052/3.052CourseReader/38_BouchiatBiophysicalJ1999.pdf

    Args:
        kbT : the thermal energy in units of [ForceOutput]/Lp
        Lp  : the persisence length, sensible units of length
        l   : is either extension/Contour=z/L0 Length (inextensible) or   
        z/L0 - F/K0, where f is the force and K0 is the bulk modulus. See 
        Bouchiat, 1999 equation 13
    Returns:
        Model-predicted value for the force
    """
    # parameters taken from paper cited above
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
    return (kbT/Lp) * inner

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

def WlcExtensible(ext,kbT,Lp,L0,K0,ForceGuess=None,**kwargs):
    """
    Fits to the (recursively defined) extensible model. 

    Args: 
        kbT,Lp,L0,ext,K0,ForceGuess:  See WlcExtensible_Helper. Note if 
        ForceGuess is None, then we use the non-extensible model to 'bootstrap'
        ourselves
    Returns:
        see WlcPolyCorrect
    """
    if (ForceGuess is None):
        n = ext.size
        ## XXX move these into parameters? essentially, how slowly we
        # move from extensible to non-extensible
        # maxFractionOfL0: determines the maximum extension we fit to
        # SplitBeyondL0: related to how we divide the system into extensible
        # and non-extensible
        maxFractionOfL0 = 0.85 
        SplitBeyondL0 = 40
        highestX = maxFractionOfL0 * L0
        maxX = max(ext)
        # check where we stop fitting the non-extensible
        if (maxX > highestX):
            maxIdx = np.argmin(np.abs(highestX-ext))
        else:
            maxIdx = n
        sliceV = slice(0,maxIdx,1)
        xToFit= ext[sliceV]
        y = WlcNonExtensible(xToFit,kbT,Lp,L0)
        # extrapolate the y back
        nLeft = (n-maxIdx+1)
        deltaX = np.mean(np.diff(ext))
        # depending on rounding, may need to go factor+1 out
        factor = int(np.ceil(SplitBeyondL0*( (maxX/L0)-maxFractionOfL0)))
        nToAdd = int(np.ceil(nLeft/factor))
        degree=3
        for i in range(factor+1):
            # make a spline interpolator of degree 2
            f = spline(xToFit,y,ext='extrapolate',k=degree,
                       bbox=[min(ext),max(ext)])
            # extrapolate the previous fit out just a smidge
            sliceV = slice(0,maxIdx+nToAdd*i,1)
            xToFit = ext[sliceV]
            prev = f(xToFit)
            y = WlcExtensible_Helper(xToFit,kbT,Lp,L0,K0,prev)
            if (y.size == n):
                break
        toRet = y
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

    

def WlcFit(ext,force,WlcOptions=WlcFitInfo()):
    """
    General fiting function.

    Fits to a WLC model using the given parameters. If Extensible is set to 
    true, then fits an extensible model, subject to running nIters times, 
    or until the relative error between fits is less than rtol, whichever
    is first

    Args:
       ext: the (experimental) extension, 1D array of size N 
       force: the (experimental) force, 1D array of size N 
       Options: WlcFitInfo Object, giving the options for the fit
    Returns: 
       XXX The fitted values
    """
    model = WlcOptions.Model
    # scale everything to avoid convergence problems
    ext = ext.copy()
    force = force.copy()
    xScale = max(ext)
    ForceScale = max(force)
    ext /= xScale
    force /= ForceScale
    # get and scale the actual parameters
    Params = WlcOptions.ParamVals
    Params.NormalizeParams(xScale,ForceScale)
    # p0 record our initial guesses; what does the user want use to fit?
    varyDict = WlcOptions.GetVaryingParamDict()
    fixed = WlcOptions.GetFixedParamDict()
    # figure out what the model is
    if (model == WLC_MODELS.EXTENSIBLE_WANG_1997):
        # initially, use non-extensible for extensible model, as a first guess
        func = WlcExtensible
    elif (model == WLC_MODELS.INEXTENSIBLE_BOUICHAT_1999):
        func = WlcNonExtensible
    else:
        raise TypeError("Didnt recognize model {:s}".format(model))
    # XXX add in bounds
    fixedNames = fixed.keys()
    varyNames = varyDict.keys()
    varyGuesses = varyDict.values()
    # force all parameters to be positive
    bounds = (0,1.0)
    # number of evaluations should depend on the number of things we are fitting
    nEval = 500*varyNames
    if ((len(varyNames) == 1) and (WlcOptions.ParamsVaried.VaryL0)):
        jacFunc = lambda *args: L0Gradient(*args,VaryNames=varyNames,
                                           FixedDictionary=fixed)
    else:
        jacFunc = '3-point'
    fitOpt = dict(gtol=1e-15,
                  xtol=1e-15,
                  ftol=1e-15,
                  method='trf',
                  jac=jacFunc,
                  bounds=bounds,
                  max_nfev=nEval,
                  verbose=0)
    mFittingFunc = GetFunctionCall(func,varyNames,fixed)
    # note: we use p0 as the initial guess for the parameter values
    params,paramsStd,predicted = FitUtil.GenFit(ext,force,mFittingFunc,
                                                p0=varyGuesses,
                                                **fitOpt)
    # all done!
    # make a copy of the information object; we will return a new one
    finalInfo = copy.deepcopy(WlcOptions)
    # update the final parameter values
    finalVals = GetFullDictionary(varyNames,fixed,*params)
    # the fixed parameters have no stdev, by the fitting
    fixedStdDict = dict((name,None) for name in fixedNames)
    finalStdevs = GetFullDictionary(varyNames,fixedStdDict,
                                    *paramsStd)
    # set the values, their standard deviations, then denomalize everything
    finalInfo.ParamVals.SetParamValues(**finalVals)
    finalInfo.ParamVals.SetParamStdevs(**finalStdevs)
    finalInfo.ParamVals.DenormalizeParams(xScale,ForceScale)
    # update the actual values and parameters; update the prediction scale
    return FitReturnInfo(finalInfo,predicted*ForceScale)


def NonExtensibleWlcFit(ext,force,VaryL0=True,VaryLp=False,**kwargs):
    """
    Non-extensible version of the WLC fit. By default, varies the contour length
    to get the fit. Uses Bouichat, 1999 (see aboce) , by default

    Args:
        ext,force : see WlcFit
        VaryL0,VaryLp : see WlcParamsToVary, boolean if we should vary
        **kwargs: passed directly to WlcParamValues (ie: initial guesses)
    Returns:
        see WlcFit
    """
    model = WLC_MODELS.INEXTENSIBLE_BOUICHAT_1999
    mVals = WlcParamValues(**kwargs)
    toVary = WlcParamsToVary(VaryL0=VaryL0,VaryLp=VaryLp)
    # create the informaiton to pass on to the fitter
    mInfo = WlcFitInfo(Model=model,ParamVals=mVals,VaryObj=toVary)
    # call the fitter
    return WlcFit(ext,force,mInfo)

def ExtensibleWlcFit(ext,force,VaryL0=True,VaryLp=False,VaryK0=False,
                     nIters=500,rtol=1e-100,**kwargs):
    """
    extensible version of the WLC fit. By default, varies the contour length
    to get the fit. Uses Bouichat, 1999 (see aboce) , by default

    Args:
        ext,force : see WlcFit
        VaryL0,VaryLp : see WlcParamsToVary, boolean if we should vary
        nIters: number of iterations for the recursively defined function,
        before breaking

        rtol: the relative tolerance
        **kwargs: passed directly to WlcParamValues (ie: initial guesses)
    Returns:
        see WlcFit
    """
    model = WLC_MODELS.EXTENSIBLE_WANG_1997
    mVals = WlcParamValues(**kwargs)
    toVary = WlcParamsToVary(VaryL0=VaryL0,VaryLp=VaryLp,VaryK0=VaryK0)
    mInfo = WlcFitInfo(Model=model,ParamVals=mVals,VaryObj=toVary,
                       nIters=nIters,rtol=rtol)
    return WlcFit(ext,force,mInfo)
        

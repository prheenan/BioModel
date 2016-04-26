# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

import copy
import FitUtils.Python.FitUtil as FitUtil
from scipy.optimize import basinhopping
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from WLC_HelperClasses import WlcParamValues,WlcParamsToVary,WlcFitInfo,\
    FitReturnInfo,BouchiatPolyCoeffs,GetFunctionCall,GetFullDictionary,BoundsObj
from WLC_HelperClasses import WLC_MODELS,WLC_DEF,MACHINE_EPSILON
from collections import OrderedDict
from scipy.interpolate import dfitpack


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

def DebugExtensibleConvergence(extOrig,yOrig,extNow,yNow,ext):
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
    """
    plt.plot(extOrig,yOrig,'k--',linewidth=2.5)
    plt.plot(extNow,yNow,'b-')
    plt.xlabel("Extension (au)")
    plt.ylabel("Force (au)")
    # make a sensible range for the plotting
    minV = min(ext)
    maxV = max(ext)
    rangeV = maxV-minV
    fudge = rangeV/100
    plt.xlim([minV-fudge,maxV+fudge])
    # do the same s
    minY = min(yNow)
    maxY = max(yNow)
    rangeY = maxY-minY
    fudgeY = rangeY/100
    plt.ylim([-fudgeY,maxY+fudgeY])
    plt.show()

def WlcExtensible(ext,kbT,Lp,L0,K0,ForceGuess=None,Debug=False,
                  DebugConvergence=False,**kwargs):
    """
    Fits to the (recursively defined) extensible model. 

    If we are given NAN for the parameters, we return K0, as a maximum

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
        ## XXX move these into parameters? essentially, how slowly we
        # move from extensible to non-extensible
        # maxFractionOfL0: determines the maximum extension we fit to
        # and non-extensible
        maxFractionOfL0 = 0.85
        highestX = maxFractionOfL0 * L0
        maxX = max(ext)
        # check where we stop fitting the non-extensible
        degree=3
        if (maxX > highestX):
            maxIdx = np.argmin(np.abs(highestX-ext))
        else:
            maxIdx = n
        maxIdx = max(degree,maxIdx)
        sliceV = slice(0,maxIdx,1)
        xToFit= ext[sliceV]
        y = WlcNonExtensible(xToFit,kbT,Lp,L0)
        yOrig = y.copy()
        extOrig = xToFit.copy()
        # extrapolate the y back
        nLeft = (n-maxIdx+1)
        pastMaxExt = maxX-highestX
        # figure out roughly how screwed we are to get out to where we want to
        # go
        nLengths = (pastMaxExt/Lp)*(L0/Lp)
        factor = max(1,int(np.ceil(nLengths/5)))
        # depending on rounding, may need to go factor+1 out
        nToAdd = max(3,int(np.ceil(nLeft/factor)))
        for i in range(factor+1):
            # make a spline interpolator of degree k=degree
            # this can throw an error with badly conditioned data.
            try:
                f = spline(xToFit,y,ext='extrapolate',k=degree,\
                           bbox=[min(ext),max(ext)])
            except Exception as e:
                # if that didnt work for some reason, then use a lower degree
                f = spline(xToFit,y,ext='extrapolate',k=degree-1,\
                           bbox=[min(ext),max(ext)])
            # extrapolate the previous fit out just a smidge
            sliceV = slice(0,maxIdx+nToAdd*i,1)
            xToFit = ext[sliceV]
            prev = f(xToFit)
            if (DebugConvergence):
                DebugExtensibleConvergence(extOrig,yOrig,xToFit,prev,ext)
            y = WlcExtensible_Helper(xToFit,kbT,Lp,L0,K0,prev)
            if (y.size == n):
                break
        if (Debug or DebugConvergence):
            DebugExtensibleConvergence(extOrig,yOrig,xToFit,prev,ext)
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


def SafeMinimize(n,func,*params,**kwargs):
    """
    Minimized a function. in the case of infinities at a certain location, or
    if we couldt minimize, we give the point a score of n (so the typical errors
    should be much less than one, guarenteed if the function is normalized to
    between 0 and 1)

    Args:
        n: the size of whatever we are tryign to minimize
        func: the function we will be minimizing. If it throws and overflow.
        We assume that the function is normalized between 0 and 1, n >1.

        error, we assume that the output was ridiculous, and give a score of
        n**2
    """
    pArray = params[0]
    try:
        naive = func(*pArray,**kwargs)
        naive[np.where(~np.isfinite(naive))] = n
    except OverflowError as e:
        # each point (n!) had 
        naive = np.ones(n) * n
    return naive
    

def WlcFit(extRaw,forceRaw,WlcOptions=WlcFitInfo(),UseBasin=False):
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
       WlcFitInfo, with updated parmaeters and standard deviations associated
       with the fit.
    """
    model = WlcOptions.Model
    # scale everything to avoid convergence problems
    xNormalization = max(extRaw)
    forceNormalization = max(forceRaw)
    ExtScaled = extRaw.copy()/xNormalization
    ForceScaled = forceRaw.copy()/forceNormalization
    # get and scale the actual parameters (note this also scales the bounds!)
    Params = WlcOptions.ParamVals
    Params.NormalizeParams(xNormalization,forceNormalization)
    # varyDict record our initial guesses; what does the user want use to fit?
    varyDict = WlcOptions.GetVaryingParamDict()
    fixed = WlcOptions.GetFixedParamDict()
    # get the bounds, convert to CurveFit's conventions
    boundsRaw = WlcOptions.DictToValues(WlcOptions.GetVaryingBoundsDict())
    # curve_fit wans a list of [ [lower1,lower2,...],[upper1,upper2,...]]
    boundsCurvefitRaw = [BoundsObj.ToCurveFitConventions(*b) for b in boundsRaw]
    boundsCurvefitLower = [b[0] for b in boundsRaw]
    boundsCurvefitUpper = [b[1] for b in boundsRaw]
    boundsCurvefit = boundsCurvefitLower,boundsCurvefitUpper
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
    # number of evaluations should depend on the number of things we are fitting
    nEval = 500*varyNames
    mFittingFunc = GetFunctionCall(func,varyNames,fixed)
    if (UseBasin):
        boundsBasin = [BoundsObj.ToMinimizeConventions(*b) for b in boundsRaw]
        # basin hopping funciton actually keeps x fixed, so we just pass it in
        basinHoppingFunc = lambda *params : mFittingFunc(ExtScaled,*params)
        #  minimize sum of the residuals/N. Since the scales are normalized,
        # this should be at most between 0 and 1, so normalizing by N
        # means this will (usually) be bettween 0 and 1 (for any reasonable fit)
        nPoints = ExtScaled.size
        basinSafe = lambda *params,**kw: SafeMinimize(nPoints,basinHoppingFunc,
                                                      *params,**kw)
        minimizeFunc = lambda *params,**kw: sum(np.abs(basinSafe(*params,**kw)-\
                                                       ForceScaled))/nPoints
        # the minimizer itself (for each 'basin') takes keywords
        # here, we are a little less 'picky' about the function tolerances
        # than before
        minimizer_kwargs = OrderedDict(method="TNC",bounds=boundsBasin,
                                       options=dict(ftol=1e-3,
                                                    xtol=1e-3,gtol=1e-3))
        # use basin-hopping to get a solid guess of where we should  start
        obj = basinhopping(minimizeFunc,x0=varyGuesses,disp=False,T=1,
                           stepsize=0.001,minimizer_kwargs=minimizer_kwargs,
                           niter_success=10,interval=10,niter=30)
        varyGuesses = obj.x
    # now, set up a slightly better-quality fit, based on the local minima
    # that the basin-hopping function
    jacFunc = '3-point'
    fitOpt = dict(gtol=1e-15,
                  xtol=1e-15,
                  ftol=1e-15,
                  method='trf', # trf support bounds, which is good!
                  jac=jacFunc,
                  # XXX kind of a kludge...
                  bounds=boundsCurvefit,
                  max_nfev=nEval,
                  verbose=0)
    # note: we use p0 as the initial guess for the parameter values
    params,paramsStd,predicted = FitUtil.GenFit(ExtScaled,ForceScaled,
                                                mFittingFunc,p0=varyGuesses,
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
    finalInfo.ParamVals.DenormalizeParams(xNormalization,
                                          forceNormalization)
    # update the actual values and parameters; update the prediction scale
    return FitReturnInfo(finalInfo,predicted*forceNormalization)


def NonExtensibleWlcFit(ext,force,VaryL0=True,VaryLp=False,**kwargs):
    """
    Non-extensible version of the WLC fit. By default, varies the contour length
    to get the fit. Uses Bouichat, 1999 (see aboce) , by default

    Args:
        ext,force : see WlcFit
        VaryL0,VaryLp : see WlcParamsToVary, boolean if we should vary
        **kwargs: passed directly to WlcParamValues (ie: initial guesses,bounds)
    Returns:
        see WlcFit
    """
    model = WLC_MODELS.INEXTENSIBLE_BOUICHAT_1999
    mVals = WlcParamValues(**kwargs)
    # non-extensible model has no K0
    toVary = WlcParamsToVary(VaryL0=VaryL0,VaryLp=VaryLp,VaryK0=False)
    # create the informaiton to pass on to the fitter
    mInfo = WlcFitInfo(Model=model,ParamVals=mVals,VaryObj=toVary)
    # call the fitter
    return WlcFit(ext,force,mInfo)

def ExtensibleWlcFit(ext,force,VaryL0=True,VaryLp=False,VaryK0=False,
                     **kwargs):
    """
    extensible version of the WLC fit. By default, varies the contour length
    to get the fit. Uses Bouichat, 1999 (see aboce) , by default

    Args:
        ext,force : see WlcFit
        VaryL0,VaryLp : see WlcParamsToVary, boolean if we should vary
        **kwargs: passed directly to WlcParamValues (ie: initial guesses,bounds)
    Returns:
        see WlcFit
    """
    model = WLC_MODELS.EXTENSIBLE_WANG_1997
    mVals = WlcParamValues(**kwargs)
    toVary = WlcParamsToVary(VaryL0=VaryL0,VaryLp=VaryLp,VaryK0=VaryK0)
    mInfo = WlcFitInfo(Model=model,ParamVals=mVals,VaryObj=toVary)
    return WlcFit(ext,force,mInfo)
        

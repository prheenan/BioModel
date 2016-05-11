# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

import copy
from collections import OrderedDict
from FitUtils.Python.FitClasses import Initialization,BoundsObj,\
    FitInfo,FitReturnInfo,GetFunctionCall,GetFullDictionary
import FitUtils.Python.FitUtil as FitUtil
from scipy.optimize import brute
import warnings


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
    # ugly parameter adapting...
    if (len(params[0]) > 1):
        # then just convert the first element to a tuple
        params = params[0]
    else:
        # convert the (only) element to its own tuple
        params = tuple([params[0]])
    # need to watch out for overflows, meaning the WLC has shot to
    # infinity and we ignore the results.
    # note this assumes the function to minimize is well-tested;
    # any errors are from bad sets of parameters, rather than the
    # model itself
    try:
        # Ignore warnings due to auto-fitting the data.
        # This assumes the function is built correctly, but it is
        # totally possible for us to give it terrible parameters --
        # we dont want to spam the user.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            naive = func(*params,**kwargs)
            naive[np.where(~np.isfinite(naive))] = n
    except (OverflowError,RuntimeError,ValueError) as e:
        # each point (n) given the highest weight, data is 'broken'
        naive = np.ones(n) * n
    return naive
    
def GetMinimizingFunction(xScaled,yScaled,mFittingFunc):
    """
    Given data for extension and force, minimizes the normalized sum of the 
    absolute derivatives

    Args:
        xScaled: the scaled x
        yScaled: the scaled y
        mFittingFunc: funciton taking in the extension and parameters,
        returning a value for the force.
    """
    # basin hopping funciton actually keeps x fixed, so we just pass it in
    basinHoppingFunc = lambda *params : mFittingFunc(xScaled,*params)
    #  minimize sum of the residuals/N. Since the scales are normalized,
    # this should be at most between 0 and 1, so normalizing by N
    # means this will (usually) be bettween 0 and 1 (for any reasonable fit)
    nPoints = xScaled.size
    basinSafe = lambda *params,**kw: SafeMinimize(nPoints,basinHoppingFunc,
                                                  *params,**kw)
    # minimize the sum of the residuals divided by the number of points
    minimizeFunc = lambda *params,**kw: sum(np.abs(basinSafe(*params,**kw)-\
                                                   yScaled))/nPoints
    return minimizeFunc

def Fit(x,y,Options):
    """
    General fiting function.

    Fits to a WLC model using the given parameters. If Extensible is set to 
    true, then fits an extensible model, subject to running nIters times, 
    or until the relative error between fits is less than rtol, whichever
    is first

    Args:
       x: 1D array of size N, independent
       y: what we want to fit to
       Options: FitInfo Object, giving the options for the fit
    Returns: 
       FitInfo, with updated parmaeters and standard deviations associated
       with the fit.
    """
    func = Options.Model
    # scale everything to avoid convergence problems
    xNormalization,yNormalization = Options.FitOptions.\
                                    GetNormalizationCoeffs(x,y)
    xScaled = x.copy()/xNormalization
    yScaled = y.copy()/yNormalization
    # get and scale the actual parameters (note this also scales the bounds!)
    Params = Options.ParamVals
    Params.NormalizeParams(xNormalization,yNormalization)
    # varyDict record our initial guesses; what does the user want use to fit?
    varyDict = Options.GetVaryingParamDict()
    fixed = Options.GetFixedParamDict()
    # get the bounds, convert to CurveFit's conventions
    boundsRaw = Options.DictToValues(Options.GetVaryingBoundsDict())
    # curve_fit wans a list of [ [lower1,lower2,...],[upper1,upper2,...]]
    boundsCurvefitRaw = [BoundsObj.ToCurveFitConventions(*b) for b in boundsRaw]
    boundsCurvefitLower = [b[0] for b in boundsRaw]
    boundsCurvefitUpper = [b[1] for b in boundsRaw]
    boundsCurvefit = boundsCurvefitLower,boundsCurvefitUpper
    # figure out what is fixed and varying
    fixedNames = fixed.keys()
    varyNames = varyDict.keys()
    varyGuesses = varyDict.values()
    # force all parameters to be positive
    # number of evaluations should depend on the number of things we are fitting
    nEval = 500*len(varyNames)
    mFittingFunc = GetFunctionCall(func,varyNames,fixed)
    # set up things for basin / brute force initalization
    toMin = GetMinimizingFunction(xScaled,yScaled,mFittingFunc)
    boundsBasin = [BoundsObj.ToMinimizeConventions(*b) for b in boundsRaw]
    initObj = Options.Initialization
    if (initObj.Type == Initialization.HOP):
        # Use the basin hopping mode
        obj = FitUtil.BasinHop(toMin,varyGuesses,boundsBasin,*initObj.Args,
                               **initObj.ParamDict)
        varyGuesses = obj.x
    elif (initObj.Type == Initialization.BRUTE):
        # use the brute force method
        x0,fval,grid,jout= brute(toMin,ranges=boundsBasin,disp=False,
                                 full_output=True,
                                 *initObj.Args,**initObj.ParamDict)
        Options.Initialization.SetInitializationInfo(x0=x0,
                                                     fval=fval,
                                                     grid=grid,
                                                     jout=jout)
        varyGuesses = x0
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
    params,paramsStd,predicted = FitUtil.GenFit(xScaled,yScaled,
                                                mFittingFunc,p0=varyGuesses,
                                                **fitOpt)
    # all done!
    # make a copy of the information object; we will return a new one
    finalInfo = copy.deepcopy(Options)
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
                                          yNormalization)
    finalInfo.FitOptions.SetNormCoeffs(xNormalization,
                                       yNormalization)
    # update the actual values and parameters; update the prediction scale
    finalPrediction = finalInfo.FunctionToPredict(xScaled,**finalVals)*\
                      yNormalization
    return FitReturnInfo(finalInfo,finalPrediction)

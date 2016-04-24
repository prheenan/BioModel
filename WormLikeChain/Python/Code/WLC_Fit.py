# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

import copy
from scipy.optimize import curve_fit
import FitUtils.Python.FitUtil as FitUtil
from collections import OrderedDict

from scipy.interpolate import InterpolatedUnivariateSpline as spline
MACHINE_EPSILON = np.finfo(float).eps

class WLC_DEF:
    """
    Class defining defaults for inputs. 

    From "Estimating the Persistence Length of a Worm-Like Chain Molecule ..."
    C. Bouchiat, M.D. Wang, et al.
    Biophysical Journal Volume 76, Issue 1, January 1999, Pages 409-413

    See Wang, 1997.
    """
    L0 = 1317.52e-9 # meters
    Lp = 40.6e-9 # meters
    K0 = 1318e-12 # Newtons
    kbT = 4.11e-21 # 4.1 pN * nm = 4.1e-21 N*m

class WLC_MODELS:
    """
    Class definining valid models.
    """
    EXTENSIBLE_WANG_1997 = 0
    INEXTENSIBLE_BOUICHAT_1999 = 1


class WlcParamsToVary:
    """
    Class to keep track of what to vary
    """
    def __init__(self,VaryL0=True,VaryLp=False,VaryK0=False):
        """
        Args: 
           VaryL0: If true, contour length is allowed to freely vary
           VaryLp: If true, persistence length is allowed to freely vary
           VaryK0: If true, bulk modulus K0 is allowed to vary
        """
        self.VaryL0 = VaryL0
        self.VaryLp = VaryLp
        self.VaryK0 = VaryK0
    
class WlcParamValues:
    """
    Class to record parameter values given to a fit or gotten from the same
    """
    class Param:
        """
        Subclass to keep track of parameters.
        """
        def __init__(self,Value,Stdev=None,Bounds=None):
            self.Value = Value
            self.Stdev = Stdev
            self.Bounds = None
        def Scale(self,scale):
            """
            Scale the parameter (and standard deviation) to scale,
            ie: just divides through by scale
            """
            self.Value /= scale
            if (self.Stdev is not None):
                self.Stdev /= scale
        def __str__(self):
            stdevStr = "+/-{:5.2g}".format(self.Stdev) \
                       if self.Stdev is not None else ""
            return "{:5.4g}{:s}".format(self.Value,stdevStr)
    def __init__(self,kbT=WLC_DEF.kbT,
                 L0=WLC_DEF.L0,Lp=WLC_DEF.Lp,K0=WLC_DEF.K0):
        """
        Args:
            kbT,Lp,L0 : see WlcPolyCorrect. Initial guesses
            K0: see WlcExtensible. Note this is ignored for non-extensible 
            models
        """
        self.SetParamValues(L0,Lp,K0,kbT)
    def SetParamValues(self,L0,Lp,K0,kbT):
        """
        Sets the parameter values

        Args:
            kbT,Lp,L0,K0: See Init
        """
        self.L0 = WlcParamValues.Param(L0)
        self.Lp = WlcParamValues.Param(Lp)
        self.K0 = WlcParamValues.Param(K0)
        self.kbT = WlcParamValues.Param(kbT)
    def CloseTo(self,other,rtol=1e-1,atol=0):
        """
        Returns true of the 
        """
        myVals = self.GetParamValsInOrder()
        otherVals = other.GetParamValsInOrder()
        return np.allclose(myVals,otherVals,rtol=rtol,atol=atol)
    def SetBounds(self,L0,Lp,K0=None,kbT=None):
        self.L0.Bounds = L0
        self.Lp.Bounds = Lp
        self.K0.Bounds = K0
        self.kbT.Bounds = kbT
    def GetParamDict(self):
        """
        Returns: in-order dictionary of the parameters
        """
        return OrderedDict(L0=self.L0,Lp=self.Lp,K0=self.K0,kbT=self.kbT)
    def SetParamStdevs(self,L0,Lp,K0,kbT):
        """
        Sets the parameter stdevs

        Args:
            See SetParamValues, except all standard deviations
        """
        attr = [(self.L0,L0),
                (self.Lp,Lp),
                (self.K0,K0),
                (self.kbT,kbT)]
        for a,stdev in attr:
            a.Stdev = stdev
    def GetParamValsInOrder(self):
        return [v.Value for v in self.GetParamsInOrder()]
    def GetParamsInOrder(self):
        """
        Conveniene function, gets the parameters in the conventional order
        """
        return self.GetParamDict().values()
    def ScaleGen(self,xScale,ForceScale):
        """
        Scales the data to an x and y scale given by xScale and ForceScale.
        
        In other words, x -> x/xScale, etc

        Args:
            xScale: What to divide the distance parts by
            yScale: What to divide the force parts by
        """
        # lengths are distances
        self.L0.Scale(xScale)
        self.Lp.Scale(xScale)
        # K0 is a force
        self.K0.Scale(ForceScale)
        # note: kbT is an energy = force * distance
        self.kbT.Scale(ForceScale*xScale)
    def NormalizeParams(self,xScale,ForceScale):
        """
        Normalize the given parameters

        Args:
           xScale/yScale: see ScaleGen
        """
        self.ScaleGen(xScale,ForceScale)
    def DenormalizeParams(self,xScale,ForceScale):
        """
        De-Normalize the given parameters

        Args:
           xScale/yScale: see ScaleGen
        """
        self.ScaleGen(1./xScale,1./ForceScale)

class WlcFitInfo:
    def __init__(self,Model=WLC_MODELS.EXTENSIBLE_WANG_1997,
                 ParamVals=WlcParamValues(),nIters=500,
                 rtol=1e-2,VaryObj=WlcParamsToVary()):
        """
        Args:
        Model: which model, from WLC_MODELS, to use
        ParamValues: values of the parameters for the fit (e.g. initial guesses,
        or final resuts )

        rTol: for extensible models, the relative tolerance between sucessive
        fits before quitting

        nIters: for extensible models, the maximum number of iterations.
        VaryObj: which parameters should be varied for the 
        """
        self.Model = Model
        self.ParamVals = ParamVals
        self.ParamsVaried = VaryObj
        self.nIters = nIters
        self.rtol = rtol
    def AddParamsGen(self,condition=lambda x: x):
        toRet = OrderedDict()
        # assume we never vary temperature
        toVary = self.ParamsVaried
        vals = self.ParamVals
        addToDict = lambda **kwargs: toRet.update(dict(kwargs))
        if (condition(toVary.VaryL0)):
            addToDict(L0=vals.L0.Value)
        if (condition(toVary.VaryLp)):
            addToDict(Lp=vals.Lp.Value)
        if (condition(toVary.VaryK0)):
            addToDict(K0=vals.K0.Value)
        return toRet
    def GetVaryingParamDict(self):
        """
        Gets the dictionary of varying parameters
        """
        toRet= self.AddParamsGen(condition=lambda x : x)
        return toRet
    def GetFixedParamDict(self):
        """
        Gets the dictionary of fixed parameters (assumes temperature is amoung
        """
        # temperature comes first in the ordering
        toRet = OrderedDict(kbT=self.ParamVals.kbT.Value)
        allButTemp = self.AddParamsGen(condition=lambda x : not x)
        toRet.update(allButTemp)
        return toRet
    def GetFullDictionary(self,ParamNamesToVary,ParamsFixedDict,*args):
        mapV = lambda key,vals: OrderedDict([(k,v) for k,v in zip(key,vals)])
        return OrderedDict(mapV(ParamNamesToVary,args),**ParamsFixedDict)
    def GetFunctionCall(self,func,ParamNamesToVary,ParamsFixedDict):
        return lambda ext,*args : func(ext,\
                 **self.GetFullDictionary(ParamNamesToVary,
                                          ParamsFixedDict,*args))
    def __str__(self):
        return "\n".join("{:10s}\t{:s}".format(k,v) for k,v in
                         self.ParamVals.GetParamDict().items())

            
class FitReturnInfo:
    """
    Class to return when we are done fitting
    """
    def __init__(self,inf,PredictedData):
        """
        Initialize the fit object

        Args:
            inf: WlcFitInfo object from the model fitting, updated with the
            parameter values we found

            PredictedData: the prediction made by the model
        """
        self.Info = inf
        self.Prediction = PredictedData
    def __str__(self):
        return str(self.Info)
        
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
    # see especially equation 13
    polyValCoeffs = [a7,a6,a5,a4,a3,a2,a1,a0]
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

def WlcExtensible(ext,kbT,Lp,L0,K0,ForceGuess=None):
    """
    Fits to the (recursively defined) extensible model. 

    Args: 
        kbT,Lp,L0,ext,K0,ForceGuess:  See WlcExtensible_Helper
    Returns:
        see WlcPolyCorrect
    """
    if (ForceGuess is None):
        n = ext.size
        ## XXX move these into parameters?
        maxFractionOfL0 = 0.85 
        SplitBeyondL0 = 40
        highestX = maxFractionOfL0 * L0
        maxX = max(ext)
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
            sliceV = slice(0,maxIdx+nToAdd*i,1)
            xToFit = ext[sliceV]
            prev = f(xToFit)
            y = WlcExtensible_Helper(xToFit,kbT,Lp,L0,K0,prev)
            if (y.size == n):
                break
        toRet = y
    else:
        toRet = WlcExtensible_Helper(ext,kbT,Lp,L0,K0,ForceGuess)
    return toRet

    
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
    fitOpt = dict(gtol=1e-15,
                  xtol=1e-15,
                  ftol=1e-15,
                  method='trf',
                  jac='3-point',
                  bounds=bounds,
                  max_nfev=nEval,
                  verbose=0)
    mFittingFunc = WlcOptions.GetFunctionCall(func,varyNames,fixed)
    # note: we use p0 as the initial guess for the parameter values
    params,paramsStd,predicted = FitUtil.GenFit(ext,force,mFittingFunc,
                                                p0=varyGuesses,
                                                **fitOpt)
    # all done!
    # make a copy of the information object; we will return a new one
    finalInfo = copy.deepcopy(WlcOptions)
    # update the final parameter values
    finalVals = WlcOptions.GetFullDictionary(varyNames,fixed,*params)
    # the fixed parameters have no stdev, by the fitting
    fixedStdDict = dict((name,None) for name in fixedNames)
    finalStdevs = WlcOptions.GetFullDictionary(varyNames,fixedStdDict,
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
        

# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys


from collections import OrderedDict
MACHINE_EPSILON = np.finfo(float).eps


class BoundsObj:
    """
    Class to keep track of bounds
    """
    def SantizeBounds(self,bound):
        # we will by default assume that an unlimited bounds is None
        if (np.isnan(bound) or bound is None):
            return None
    def __init__(self,lower,upper):
        self.upper = upper
        self.lower = lower
    def scale(self,scale):
        if (self.upper is not None):
            self.upper /= scale
        if (self.upper is not None):
            self.lower /= scale
    def tuple(self,returnInf=None):
        dealWithUnbounded = lambda x: x if (x is not None) else returnInf
        upper = dealWithUnbounded(upper)
        lower = dealWithUnbounded(lower)
        return lower,upper

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
    BoundsDictionary = OrderedDict(L0=BoundsObj(0,None),
                                   Lp=BoundsObj(0,None),
                                   K0=BoundsObj(0,None),
                                   kbT=BoundsObj(0,None))
    

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
        def __init__(self,Value=0,Stdev=None,Bounds=None):
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
            self.Bounds.scale(scale)
        def __str__(self):
            stdevStr = "+/-{:5.2g}".format(self.Stdev) \
                       if self.Stdev is not None else ""
            return "{:5.4g}{:s}".format(self.Value,stdevStr)
        def __repr__(self):
            return str(self)
    def __init__(self,
                 Values=WLC_DEF.ValueDictionary,
                 Bounds=WLC_DEF.BoundsDictionary):
        """
        Args:
            kbT,Lp,L0 : see WlcPolyCorrect. Initial guesses
            K0: see WlcExtensible. Note this is ignored for non-extensible 
            models
        """
        self._InitParams()
        self.SetParamValues(**Values)
        self.SetBounds(**Bounds)
    def _InitParams(self):
        """
        Initiliaze parameters...
        """
        self.L0 = WlcParamValues.Param()
        self.Lp = WlcParamValues.Param()
        self.K0 = WlcParamValues.Param()
        self.kbT = WlcParamValues.Param()
    def SetParamValues(self,L0,Lp,K0,kbT):
        """
        Sets the parameter values, initializing new objects

        Args:
            kbT,Lp,L0,K0: See Init
        """
        self.L0.Value = L0
        self.Lp.Value = Lp
        self.K0.Value = K0
        self.kbT.Value = kbT
    def CloseTo(self,other,rtol=1e-1,atol=0):
        """
        Returns true if the other set of parameters is the 'same' as this

        Args:
            other: other WlcParamValues object 
            rtol: relative tolerance, see np.allclose
            atol: absolute tolerance, see allclose
        Returns:
            True/False if the condition holds
        """
        myVals = self.GetParamValsInOrder()
        otherVals = other.GetParamValsInOrder()
        return np.allclose(myVals,otherVals,rtol=rtol,atol=atol)
    def SetBounds(self,L0,Lp,K0,kbT):
        """
        Sets the bounds for each Parameter. See __init__

        Args:
           L0,Lp,K0,kbT: each should be a tuple of [start,end]. e.g.
           (0,np.inf) for only positive numbers
        """
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
    def _GenGetInOrder(self,func):
        """
        Reutrns some transform on each inorder parameter

        Args:
           func: takes in a parameter, returns whatever we want from it
        """
        return [func(v) for v in self.GetParamsInOrder()]
    def GetParamValsInOrder(self):
        """
        Returns:
            a list of the in-order parameters (ie: the conventional order
            I use, see any fitting function)
        """
        return self._GenGetInOrder(lambda x: x.Value)
    def GetParamBoundsInOrder(self):

        return self._GenGetInOrder(lambda x: x.Bounds)
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
                 ParamVals=WlcParamValues(),
                 VaryObj=WlcParamsToVary()):
        """
        Args:
        Model: which model, from WLC_MODELS, to use
        ParamValues: values of the parameters for the fit (e.g. initial guesses,
        or final resuts )

        VaryObj: which parameters should be varied for the 
        """
        self.Model = Model
        self.ParamVals = ParamVals
        self.ParamsVaried = VaryObj
    def AddParamsGen(self,condition=lambda x: x):
        """
        General function to add paraters to a list we return.

        Args: 
            condition: function taking in a flag if we should vary the object,
            and returning true/false if it should be added to the lisst
        Returns:
            list of elememnts we added by condition
        """
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


def GetFullDictionary(ParamNamesToVary,ParamsFixedDict,*args):
    """
    Given names of parmeters to vary, a dictionary of fixed parameters,
    And the actual argumennts, gets a dictionry of all name:value
    arguments (bot varying and fixed)

    Useful if we want to 'dynamically' change what we vary. This is probably
    used in a lambda function or similar, which gives *args (see 
    GetFunctionCall)

    Args:
        ParamNamesToVary; see GetFunctionCall
        ParamsFixedDict: ee GetFunctionCall
        *args: the values of the actual arguments
    Returns:
        Unambiguous dictionary of all arguments
    """
    mapV = lambda key,vals: OrderedDict([(k,v) for k,v in zip(key,vals)])
    return OrderedDict(mapV(ParamNamesToVary,args),**ParamsFixedDict)

def GetFunctionCall(func,ParamNamesToVary,ParamsFixedDict):
    """
    Method to get a function call, given which parameters we vary and
    which we keep fixed

    Args:
         func: the (fitting) function, assumed to have a signature like
         func(extension,<parameters in customary order>)

         ParamNamesToVary: The names of the parameters we will vary, in the
         proper order

         ParamsFixedDict: The names of the parameters which are fixed
    Returns:
         lambda function like func(extension,arg1,arg2,...), where arg1,
         arg2, etc are the function to varu
    """
    return lambda ext,*args : func(ext,\
             **GetFullDictionary(ParamNamesToVary,ParamsFixedDict,*args))

# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys


from collections import OrderedDict
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

         ParamNamesToVary: The names of the parameters we will vary
         ParamsFixedDict: The names of the parameters which are fixed
    Returns:
         lambda function like func(extension,arg1,arg2,...), where arg1,
         arg2, etc are the function to varu
    """
    return lambda ext,*args : func(ext,\
             **GetFullDictionary(ParamNamesToVary,ParamsFixedDict,*args))

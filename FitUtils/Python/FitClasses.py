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
    def __init__(self,lower,upper):
        """
        initialize the bounds to upper and lwoer
        """
        self.upper = upper
        self.lower = lower
    @classmethod
    def ClosedBound(BoundsObj,bound):
        """
        Returns true if the bound is non-infinite
        """
        return (bound is not None) and (np.isfinite(bound))
    def scale(self,scale):
        """
        Scale the bounds so they are normalized by scale

        Args:
           scale: what to normalize to 
        """
        if (BoundsObj.ClosedBound(self.upper)):
            self.upper /= scale
        if (BoundsObj.ClosedBound(self.lower)):
            self.lower /= scale
    def AsTuple(self):
        """
        return the lower and upper bounds as a tuple
        """
        return self.lower,self.upper
    @classmethod
    def _ToConvention(BoundsObj,lower,upper,lowerConvention,upperConvention):
        if (not BoundsObj.ClosedBound(lower)):
            lower = lowerConvention
        if (not BoundsObj.ClosedBound(upper)):
            upper = upperConvention
        return lower,upper
    @classmethod
    def ToCurveFitConventions(BoundsObj,lower,upper):
        """
        Scipy's curve fit uses infs as the bounds
        
        See: scipy.optimize.curve_fit
docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.optimize.curve_fit.html
        """
        return BoundsObj._ToConvention(lower,upper,-np.inf,np.inf)
            
    @classmethod
    def ToMinimizeConventions(BoundsObj,lower,upper):
        """
        Scipy 'Minimize' uses None as the positive and negative conditions

        See: scipy.optimize.minimize, 'bounds'
docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.optimize.minimize.html
        """
        return BoundsObj._ToConvention(lower,upper,None,None)
    

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
    BoundsDictionary = OrderedDict(L0=BoundsObj(0,np.inf),
                                   Lp=BoundsObj(0,np.inf),
                                   K0=BoundsObj(0,np.inf),
                                   kbT=BoundsObj(0,np.inf))
    # default varying dictionary
    VaryDictionary =  OrderedDict(L0=False,
                                  Lp=False,
                                  K0=False,
                                  kbT=False)

    
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
        def __init__(self,Value=0,Stdev=None,Bounds=None,Name="",Vary=False):
            """
            Initialize the properties of the parameter

            Args:
                Value: the value of the parameter
                Stdev: the standard deviation fo the paameter
                Bounds: the bound of the paramters
                Name: the name of the parameter
                Vary: if the parameter should be varied
            """
            self.Name=Name
            self.Value = Value
            self.Stdev = Stdev
            self.Bounds = Bounds
            self.Vary = Vary
        """
        Various Setters below, to set the properites
        """
        def SetVarying(self,x):
            """
            Args:
                x: should be a Boolean
            """
            self.Vary = x
        def SetBounds(self,x):
            """
            Args:
                x: should be a Bounds obj
            """
            self.Bounds = x
        def SetStdev(self,x):
            self.Stdev = x
        def SetValue(self,x):
            self.Value = x
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
                 Bounds=WLC_DEF.BoundsDictionary,
                 Vary=WLC_DEF.VaryDictionary):
        """
        Args:
            Values: dictionary of <Parameter:Value> pairs
            Bounds: dictionary of <Parameter:BoundsObj> pairs
            Vary: dictionary of <Parameter:VaryingObj> pairs
        """
        self._InitParams()
        self.SetParamValues(**Values)
        self.SetBounds(**Bounds)
        self.SetVarying(**Vary)
    def _InitParams(self):
        """
        Initiliaze parameters...
        """
        Params = [WlcParamValues.Param(Name="L0"),
                  WlcParamValues.Param(Name="Lp"),
                  WlcParamValues.Param(Name="K0"),
                  WlcParamValues.Param(Name="kbT")]
        self.ParamDict = OrderedDict()
        for p in Params:
            self.ParamDict[p.Name] = p

    def _SetGen(self,func,**kwargs):
        """
        General setting method, loops through each parameter and sets
        
        Args:
            func: the function to apply, takes in a Param object and a value
            corresponding to the parameter from kwargs
           
            **kwargs: dictionary of <key:value> pairs
        """
        for k,v in kwargs.items():
            func(self.ParamDict[k],v)
    def SetParamStdevs(self,**kwargs):
        """
        Sets the parameter stdevs

        Args:
            **kwargs: see SetVarying for keys. Each value is a standard dev
        """
        self._SetGen(lambda x,y: x.SetStdev(y),**kwargs)
    def SetVarying(self,**kwargs):
        """
        Sets the relevant parameters to their values

        Args:
            **kwargs: dictionary, keys must match the parameter names this was 
            initialized with. values are booleans
        """
        self._SetGen(lambda x,y: x.SetVarying(y),**kwargs)
    def SetParamValues(self,**kwargs):
        """
        Sets the parameter values

        Args:
            **kwargs: See: SetVarying. each value is a float
        """
        self._SetGen(lambda x,y: x.SetValue(y),**kwargs)
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
    def SetBounds(self,**kwargs):
        """
        Sets the bounds for each Parameter. See: SetVarying.

        Args:
            **kwargs: see SetVarying for keys. Each value is a bounds tuple
        """
        for k,v in kwargs.items():
            self.ParamDict[k].Bounds = v
    def GetParamDict(self):
        """
        Returns: in-order copy dictionary of the parameters
        """
        toRet = OrderedDict(self.ParamDict.items())
        return toRet
    def GetValueDict(self):
        """
        Returns an ordered dictionary of the parameter *values*
        """
        return OrderedDict( (k,v.Value) for k,v in self.GetParamDict().items())

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
    def _Scale(self,**kwargs):
        for k,v in kwargs.items():
            self.ParamDict[k].Scale(v)
    def ScaleGen(self,xScale,ForceScale):
        """
        Scales the data to an x and y scale given by xScale and ForceScale.
        
        In other words, x -> x/xScale, etc

        Args:
            xScale: What to divide the distance parts by
            yScale: What to divide the force parts by
        """
        # lengths are distances
        Scales = dict(L0=xScale,
                      Lp=xScale,
                      K0=ForceScale,
                      kbT=ForceScale*xScale)
        self._Scale(**Scales)
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
    def __str__(self):
        return "\n".join(["{:s}={:s}".format(k,v)
                          for k,v in self.GetParamDict().items()])

class Initialization:
    """
    Class to keep track of how we want to initialize
    """
    GUESS = 0
    HOP = 1
    BRUTE = 2
    def __init__(self,Type=GUESS,*params,**kwargs):
        self.Type = Type
        self.Args = params
        self.ParamDict = dict(**kwargs)
        # initialization info is set when we
        # initialize everythign by the actual method
        self.InitializationInfo = None
    def SetInitializationInfo(self,**kwargs):
        """
        Sets the initialization information dictionary
        """
        self.InitializationInfo = dict(**kwargs)
        
class WlcFitInfo:
    def __init__(self,Model=WLC_MODELS.EXTENSIBLE_WANG_1997,
                 ParamVals=WlcParamValues(),
                 Initialization=Initialization()):
        """
        Args:
        Model: which model, from WLC_MODELS, to use
        ParamValues: values of the parameters for the fit (e.g. initial guesses,
        or final resuts )

        VaryObj: which parameters should be varied for the 
        """
        self.Model = Model
        self.ParamVals = ParamVals
        self.Initialization = Initialization
    """
    The Following are helper functions that only make sense
    in the context of 'AddParamsGen', which lets us get
    varying and fixed bounds, values, and the like
    """
    @property
    def _VaryCondition(self):
        """
        Return true if we should vary x (If its flag is Low)
        """
        return lambda x: x
    @property
    def _FixedCondition(self):
        """
        Return true if we shouldn't vary x (If its flag is False)
        """
        return lambda x: not x
    @property
    def _AddValue(self):
        """
        Function to get the value of a parameter
        """
        return lambda x: x.Value
    def _AddBounds(self):
        """
        Functon to get the bounds of a parameter
        """
        return lambda x: x.Bounds.AsTuple()
    def AddParamsGen(self,condition,AddFunction):
        """
        General function to add paraters to a list we return.

        Args: 
            condition: function taking in a flag if we should vary the object,
            and returning true/false if it should be added to the lisst

            AddFunction: function taking in an object, returning what we want
            to add
        Returns:
            list of elememnts we added by condition
        """
        toRet = OrderedDict()
        # assume we never vary temperature
        vals = self.ParamVals.GetParamDict()
        # XXX could generalize, make toVary have L0, just use lambdas everywhere
        for key,val in vals.items():
            if (condition(val.Vary)):
                valToAdd = AddFunction(val)
                toRet[key] = valToAdd
        return toRet
    def GetVaryingParamDict(self):
        """
        Gets the (ordered) dictionary of varying parameters
        """
        toRet= self.AddParamsGen(self._VaryCondition,self._AddValue)
        return toRet
    def GetFixedParamDict(self):
        """
        Gets the (ordered) dictionary of fixed parameters (assumes temperature 
        is amoung)
        """
        toRet= self.AddParamsGen(self._FixedCondition,self._AddValue)
        return toRet
    def GetVaryingBoundsDict(self,InfValue=None):
        """
        Gets the (ordered) dictionary of varying parameters, dealign with 
        open bounds by returnign InfValue

        Args:
            InfValue: value to return for an open bound
        """
        return self.AddParamsGen(self._VaryCondition,self._AddBounds())
    def DictToValues(self,Dict):
        return [v for key,v in Dict.items()]
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
    return GetBoundsDict(L0=TupleL0,
                         Lp=TupleLp,
                         K0=TupleK0,
                       # Note that we typically dont fit temperature,
                       # really no way to know.
                         kbT=[0,np.inf])

def GetBoundsDict(**kwargs):
    """
    Utility function: given tuples, returns a dictionary of bounds objects

    Args:
        kwargs: keys are parameter names, values are tuple of upper and lower 
        bounds for each parameter
    Returns:
        Ordered dictionary of <Paramter>:<Bounds>
    """
    toRet = OrderedDict()
    for k,v in kwargs.items():
        # assume that v is a tuple
        toRet[k] = BoundsObj(*v)
    return toRet

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

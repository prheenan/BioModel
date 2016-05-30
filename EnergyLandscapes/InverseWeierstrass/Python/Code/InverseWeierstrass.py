# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.integrate import cumtrapz
import itertools
from collections import defaultdict


class FEC_Pulling_Object:

    def __init__(self,Time,Extension,Force,StringConstant=0.4e-3,
                 ZFunc=None,
                 Velocity=20e-9,Beta=1./(4.1e-21)):
        """
        Args:
            Time: Time, in seconds
            Extension: Extension[i] as a function of time[i]
            Force: Force[i] as a function of time[i]
            SpringConstant: Force per distance, SI units (N/m). Default from 
            see pp 634, 'Methods' of : 
            Gupta, Amar Nath, Abhilash Vincent, Krishna Neupane, Hao Yu, 
            Feng Wang, and Michael T. Woodside. 
            "Experimental Validation of Free-Energy-Landscape Reconstruction 
            from  Non-Equilibrium Single-Molecule Force Spectroscopy 
            Measurements." 
            Nature Physics 7, no. 8 (August 2011)

            ZFunc: Function which takes in an FEC_Pulling_Object
            and returns a list of z values at each time. If none, defaults
            to simple increase from first extension
        
            Velocity: in m/s, default from data from ibid.
            Beta: 1/(kbT), defaults to room temperature (4.1 pN . nm)
        """
        self.Time = Time
        self.Extension = Extension
        self.Force = Force
        self.SpringConstant=StringConstant
        self.Velocity= Velocity
        self.Beta=Beta
        if (ZFunc is None):
            ZFunc = lambda Obj: Obj.Extension[0] + (Obj.Velocity * Obj.Time)
        self.ZFunc = ZFunc
        self.Work=None
        self.WorkDigitized=None
    def GetBias(self):
        """
        Returns the bias associated with the given extension, force, and
        strong constant

        Args:
            ZFunc: See GetDigitizedBias 
        """
        return self.ZFunc(self)-self.Extension
    def GetWorkArgs(self,ZFunc):
        """
        Gets the in-order arguments for the work functions

        Args:
            ZFunc: see GetDigitizedBias
        """
        return self.SpringConstant,self.Velocity,self.Time,self.Extension
    def CalculateForceCummulativeWork(self):
        """
        Gets the position-averaged work, see methods section of 
        paper cited in GetDigitizedBias
         
        Args:
            ZFunc: See GetDigitizedBias
        Returns:
            The cummulative integral of work, as defined in ibid, before eq18
        """
        # get the z (position of probe)
        Z = self.ZFunc(self)
        # get the bias (z-q)
        Bias = self.GetBias()
        # compute the forst from the bias 
        Force = self.SpringConstant * Bias
        ToRet = cumtrapz(x=Z,y=Force,initial=0)
        return ToRet
    def SetWork(self,Work):
        self.Work = Work
    def _GetDigitizedGen(self,BinExt,BinTime,ToDigitize):
        """
        Generalized method to get 'digitized' results

        Args:
            BinTime: see GetDigitizedWork
            BinExt: see  GetDigitizedWork
            ToDigitize: Array to dizitize, e.g. self.Forces
        Returns:
            See GetDigitizedWork, except digitized contents of 'ToDigitize'
        """
        NumTimes = BinTime.size
        NumExt = BinExt.size
        IdxExtArr = np.digitize(self.Extension,bins=BinExt)-1
        IdxTimeArr = np.digitize(self.Time,bins=BinTime)-1
        DigitzedMatrix = defaultdict(lambda : defaultdict(list))
        for i,idx_ext in enumerate(IdxExtArr):
            idx_time = IdxTimeArr[i]
            DigitzedMatrix[idx_ext][idx_time].append(ToDigitize[i])
        return DigitzedMatrix
    def GetDigitizedBoltzmann(self,BinExt,BinTime):
        """
        Gets the digitized boltzmann factor np.exp(-beta*W)
        (averaged in a bin), given the Bins

        Args:
            BinTime: the time binning to use
            BinExt: the extension binning to use
        Returns: 
            The digitized boltman matrix exp(-Beta*W), 
            where W[i][j] is a *list* of Work values associated with BinExt[i]
            and BinTime[j]
        """
        ToDigitize = np.exp(-self.Beta*self.Work)
        return self._GetDigitizedGen(BinExt,BinTime,ToDigitize)
    def GetDigitizedBias(BinExt,BinTime,ZFunc):
        """
        Gets the digitized bias (q-z), See equation 11 and 'Methods' (above 18)
        of:
        
        Hummer, Gerhard, and Attila Szabo. 
        "Free Energy Profiles from Single-Molecule Pulling Experiments." 
        Proceedings of the National Academy of Sciences 
        107, no. 50 (December 14, 2010)

        Args:
            BinTime, BinExt: see GetDigitizedBoltzmann
            ZFunc: Function taking in a time array
            for the position of the bead, typically z(t) = z0 + v(t)
        Returns:
            see GetDigitizedBoltzmann, except Bias is the content
        """
        z = ZFunc(self.Time)
        ToDigitize = self.Extension - z
        return self._GetDigitizedGen(BinExt,BinTime,ToDigitize)
    def GetDigitizedForce(self,BinExt,BinTime):
        """
        Gets the digitized force within the bins. See materials cited in 
        GetDigitizedBias

        Args:
            BinTime,BinExt: see ibid
        Returns:
            see GetDigitizedBoltzmann, except Force is the content
        """
        return self._GetDigitizedGen(BinExt,BinTime,self.Force)

def CummulativeWorkIntegral(Z,Force):
    """
    Get the integral of the extension from t=0 to t=1,2,3,....etc
    
    See last part of 'W_t' equation' Paper cited in "GetWork"

    Args:
        Z: the probe position as a function of time

        Force: the force to integrate along z
    Returns:
        cummulative integral of z from z0 to z
    """
    return cumtrapz(x=Z,y=Force,initial=0)


def SetAllWorkOfObjects(PullingObjects):
    """
    Gets the work associated with each force extension curve in PullingObjects

    Args:
        PullingObjects: list of FEC_Pulling_Object
    Returns:
        Nothing, but sets work as a function of all time for each element
        in PullingObjects
    """
    # calculate and set the work for each object
    _ = [o.SetWork(o.CalculateForceCummulativeWork())
         for o in PullingObjects]

def _GetGenBounds(PullingObjects,FuncLower,FuncUpper):
    """
    Function to get the General bounds from a list of objects

    Args:
        PullingObjects: see SetAllWorkOfObjects
        FuncLower: takes in a single object, gets the lower bound
        FuncUpper: takes ina a single object, gets the Upper bound
    Returns:
        Tuple of absolute minimum and maximum
    """
    LowerArr = [FuncLower(o) for o in PullingObjects]
    UpperArr = [FuncUpper(o) for o in PullingObjects]
    return min(LowerArr),max(UpperArr)
    
def GetTimeBounds(PullingObjects):
    """
    Given a list of pulling objects, gets the extrema of the time bounds

    Args:
        PullingObjects: see SetAllWorkOfObjects
    Returns:
        Tuple of <Lower,Upper> bounds on the time
    """
    return _GetGenBounds(PullingObjects,
                         # lower bound is just the first time point
                         lambda x: x.Time[0],
                         # upper bound is the last time point
                         lambda x: x.Time[-1])

def GetExtensionBounds(PullingObjects):
    """
    See GetTimeBounds, except gets the extesnion bounds

    Args:
        PullingObjects: see SetAllWorkOfObjects
    Returns:
        Tuple of <Lower,Upper> bounds on the extensions
    """
    return _GetGenBounds(PullingObjects,
                         # lower bound is just the first extension
                         lambda x: x.Extension[0],
                         # upper bound is the last extension
                         lambda x: x.Extension[-1])


def HarmonicPotential(Stiffness,Velocity,Times,Extensions):
    """
    Modeled after potential in methods of paper cited in "GetWork".
    Using that notation

    Args:
        Stiffness,Velocity,Times,Extensions: See "GetWork"
    Returns:
        Relevant Potential
    """
    # z is the center position of the bead
    z = Extensions[0] + Velocity*Times
    # x is the difference between molecular (q) and bead position (z)
    x = Extensions-z
    Potential = (Stiffness/2) * x**2
    return Potential

def FreeEnergyAtZeroForceWeightedHistogram(Beta,MolExtesionBins,TimeBins,
                                           BoltzmannMatrix,Potential,
                                           Stiffness,Velocity):
    """
    Gets the free energy at zero force

    See last part of 'W_t' equation' Paper cited in "GetWork"

    Args:
        Beta: 1/(k_b*T), units of 1/[Work] 
        MolExtensionBins : MolExtensionTimesSeries[i] is the ith (binnned)
        extension.

        TimeBins: list, i-th element is the ith time
    
        BoltzmannMatrix: Double nested default dictionary, 
        element [i][j] is the average boltzmann factor at time [i], 
        extension [j]

        Potential: Potential function, with signature 
        (Stiffness,Velocity,Times,Extensions), returning the appropriate 
        potential

        Stiffness,Velocity: See GetWork
    Returns:
        Free Energy G. G[i] is the free energy at 
        zero force at MolExtesionBins q[i]
    """
    # conver the Work matrix into a matrix where we have element [l][k]
    # being extension l, time k
    # to do this, we essentially just swap the keys
    TxFunc = lambda Dict,k1,v1,k2,v2 : Dict[k2].update(k1=v2)
    BoltzmannMatrixByTime = BoltzmannMatrix
    BoltzmannMatrixByExt = TransformNestedDefaultDict(BoltzmannMatrix,float,
                                                      TxFunc)
    # for each time, get the mean parition function associated
    Partition =np.array([ np.mean(BoltzmannMatrixByTime[i].values())
                          for i,_ in enumerate(TimeBins)])
    # for each an extension ('j'), get the average across all times
    SumTerms = [np.sum(BoltzmannMatrixByExt[j].values()/Partition[j])
                        for j,_ in enumerate(MolExtesionBins)]
    # for each extension, get the denominator term (weighted potential)
    PotentialWeighted = lambda q: np.exp(-Beta*Potential(Stiffness,Velocity,q,
                                                         TimeBins))
    PotentialWeight = [sum(PotentialWeighted(q)/Partition)
                       for q in MolExtesionBins]
    Numer = SumTerms/Partition
    Denom = PotentialWeight
    # get the free energy at all q
    FreeEnergyAtZeroForce = -np.log(np.array(Numer)/np.array(Denom))/Beta
    return FreeEnergyAtZeroForce

def TransformNestedDefaultDict(Original,ElementFunc,TxFunction):
    return TransformListOfNestedDefaultDict([Original],ElementFunc,TxFunction)

def TransformListOfNestedDefaultDict(Original,ElementFunc,TxFunction):

    """
    Given a douby-nested dictionary of lists and a transformation function,
    returns a default dictionary by transforming

    Args:
        Original: a *list* of dictionaries to concatenate
        ElementFunc: what the individual elements are (ie: should be
        only argument passed to inner defaultdict; what are we actually
        storing in the dictionary?). Eg: 'list' or 'float'

        TxFunction: Function with signature <dictionary,k1,v1,k2,v2>
        where k<xx> is key <xx> and v<xx> is value xx. dictionary is the
        same one we return
    Returns:
        a doubly-nested default dictionary of lists 
    """
    # note this looks bad (triply nested), but because we
    # are using default dictionaries, it is linear in the
    # total number of points (ie: we are just regrouping all the
    # individual FEC into a matrix)
    ToRet = defaultdict(lambda : defaultdict(ElementFunc))
    for o in Original:
        for k1,v1 in o.items():
            for k2,v2 in v1.items():
                TxFunction(ToRet,k1,v1,k2,v2)
    return ToRet

def DictUpdateFunc(f,DictV,k1,v1,k2,v2):
    DictV[k1][k2] = f(v2)


def FlattenNestedDefaultDict(NestedDict,Func):
    """
    Given a doubly-nested dictionary an a transofrmation function, 
    returns a flatted, transformed original array

    Args:
        NestedDict: The doubly nested dictionary. Each element of the first
        level is a dictionary; we apply Func to each of the values in this 
        dicitonary

        Func: Function taking a list of values of a dictionary and combining 
        them.
    Returns:
    """
    return np.array([ Func([v2 for k2,v2 in v1.items()])
                      for k1,v1 in NestedDict.items()])

def GetBoltzmannWeightedAverage(BoltzmannDefaultDict,
                                ValueDict):
    """
    Given a matrix BoltzmannDefaultDict[i][j][k] where i refers to
    the extenion, j refers to the time, and k refers to the FEC label,
    and a matrix ValueDict[i][j][k] with the same index but referring
    to a physical constant (e.g. force or force squared), returns
    the boltzmann-weighted force a la Hummer2010, just after Equation 11

    Args:
         BoltzmannDefaultDict: douly-nested default dict; element are 
         list of Boltzmann factor values
    
         ValueDict: doubly=nested default dict with same ordering
         as BoltzmannDefaultDict. 
    Returns:
         Weighted average as a function of extension, as an *array*
    """
    NumExt = len(BoltzmannDefaultDict.keys())
    # create an array for storing everything.
    BoltzmannWeightedAverage = np.zeros(NumExt)
    Partition = np.zeros(NumExt)
    # loop through each extension and get the weights
    for i,(BoltzKey,BoltzVals) in enumerate(BoltzmannDefaultDict.items()):
        # get every boltzmann factor present
        keys = BoltzVals.keys()
        TmpBoltzmannFactors = [item for k in keys for item in BoltzVals[k]]
        # get every value present
        TmpValues = [item for k in keys for item in ValueDict[BoltzKey][k]]
        Partition[i] = np.mean(TmpBoltzmannFactors)
        # get the mean, weighted value at this z
        WeightedVals = np.mean(np.array(TmpBoltzmannFactors) * \
                               np.array(TmpValues))
        # get the weighted average at this z
        BoltzmannWeightedAverage[i] = WeightedVals/Partition[i]
    return BoltzmannWeightedAverage,Partition

def CombineNestedDicts(DigitizedMatrices):
    # create a function that given a dictionary and the two keys,
    # extends the existing dictionary list with the list in that bin.
    # essentially this just concatenates all the FEC into a single list
    CombineFunction = lambda DictV,k1,_,k2,v2: DictV[k1][k2].extend(v2)
    FlatDigitized = TransformListOfNestedDefaultDict(DigitizedMatrices,list,
                                                     CombineFunction)
    return FlatDigitized

def GetBoltmannWeightedQuantity(Objs,FlatBoltzmann,FuncForValues):
    ValueMatrix = [FuncForValues(o) for o in Objs]
    FlattenedValueMatrix = CombineNestedDicts(ValueMatrix)
    Weighted,_ = GetBoltzmannWeightedAverage(FlatBoltzmann,
                                             FlattenedValueMatrix)
    return Weighted

def FreeEnergyAtZeroForce(Objs,NumTimeBins,NumPositionBins):
    """
    Wrapper to make it easier to get the weighted histograms, etcs.

    Args:
        obj: list of FEC_Pulling_Object
    """
    # get the bounds associated with the times and extensions
    ExtBounds = GetExtensionBounds(Objs)
    TimeBounds = GetTimeBounds(Objs)
    # Create the time and position bins using a helper function
    BinIt= lambda x,n: np.linspace(start=x[0],
                                   stop=x[1],
                                   endpoint=False,
                                   num=n)
    # create the bins...
    ExtBins = BinIt(ExtBounds,NumPositionBins)
    TimeBins = BinIt(TimeBounds,NumTimeBins)
    # Get the time-digitized boltmann factors for each FEC
    DigitizedMatrices = [o.GetDigitizedBoltzmann(ExtBins,TimeBins)
                         for o in Objs]
    FlatDigitized = CombineNestedDicts(DigitizedMatrices)
    ## POST: FlatDigitizes has the (flattened) array, where element [i][j]
    ## is the *list* of Boltzmann factors at extension i, time j (where i and
    ## j are indices into the respective bins)
    #get the boltzmann-averaged things we need
    GetForceFunc = lambda x : x.GetDigitizedForce(ExtBins,TimeBins)
    GetForceSqFunc = lambda x : x._GetDigitizedGen(ExtBins,TimeBins,
                                                   x.Force**2)
    BoltzmannWeightedForce = GetBoltmannWeightedQuantity(Objs,FlatDigitized,
                                                         GetForceFunc)
    BoltzmannWeightedForceSq = GetBoltmannWeightedQuantity(Objs,FlatDigitized,
                                                           GetForceSqFunc)
    """
    for the second derivative, we really just want the
    variance at each z,  see equation 12 of
    Hummer, Gerhard, and Attila Szabo. 
    "Free Energy Profiles from Single-Molecule Pulling Experiments."
    Proceedings of the National Academy of Sciences 107, no. 50 
    (December 14, 2010)
    """
    VarianceForceBoltzWeighted = BoltzmannWeightedForceSq-\
                                 (BoltzmannWeightedForce**2)
    # now get the free energy from paragraph before eq18, ibid.
    # This is essentially the ensemble-averaged 'partition function' at each z
    _,Parition = GetBoltzmannWeightedAverage(FlatDigitized,
                                             FlatDigitized)
    Beta = np.mean([o.Beta for o in Objs])
    SpringConst = np.mean([o.SpringConstant for o in Objs])
    k = SpringConst
    FreeEnergy_A = (-1/Beta)*np.log(Parition)
    # write down the terms involving the first and second derivative of A
    dA_dz = BoltzmannWeightedForce.copy()
    # for the second derivative, just use 1-A''/k
    SecondDerivTerm = Beta *VarianceForceBoltzWeighted/k
    # perform the IWT, ibid equation 10
    FreeEnergyAtZeroForce = FreeEnergy_A - (dA_dz)**2/(2*k) + \
                            (1/(2*Beta)) * np.log(SecondDerivTerm)
    # write down q, using ibid, 10, argument to G0
    # XXX should fix...
    q = ExtBins-dA_dz/k
    # shift the free energy to F_1/2
    # approximately 20pN, see plots
    F0 = 21.5e-12
    q = ExtBins-dA_dz/k
    FreeEnergyAtF0_kbT = (FreeEnergyAtZeroForce-ExtBins*F0)*Beta
    n=3
    fig = plt.figure(figsize=(5,7))
    # plot energies in units of 1/Beta (kT), force in pN, dist in nm
    plt.subplot(n,1,1)
    for o in Objs:
        plt.plot(o.Extension*1e9,o.Force*1e12)
    plt.subplot(n,1,2)
    plt.plot(ExtBins*1e9,FreeEnergyAtZeroForce*Beta)
    plt.subplot(n,1,3)
    plt.plot(ExtBins*1e9,FreeEnergyAtF0_kbT)
    plt.tight_layout()
    plt.show()

    return FreeEnergyAtZeroForce

    


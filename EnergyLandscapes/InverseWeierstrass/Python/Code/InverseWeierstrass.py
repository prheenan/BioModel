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

    def __init__(self,Time,Extension,Force,StringConstant=0.23e-3,
                 Velocity=10e-9,Beta=1./(4.1e-21)):
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

            Velocity: in m/s, default from data from ibid.
            Beta: 1/(kbT), defaults to room temperature (4.1 pN . nm)
        """
        self.Time = Time
        self.Extension = Extension
        self.Force = Force
        self.SpringConstant=StringConstant
        self.Velocity= Velocity
        self.Beta=Beta
        self.Work=None
        self.WorkDigitized=None
    def GetWorkArgs(self):
        """
        Gets the in-order arguments for the work functions
        """
        return self.SpringConstant,self.Velocity,self.Time,self.Extension
    def SetMyWork(self):
        self.Work = GetWork(self)
    def GetDigitizedWork(self,BinTime,BinExt):
        """
        Gets the digitized work (averaged in a bin), given the Bins

        Args:
            BinTime: the time binning to use
            BinExt: the extension binning to use
        Returns: 
            The digitized boltman matrix exp(-Beta*W), 
            where W[i][j] is a *list* of Work values associated with BinTime[i]
            and BinExt[j]
        """
        NumTimes = BinTime.size
        NumExt = BinExt.size
        IdxExtArr = np.digitize(self.Extension,bins=BinExt)-1
        IdxTimeArr = np.digitize(self.Time,bins=BinTime)-1
        DigitzedMatrix = defaultdict(lambda : defaultdict(list))
        BoltzmanFactors = np.exp(-self.Beta*self.Work)
        for i,idx_ext in enumerate(IdxExtArr):
            idx_time = IdxTimeArr[i]
            DigitzedMatrix[idx_time][idx_ext].append(BoltzmanFactors[i])
        return DigitzedMatrix


def CummulativeWorkIntegral(Times,Extension):
    """
    Get the integral of the extension from t=0 to t=1,2,3,....etc
    
    See last part of 'W_t' equation' Paper cited in "GetWork"

    Args:
        Times: array of times to cummulatively integrate the extension. Units of
        [Work]/([stifffness]*[velocity]*[Extension]

        Extension: array of extensions to integrate 
    Returns:
        integral of z from t=0 to t=tau, for tau=[0,...,n], where n is 
        len(Times)
    """
    return cumtrapz(x=Times-Times[0],y=Extension,initial=0)

def GetWork(Obj):
    """
    see pp 634, 'Methods' of : 
    Gupta, Amar Nath, Abhilash Vincent, Krishna Neupane, Hao Yu, Feng Wang, 
    and Michael T. Woodside. 
    "Experimental Validation of Free-Energy-Landscape Reconstruction from 
    Non-Equilibrium Single-Molecule Force Spectroscopy Measurements." 
    Nature Physics 7, no. 8 (August 2011)

    Args:
        Obj: instance of FEC_Pulling_Object
    """
    Stiffness = Obj.SpringConstant
    Times = Obj.Time
    Extensions = Obj.Extension
    Velocity = Obj.Velocity
    coeff = Stiffness*Velocity
    TimeDependent = coeff * (1/2 * Velocity * (Times**2) + \
                             Extensions[0]*Times)
    # now we so the cummulative intergral
    Cummulative = -coeff * CummulativeWorkIntegral(Times,Extensions)
    # add up all the work compontents
    TotalWork = Cummulative+TimeDependent
    return TotalWork


def SetAllWorkOfObjects(PullingObjects):
    """
    Gets the work associated with each force extension curve in PullingObjects

    Args:
        PullingObjects: list of FEC_Pulling_Object
    Returns:
        Nothing, but sets work as a function of all time for each element
        in PullingObjects
    """
    [o.SetMyWork() for o in PullingObjects]

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


def FreeEnergyWeightedHistogramByObject(Objs,NumTimeBins,NumPositionBins):
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
    DigitizedMatrices = [o.GetDigitizedWork(TimeBins,ExtBins) for o in Objs]
    # create a function that given a dictionary and the two keys,
    # extends the existing dictionary list with the list in that bin.
    # essentially this just concatenates all the FEC into a single list
    CombineFunction = lambda DictV,k1,_,k2,v2: DictV[k1][k2].extend(v2)
    FlatDigitized = TransformListOfNestedDefaultDict(DigitizedMatrices,list,
                                                     CombineFunction)
    ## POST: FlatDigitizes has the (flattened) array, where element [i][j]
    ## is the *list* of Boltzmann factors at time i, element j (where i and
    ## j are indices into the respective bins)
    # get their average and standard deviation for each extension and time
    # create functions to take the *flattened* dictionary (ie: each element
    # is a list) and get the mean/std of the list
    
    MeanFunc = lambda *args : DictUpdateFunc(np.mean,*args)
    StdFunc = lambda *args : DictUpdateFunc(np.std,*args)
    # mean in each bin
    MeanDigitized = TransformNestedDefaultDict(FlatDigitized,float,MeanFunc)
    StdDigitized = TransformNestedDefaultDict(FlatDigitized,float,StdFunc)
    #XXX assume that all betas are the same
    #XXX assume specific potential, given by Gupta2011 
    Potential = HarmonicPotential
    Beta = np.mean([o.Beta for o in Objs])
    Stiffness = np.mean([o.SpringConstant for o in Objs])
    Velocity =  np.mean([o.Velocity for o in Objs])
    FreeEnergyAtZeroForce =  \
        FreeEnergyAtZeroForceWeightedHistogram(Beta,ExtBins,TimeBins,
                                               MeanDigitized,Potential,
                                               Stiffness,Velocity)
    F0 = 12e-12
    FreeEnergyAtF0 = FreeEnergyAtZeroForce - ExtBins*F0
    plt.subplot(2,1,1)
    plt.plot(ExtBins,FreeEnergyAtZeroForce)
    plt.subplot(2,1,2)
    plt.plot(ExtBins,FreeEnergyAtF0)
    plt.show()
    # stdev in each bin
    


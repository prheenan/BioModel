# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.integrate import cumtrapz
import itertools


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
        self.Work = GetWork(*self.GetWorkArgs())
    def GetDigitizedWork(self,Bins):
        """
        Gets the digitized work (averaged in a bin), given the Bins

        Args:
            Bins: the position binning to use
        Returns: 
            The digitized work W, where W[i] is a list of Work values associated
            with Bins[i]
        """
        N = Bins.size
        IdxBinned = np.digitize(self.Extension,bins=Bins)-1
        # slow, but works: get a list of list, append all the works.
        ToRet = [[] for i in range(N)]
        for i,idx in enumerate(IdxBinned):
            print(self.Work[i])
            print(idx,len(ToRet))
            ToRet[idx].append(self.Work[i])
        print(ToRet)
        return ToRet


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
    return cumtrapz(x=Times,y=Extension,initial=0)

def GetWork(Stiffness,Velocity,Times,Extensions):
    coeff = Stiffness*Velocity
    """
    see pp 634, 'Methods' of : 
    Gupta, Amar Nath, Abhilash Vincent, Krishna Neupane, Hao Yu, Feng Wang, 
    and Michael T. Woodside. 
    "Experimental Validation of Free-Energy-Landscape Reconstruction from 
    Non-Equilibrium Single-Molecule Force Spectroscopy Measurements." 
    Nature Physics 7, no. 8 (August 2011)
    """
    TimeDependent = coeff * (1/2 * Velocity * Times**2 + \
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
    
def _WeightedHistogramNumerator(Beta,MolExtensionTimesSeries,WorkTimeSeries,q,
                                tol=1e-15):
    """
    Args:
        beta: see FreeEnergyAtZeroForceWeightedHistogram
        extensions are within this value, we consider them equal for the 
        purposes of the Delta function
        MolExtensionTimesSeries: see FreeEnergyAtZeroForceWeightedHistogram
        WorkTimeSeries: see FreeEnergyAtZeroForceWeightedHistogram
        q: desired extension
        tol: tolerance, in absolute extension, for the binning. If two 
    """
    WhereEqual = np.where(np.abs(MolExtensionTimesSeries-q) < tol)
    Partition = WorkTimeSeries[WhereEqual]
    Delta = np.exp(-Beta*WorkTimeSeries[WhereEqual])
    return sum(Delta/Partition)

def _WeightedHistogramDenominator(Beta,WorkTimeSeries,Potential,q):
    """
    Args:
        Beta: see FreeEnergyAtZeroForceWeightedHistogram
        WorkTimeSeries: see FreeEnergyAtZeroForceWeightedHistogram
        Potential: functiomn taking in q,t, returning an energy, units of 1/Beta
        q: see WeightedHistogramNumerator
    """
    Numer = np.exp(-Beta*Potential(q,WorkTimeSeries))
    Denom = WorkTimeSeries
    AllSumsTerms = Numer/Denom
    return sum(AllSumTerms)

def FreeEnergyAtZeroForceWeightedHistogram(Beta,Times,MolExtensionTimesSeries,
                                           WorkTimeSeries,Potential):
    """
    Gets the free energy at zero force

    See last part of 'W_t' equation' Paper cited in "GetWork"

    Args:
        Beta: 1/(k_b*T), units of 1/[Work] 
        Times: Times[i] is the times for WorkTimeSeries[i] (W_t in ibid)
        MolExtensionTimesSeries : MolExtensionTimesSeries[i] is the (binnned)
        extension at Times[i]. (q_t in ibid)
    
        WorkTimeSeries: Work at time t. See ibid
        Potential: Potential function, taking in *both* an extension and a time
    Returns:
        Tuple of <Extension q, Free Energy G>. G[i] is the free energy at 
        zero force at extenson q[i]
    """
    allQ = sorted(list(set(MolExtensionTimesSeries)))
    denom = [_WeightedHistogramDenominator(Beta,WorkTimeSeries,q) for q in allQ]
    numer = [_WeightedHistogramNumerator(Beta,MolExtensionTimesSeries,
                                         WorkTimeSeries,q) for q in allQ]
    # get the free energy at all q
    FreeEnergyAtZeroForce = np.log(np.array(denom)/np.array(numer))/Beta
    return allQ,FreeEnergyAtZeroForce


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
    # Get the time-digitized work for each FEC
    DigitizedWork = [o.GetDigitizedWork(ExtBins) for o in Objs]
    # flatten all the works
    WorkByExtenson = [ [Work[i] for Work in DigitizedWork ]
                       for i,_ in enumerate(ExtBins)]
    FlattenedWork = [np.array(list(itertools.chain(*a)))
                     for a in WorkByExtenson ]
    # POST: FlattenedWork now constants the work values in each bin.
    Beta = 1/(4.1e-21)
    plt.boxplot(FlattenedWork)
    plt.show()
    
    
    


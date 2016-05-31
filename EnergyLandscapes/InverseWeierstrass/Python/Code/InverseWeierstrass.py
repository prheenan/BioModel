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
    def _GetDigitizedGen(self,BinTime,ToDigitize):
        """
        Generalized method to get 'digitized' results

        Args:
            BinTime: see GetDigitizedBoltzmann
            ToDigitize: Array to dizitize, e.g. self.Forces
        Returns:
            See GetDigitizedBoltzmann, except digitized contents of 
            'GetDigitizedBoltzmann'
        """
        NumTimes = BinTime.size
        IdxTimeAdd = np.digitize(self.Time,bins=BinTime)-1
        DigitzedMatrix = [[] for _ in range(NumTimes)]
        for i,idx_time in enumerate(IdxTimeAdd):
            DigitzedMatrix[idx_time].append(ToDigitize[i])
        return DigitzedMatrix
    def GetDigitizedBoltzmann(self,BinTime):
        """
        Gets the digitized boltzmann factor np.exp(-beta*W)
        (averaged in a bin), given the Bins

        Args:
            BinTime: the time binning to use
        Returns: 
            The digitized boltman matrix exp(-Beta*W), 
            where W[i] is a *list* of Work values associated with BinTime[i]
        """
        ToDigitize = np.exp(-self.Beta*self.Work)
        return self._GetDigitizedGen(BinTime,ToDigitize)
    def GetDigitizedBias(BinTime,ZFunc):
        """
        Gets the digitized bias (q-z), See equation 11 and 'Methods' (above 18)
        of:
        
        Hummer, Gerhard, and Attila Szabo. 
        "Free Energy Profiles from Single-Molecule Pulling Experiments." 
        Proceedings of the National Academy of Sciences 
        107, no. 50 (December 14, 2010)

        Args:
            BinTime: see GetDigitizedBoltzmann
            ZFunc: Function taking in a time array
            for the position of the bead, typically z(t) = z0 + v(t)
        Returns:
            see GetDigitizedBoltzmann, except Bias is the content
        """
        z = ZFunc(self.Time)
        ToDigitize = self.Extension - z
        return self._GetDigitizedGen(BinTime,ToDigitize)
    def GetDigitizedForce(self,BinTime):
        """
        Gets the digitized force within the bins. See materials cited in 
        GetDigitizedBias

        Args:
            BinTime: see ibid
        Returns:
            see GetDigitizedBoltzmann, except Force is the content
        """
        return self._GetDigitizedGen(BinTime,self.Force)

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

def GetBoltzmannWeightedAverage(BoltzmannFactors,
                                Values,
                                Partition):
    """
    Given a matrix BoltzmannFactors[i][j] where i refers to
    the time, and j refers to the FEC label,
    and a matrix Values[i][j] with the same index but referring
    to a physical constant (e.g. force or force squared), returns
    the boltzmann-weighted force a la Hummer2010, just after Equation 11

    Args:
         BoltzmannFactors: nested list; element [i] is a list of in-order
         emsemble boltzman factors associated with all measuements in the 
         ensemble
    
         Values: nested list; same ordering as BoltzmannDefault
         Partition: Partition function, mean value of the boltzmann factor
         at each time
    Returns:
         Weighted average as a function of extension, as an *array*, defined
         by Hummer and Szabo, 2010, PNAS, after equation 12
    """
    # get the mean boltzmann factor (partition function) at each bin
    NumExt = len(Partition)
    # create an array for storing everything.
    BoltzmannWeightedAverage = np.zeros(NumExt)
    # loop through each extension and get the weights
    for i,BoltzVals in enumerate(BoltzmannFactors):
        # get the associated values
        TmpValues = Values[i]
        # get the mean, weighted value at this z
        WeightedVals = np.array(BoltzVals) * np.array(TmpValues)
        # get the weighted average at this z
        BoltzmannWeightedAverage[i] = np.mean(WeightedVals)
    return BoltzmannWeightedAverage/Partition

def FreeEnergyAtZeroForce(Objs,NumTimeBins):
    """
    Wrapper to make it easier to get the weighted histograms, etcs.

    Args:
        obj: list of FEC_Pulling_Object
    """
    # get the bounds associated with the times and extensions
    TimeBounds = GetTimeBounds(Objs)
    # Create the time and position bins using a helper function
    BinIt= lambda x,n: np.linspace(start=x[0],
                                   stop=x[1],
                                   endpoint=False,
                                   num=n)
    # create the bins...
    TimeBins = BinIt(TimeBounds,NumTimeBins)
    # Set up functions for getting the force and boltzmann factors
    BoltzmanFunc = lambda o : o.GetDigitizedBoltzmann(TimeBins)
    ForceFunc = lambda o: o.GetDigitizedForce(TimeBins)
    GetForceSqFunc = lambda x : x._GetDigitizedGen(TimeBins,
                                                   x.Force**2)
    # get the (per-instance) boltmann factors, for weighing
    BoltzByFEC = [BoltzmanFunc(o) for o in Objs]
    # get the (flattend) boltmann factors
    FlatFunc = lambda objs,bins : [ [item
                                     for x in objs
                                     for item in x[i]]
                                    for i in range(bins)]
    BoltzHistogramByTime = FlatFunc(BoltzByFEC,NumTimeBins)
    Partition = np.array([np.mean(b) for b in BoltzHistogramByTime])
    """
    Get the mean boltzmann factor, <exp(-Beta*W(z))>, like a partition
    function, see 

    Hummer, Gerhard, and Attila Szabo. 
    "Free Energy Profiles from Single-Molecule Pulling Experiments."
    Proceedings of the National Academy of Sciences 107, no. 50 
    (December 14, 2010)
    
    Especially  right after equation 12 (<exp(-Beta(W(z)))> is the denominator)
    """
    # do the same, for the force and force squared
    ForcePerEnsemble = [ForceFunc(o) for o in Objs]
    ForceSquaredPerEnsemble = [GetForceSqFunc(o) for o in Objs]
    # Get the histograms by time
    ForcePerTime = FlatFunc(ForcePerEnsemble,NumTimeBins)
    ForceSqPerTime = FlatFunc(ForceSquaredPerEnsemble,NumTimeBins)
    BoltzmannWeightedForce = GetBoltzmannWeightedAverage(BoltzHistogramByTime,
                                                         ForcePerTime,
                                                         Partition)
    BoltzmannWeightedForceSq = GetBoltzmannWeightedAverage(BoltzHistogramByTime,
                                                           ForceSqPerTime,
                                                           Partition)
    """
    for the second derivative, we really just want the
    variance at each z,  see equation 12 of ibid
    """
    VarianceForceBoltzWeighted = BoltzmannWeightedForceSq-\
                                 (BoltzmannWeightedForce**2)
    # now get the free energy from paragraph before eq18, ibid.
    # This is essentially the ensemble-averaged 'partition function' at each z
    Beta = np.mean([o.Beta for o in Objs])
    SpringConst = np.mean([o.SpringConstant for o in Objs])
    Velocities = np.mean([o.Velocity for o in Objs])
    ExtBins = np.mean([o.Extension[0] for o in Objs]) + Velocities * TimeBins
    k = SpringConst
    FreeEnergy_A = (-1/Beta)*np.log(Partition)
    # write down the terms involving the first and second derivative of A
    dA_dz = BoltzmannWeightedForce.copy()
    # for the second derivative, just use 1-A''/k
    SecondDerivTerm = Beta *VarianceForceBoltzWeighted/k
    # perform the IWT, ibid equation 10
    FreeEnergyAtZeroForce = FreeEnergy_A - (dA_dz)**2/(2*k) + \
                            (1/(2*Beta)) * np.log(SecondDerivTerm)
    FreeEnergyAtZeroForce -= FreeEnergyAtZeroForce[0]
    # for plotting, only look at finite.
    GoodIdx = np.where(np.isfinite(FreeEnergyAtZeroForce))[0]
    # write down q, using ibid, 10, argument to G0
    # XXX should fix...
    q = ExtBins-dA_dz/k
    # shift the free energy to F_1/2
    # approximately 20pN, see plots
    F0 = 19.5e-12
    q = ExtBins-dA_dz/k
    FreeEnergyAtF0_kbT = ((FreeEnergyAtZeroForce-ExtBins*F0)*Beta)[GoodIdx]
    FreeEnergyAtF0_kbT -= np.min(FreeEnergyAtF0_kbT)
    n=3
    fig = plt.figure(figsize=(5,7))
    # plot energies in units of 1/Beta (kT), force in pN, dist in nm
    plt.subplot(n,1,1)
    for o in Objs:
        plt.plot(o.Extension*1e9,o.Force*1e12)
    # Plot the free energy versus txtension as well
    plt.ylabel("Force [pN]")
    FreeEnergyExt = (q * 1e9)[GoodIdx]
    plt.subplot(n,1,2)
    plt.plot(ExtBins * 1e9,FreeEnergyAtZeroForce*Beta)
    plt.ylabel("Free Energy at Zero Force (kT)")
    plt.xlabel("Extension")
    plt.ylim([-2,max(FreeEnergyAtZeroForce*Beta)])
    plt.subplot(n,1,3)
    # just get the region we care about
    ExtIdx = np.where( (FreeEnergyExt < 925) & (FreeEnergyExt > 900) )
    Ext = FreeEnergyExt[ExtIdx]
    Ext -= min(Ext)
    Ext -= 7.5
    FreeEnergy = FreeEnergyAtF0_kbT[ExtIdx]
    plt.plot(Ext,FreeEnergy)
    plt.ylabel("Free Energy at F-1/2 (kT)")
    plt.xlabel("Distance around Barrier (nm)")
    plt.tight_layout()
    plt.ylim([-0.5,10])
    plt.show()

    return FreeEnergyAtZeroForce

    


# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.integrate import cumtrapz
import itertools
from collections import defaultdict


class EnergyLandscape:
    def __init__(self,EnergyLandscape,Extensions,ExtensionBins,Beta):
        # sort the energy landscape by the exensions
        SortIdx = np.argsort(Extensions)
        self.EnergyLandscape = EnergyLandscape[SortIdx]
        self.Extensions = Extensions[SortIdx]
        self.ExtensionBins = ExtensionBins
        self.Beta = Beta

class FEC_Pulling_Object:
    def __init__(self,Time,Extension,Force,SpringConstant=0.4e-3,
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
        self.SpringConstant=SpringConstant
        self.Velocity= Velocity
        self.Beta=Beta
        
        self.Work=None
        self.WorkDigitized=None
        self.ZFunc = self.ZFuncSimple if ZFunc is None else ZFunc
    def ZFuncSimple(self):
        return self.Extension[0] + (self.Velocity * self.Time)
    def GetWorkArgs(self,ZFunc):
        """
        Gets the in-order arguments for the work functions
        Args:
            ZFunc: see GetDigitizedBoltzmann
        """
        return self.SpringConstant,self.Velocity,self.Time,self.Extension
    def CalculateForceCummulativeWork(self):
        """
        Gets the position-averaged work, see methods section of 
        paper cited in GetDigitizedBoltzmann
         
        Args:
            ZFunc: See GetDigitizedBoltzmann
        Returns:
            The cummulative integral of work, as defined in ibid, before eq18
        """
        # compute the forst from the bias 
        Force = self.Force
        Z = self.ZFunc()
        ToRet = cumtrapz(x=Z,y=Force,initial=0)
        return ToRet
    def SetWork(self,Work):
        self.Work = Work
    def _GetDigitizedGen(self,Bins,ToDigitize):
        """
        Generalized method to get 'digitized' results

        Args:
            Bins: see GetDigitizedBoltzmann
            ToDigitize: Array to dizitize, e.g. self.Forces
        Returns:
            See GetDigitizedBoltzmann, except digitized contents of 
            'GetDigitizedBoltzmann'
        """
        NumTimes = Bins.size
        IdxAdd = np.digitize(self.Extension,bins=Bins)
        DigitzedMatrix = [[] for _ in range(NumTimes)]
        for i,idx in enumerate(IdxAdd):
            DigitzedMatrix[idx-1].append(ToDigitize[i])
        return DigitzedMatrix
    def GetDigitizedBoltzmann(self,Bins):
        """
        Gets the digitized boltzmann factor np.exp(-beta*W)
        (averaged in a bin), given the Bins. For nomenclature, see:
        
        Hummer, G. & Szabo, A. 
        Free energy profiles from single-molecule pulling experiments. 
        PNAS 107, 21441-21446 (2010).

        Especially equaitons 11-12 and 18-19 and relevant discussion


        Args:
            Bins: the bins to use
        Returns: 
            The digitized boltman matrix exp(-Beta*W), 
            where W[i] is a *list* of Work values associated with BinTime[i]
        """
        ToDigitize = np.exp(-self.Beta*self.Work)
        return self._GetDigitizedGen(Bins,ToDigitize)
    def GetDigitizedForce(self,Bins):
        """
        Gets the digitized force within the bins. See materials cited in 
        GetDigitizedBoltzmann

        Args:
            Bins: see ibid
        Returns:
            see GetDigitizedBoltzmann, except Force is the content
        """
        return self._GetDigitizedGen(Bins,self.Force)


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
                                Values,Partition,Delta):
    """
    Given a matrix BoltzmannFactors[i][j] where i refers to
    the bin, and j refers to the FEC label,
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
    ToRet = np.zeros(BoltzmannWeightedAverage.size)
    GoodIdx = np.where(np.isfinite(Partition))[0]
    PartitionGood = Partition[GoodIdx]
    IdxWhereNonZero = GoodIdx[np.where(PartitionGood > 0)]
    NonZeroAverages = BoltzmannWeightedAverage[IdxWhereNonZero]/\
                      Partition[IdxWhereNonZero]
    ToRet[IdxWhereNonZero] = NonZeroAverages
    # avoid divide by zero error...
    return ToRet

def DistanceToRoot(DeltaA,Beta,ForwardWork,ReverseWork):
    """
    Gives the distance to the root in equation 18 (see NumericallyGetDeltaA)

    Args:
        Beta,DeltaA: See ibid
        FowardWork,ReverseWork: list of the works as defined in ibid, same
        Units as DeltaA
    """
    nf = len(ForwardWork)
    nr = len(ReverseWork)
    # get the forward and reverse 'factor': difference should be zero
    Forward = np.mean(1/(nr + nf * np.exp(Beta * (ForwardWork-DeltaA))))
    Reverse = np.mean(1/(nf + nr * np.exp(Beta * (ReverseWork+DeltaA))))
    # we really only case about the abolute value of the expression, since
    # we want the two sides to be equal...
    return np.abs(Forward-Reverse)

def NumericallyGetDeltaA(Forward,Reverse,disp=3,**kwargs):
    """
    Numerically solves for DeltaA, as in equation 18 of 

    Hummer, G. & Szabo, A. 
    Free energy profiles from single-molecule pulling experiments. 
    PNAS 107, 21441-21446 (2010).

    Note that we use a root finder to find the difference in units of kT,
    then convert back (avoiding large floating point problems associated with
    1e-21)

    Args:
        Forward: List of forward paths
        Reverse: List of reverse paths
    Returns:
        Free energy different, in joules
    """
    # XXX should fix/ not hard code
    beta = 1/(4.1e-21)
    # get the work in terms of beta, should make it easier to converge
    Fwd = [f.Work*beta for f in Forward]
    Rev = [f.Work*beta for f in Reverse]
    MaxWorks = [np.max(np.abs(Fwd)),
                np.max(np.abs(Rev))]
    MinWorks = [np.min(Fwd),
                np.min(Rev)]
    Max = max(MaxWorks)
    Min = min(MinWorks)
    # only look between +/- the max. Note that range is guarenteed positive
    Range = Max-Min
    FMinArgs = dict(x1=-Range,x2=Range,full_output=True,disp=disp,**kwargs)
    # note we set beta to one, since it is easier to solve in units of kT
    ToMin = lambda A: DistanceToRoot(A,Beta=1,ForwardWork=Fwd,ReverseWork=Rev)
    xopt,fval,ierr,nfunc = fminbound(ToMin,**FMinArgs)
    return xopt/beta

def FoldUnfoldAverage(ValueFunction,Unfolding,Refolding,DeltaA):
    """
    Given unfolding and refolding data, gets the average of the property given
    by Valuefunction. Note that we average over the first axis (assumed to be
    the ensemble of measurements) 

    Args:
        ValueFunction: takes in an item from Unfolding or folding, returns the 
        value we want (e.g. force, forcesq, probably as a function of Z)

        Unfolding:  list of FEC_Pulling_Object, representing unfolding curves
        Refolding: list of FEC_Pulling_Object, representing folding curves
        DeltaA: from NumericallyGetDeltaA
    Returns:
        Average at each value of z
    """
    nf = len(Folding)
    nu = len(Unfolding)
    value_folded =[ValueFunction(o) for o in Folding]
    value_unfolded =[ValueFunction(o) for o in Unfolding]
    """
    get the boltzmann factors, following  equaition 18 of

    Hummer, G. & Szabo, A. 
    Free energy profiles from single-molecule pulling experiments. 
    PNAS 107, 21441-21446 (2010).

    and equation 6 of 
    
    Minh, D. D. L. & Adib, A. B. 
    Optimized Free Energies from Bidirectional Single-Molecule 
    Force Spectroscopy. Phys. Rev. Lett. 100, 180602 (2008).

    Noting the average is boltzmann weighted (eg: 19 reduces to 1 when 
    nr=0, meaning no unfolding is available)
    """
    boltz_fold = [np.exp(-o.Beta*(o.Work-DeltaX)) for o in Folding]
    boltz_unfold = [np.exp(-o.Beta*(o.Work+DeltaX)) for o in Unfolding]
    # get the averaged value, noting 
    value_folded = nf * value_folded       / (nf + nr * boltz_fold)
    value_unfold = nr * value_unfoldfolded / (nr + nf * boltz_unfold)
    return np.mean(value_folded,axis=0) + np.mean(value_unfold,axis=0)


def FreeEnergyAtZeroForce(UnfoldingObjs,NumBins,RefoldingObjs=None):
    """
    Wrapper to make it easier to get the weighted histograms, etcs.

    Args:
        obj: list of FEC_Pulling_Object
        NumBins: number of bins to put things into
    """
    # get the bounds associated with the times and extensions
    TimeBounds = GetTimeBounds(UnfoldingObjs)
    ExtBounds = GetExtensionBounds(UnfoldingObjs)
    # Create the time and position bins using a helper function
    BinIt= lambda x,n: np.linspace(start=x[0],
                                   stop=x[1],
                                   endpoint=False,
                                   num=n)
    # create the extension bins 
    ExtBins = BinIt(ExtBounds,NumBins)
    # Set up functions for getting the force and boltzmann factors
    BinDataTo = ExtBins
    BoltzmanFunc = lambda o : o.GetDigitizedBoltzmann(BinDataTo)
    ForceFunc = lambda o: o.GetDigitizedForce(BinDataTo)
    GetForceSqFunc = lambda x : x._GetDigitizedGen(BinDataTo,
                                                   x.Force**2)
    # get the (per-instance) boltmann factors, for weighing
    BoltzByFEC = [BoltzmanFunc(o) for o in UnfoldingObjs]
    # get the (flattend) boltmann factors
    FlatFunc = lambda objs,bins : [ [item
                                     for x in objs
                                     for item in x[i]]
                                    for i in range(bins)]
    NBins = len(BinDataTo)
    BoltzHistogramByTime = FlatFunc(BoltzByFEC,NBins)
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
    ForcePerEnsemble = [ForceFunc(o) for o in UnfoldingObjs]
    ForceSquaredPerEnsemble = [GetForceSqFunc(o) for o in UnfoldingObjs]
    # Get the histograms by time
    ForcePerTime = FlatFunc(ForcePerEnsemble,NBins)
    ForceSqPerTime = FlatFunc(ForceSquaredPerEnsemble,NBins)
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
    GoodIndex = np.where( (VarianceForceBoltzWeighted > 0) &
                          (np.isfinite(VarianceForceBoltzWeighted)))
    # now get the free energy from paragraph before eq18, ibid.
    # This is essentially the ensemble-averaged 'partition function' at each z
    Beta = np.mean([o.Beta for o in UnfoldingObjs])
    SpringConst = np.mean([o.SpringConstant for o in UnfoldingObjs])
    Velocities = np.mean([o.Velocity for o in UnfoldingObjs])
    k = SpringConst
    FreeEnergy_A = (-1/Beta)*np.log(Partition[GoodIndex])
    # write down the terms involving the first and second derivative of A
    dA_dz = BoltzmannWeightedForce[GoodIndex].copy()
    # for the second derivative, just use 1-A''/k
    SecondDerivTerm = Beta *VarianceForceBoltzWeighted[GoodIndex]/k
    # perform the IWT, ibid equation 10
    FreeEnergyAtZeroForce = FreeEnergy_A - (dA_dz)**2/(2*k) + \
                            (1/(2*Beta)) * np.log(SecondDerivTerm)
    # bottom of well is prettu much arbitrary
    okIdx =np.isfinite(FreeEnergyAtZeroForce)
    MinVal = np.nanmin(FreeEnergyAtZeroForce[okIdx])
    FreeEnergyAtZeroForce -= MinVal
    # write down q, using ibid, 10, argument to G0
    q = ExtBins[GoodIndex]-dA_dz/k
    return EnergyLandscape(FreeEnergyAtZeroForce,q,ExtBins,Beta)



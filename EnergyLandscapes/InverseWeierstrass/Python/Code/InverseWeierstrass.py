# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.integrate import cumtrapz
import itertools
from collections import defaultdict
from scipy.optimize import fminbound
from scipy.sparse import csr_matrix


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
    def GetDigitizedOnes(self,Bins):
        """
        see GetDigitizedBoltzmann, except returns the 
        """
        return self._GetDigitizedGen(Bins,np.ones(self.Work.size))
    def GetDigitizedWork(self,Bins):
        return self._GetDigitizedGen(Bins,self.Work)
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
        disp: arugment to root finder: default shows all the steps
    Returns:
        Free energy different, in joules
    """
    if (len(Reverse) == 0):
        return 0
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

def FuncPerEnsemble(func1,x,func2):
    return func2([func1(x) for example in x if len(x) >0])

def Mean(x):
    return np.mean(x)

def Exp(x):
    return np.exp(x)

def ReverseAverage(nf,nr,vr,Wr,DeltaA,Beta):
    if (nr == 0):
        return 0
    Wrn = Wr[-1]
    Dist = (vr*nr*Exp(-Beta*(Wr + DeltaA)))/(nr + nf*Exp(-Beta*(Wfn + DeltaA)))
    return Mean(Dist)

def ForwardAverage(nf,nr,vf,Wf,DeltaA,Beta):
    """
    Returns the 'Forward average' for a single Z

    Args:
        see BidirectionalAverage, except work and values are 1-D lists
        (for a single z)
    Returns:
        Bi-directional average of the values (see ibid)
    """
    Wfn = Wf[-1]
    Dist = (vf*nf*Exp(-Beta*Wf))/(nf + nr*Exp(-Beta*(Wfn - DeltaA)))
    # average over the results of this *single* FEC in a single bin
    return Mean(Dist)


def BidirectionalAverage(nf,nr,vf,vr,Wf,Wr,DeltaA,Beta):
    """
    Return the bi-directional average given forward and reverse values
    Assumes that for all thevalues and work, the are arrays where 
    element [i,j] refers to bin i, ensemble number j. Note that one
    FEC might have multiple measurements in a single bin (ie: FEC 4 could 
    have elements 4,5,6 in bin 0)
    
    For Context, See: Equation 19 of 
    Hummer, G. & Szabo, A. 
    Free energy profiles from single-molecule pulling experiments. 
    PNAS 107, 21441-21446 (2010).

    Args:
         vf/vr: the forward and reverse values we want to average over
         Wf/Wr: the forward and reverse work, as defined by ibid
         DeltaA:Free energy difference between unfolded and folded start states
         (See Ibid)

         Beta: 1/(k*T), The Boltzmann constant
    Returns:
         Boltzmann-weights value across forward and reverse states
    """
    # we average along columns (ensemble numbers), so that we wind
    # up with one estimate per Z bin
    NumBins = len(vf)
    Fwds = np.zeros(NumBins)
    Revs = np.zeros(NumBins)
    Pre = lambda x: np.array(x)
    for i,(vf_z,Wf_z) in enumerate(zip(vf,Wf)):
        Vals = [ForwardAverage(nf,nr,Pre(v),Pre(W),DeltaA,Beta)
                for v,W in zip(vf_z,Wf_z) if len(v) >0]
        Fwds[i] = np.mean(Vals)
    # only add the reverse if we have them
    if (nr > 0):
        for i,(vr_z,Wr_z) in enumerate(zip(vr,Wr)):
            Vals = [ForwardAverage(nf,nr,Pre(v),Pre(W),DeltaA,Beta)
                    for v,W in zip(vr_z,Wr_z) if len(v) >0]
            Revs[i] = np.mean(Vals) 
    ToRet = Fwds + Revs
    print(Revs)
    return ToRet


def ListsToHistogramPerBin(L,Bins):
    """
    Converts from FEC ensemble to binned extensions 
    
    Args:
        L: 3-D list, where L[i,j,k] is FEC i,bin j, value k
    Returns:
        T[l,m], where l is bin l and m runs over all the FEC and values in 
        that bin
    """
    NumCurves = len(L)
    if (NumCurves == 0):
        return [[] for b in range(Bins)]
    NumBins = len(L[0])
    if (NumBins == 0):
        return [[] for b in range(Bins)]
    TransposedLists = map(list,zip(*L))
    # 'flatten' all the values so we have a list, each element is a list for
    # a single bin
    # POST: TransposedLists[i,j,k] is bin i,FEC j, value k
    return TransposedLists

def FoldUnfoldAverage(ValueFunction,Unfolding,Refolding,DeltaA,
                      WorkFunction,Beta):
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
        NormalizeByPartition: if true, normalize by the parition function
    Returns:
        Average at each value of z
    """
    """
    get the weighted values, following equation 18 of

    Hummer, G. & Szabo, A. 
    Free energy profiles from single-molecule pulling experiments. 
    PNAS 107, 21441-21446 (2010).
    """ 
    value_fwd =[ValueFunction(o) for o in Unfolding]
    value_rev =[ValueFunction(o) for o in Refolding]
    work_fwd = [WorkFunction(o) for o in Unfolding]
    work_rev = [WorkFunction(o) for o in Refolding]
    NumFec = len(value_fwd)
    assert NumFec>0 , "No data given"
    NumBins = len(value_fwd[0])
    assert NumBins>0 , "No bins given"
    # to preprocess, we get all of the elements (from FEC) in a single bin
    Preprocess = lambda x : ListsToHistogramPerBin(x,NumBins)
    return BidirectionalAverage(nf=len(Unfolding),
                                nr=len(Refolding),
                                vf=Preprocess(value_fwd),
                                vr=Preprocess(value_rev),
                                Wf=Preprocess(work_fwd),
                                Wr=Preprocess(work_rev),
                                DeltaA=DeltaA,
                                Beta=Beta)


def FreeEnergyAtZeroForce(UnfoldingObjs,NumBins,RefoldingObjs=[]):
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
    DeltaA = NumericallyGetDeltaA(UnfoldingObjs,RefoldingObjs)
    # XXX make sure they match?
    Beta = np.mean([o.Beta for o in UnfoldingObjs])
    # create the extension bins 
    ExtBins = BinIt(ExtBounds,NumBins)
    # Set up functions for getting the force and boltzmann factors
    BinDataTo = ExtBins
    # XXX debug
    digit = UnfoldingObjs[0].GetDigitizedBoltzmann(BinDataTo)
    ParitionFunc = lambda o : o.GetDigitizedOnes(BinDataTo)
    ForceFunc = lambda o: o.GetDigitizedForce(BinDataTo)
    GetForceSqFunc = lambda x : x._GetDigitizedGen(BinDataTo,x.Force**2)
    WorkFunction = lambda o: o.GetDigitizedWork(BinDataTo)
    # get the Parition function
    avg_kwargs = dict(Unfolding=UnfoldingObjs,
                      Refolding=RefoldingObjs,
                      DeltaA=DeltaA,
                      Beta=Beta,
                      WorkFunction=WorkFunction)
    print("Parition")
    Partition = FoldUnfoldAverage(ValueFunction=ParitionFunc,
                                  **avg_kwargs)
    print("Force")
    WeightedForce = FoldUnfoldAverage(ValueFunction=ForceFunc,**avg_kwargs)
    print("ForceSq")
    WeightedForceSq = FoldUnfoldAverage(ValueFunction=GetForceSqFunc,
                                        **avg_kwargs)
    """
    for the second derivative, we really just want the
    variance at each z,  see equation 12 of ibid
    """
    VarianceForceBoltzWeighted = WeightedForce-(WeightedForceSq**2)
    GoodIndex = np.where( (VarianceForceBoltzWeighted > 0) &
                          (np.isfinite(VarianceForceBoltzWeighted)))
    # now get the free energy from paragraph before eq18, ibid.
    # This is essentially the ensemble-averaged 'partition function' at each z
    SpringConst = np.mean([o.SpringConstant for o in UnfoldingObjs])
    Velocities = np.mean([o.Velocity for o in UnfoldingObjs])
    k = SpringConst
    FreeEnergy_A = (-1/Beta)*np.log(Partition[GoodIndex])
    # write down the terms involving the first and second derivative of A
    dA_dz = WeightedForce[GoodIndex].copy()
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



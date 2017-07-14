# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.integrate import cumtrapz
import itertools
from collections import defaultdict
from scipy.optimize import fminbound,newton
from scipy import sparse

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
        
            Velocity: in m/s, default from data from ibid.
            Beta: 1/(kbT), defaults to room temperature (4.1 pN . nm)
        """
        self.Time = Time
        self.Extension = Extension 
        self.Force = Force
        self.SpringConstant=SpringConstant
        self.Velocity= Velocity
        self.Beta=Beta
        self.Offset = self.Extension[0]
        self.SetWork(self.CalculateForceCummulativeWork())
        self.WorkDigitized=None
    @property
    def ZFunc(self):
        return self.ZFuncSimple
    @property
    def Separation(self):
        return self.Extension
    def ZFuncSimple(self):
        return self.Offset + (self.Velocity * self.Time)
    def SetVelocityAndOffset(self,Offset,Velocity):
        """
        Sets the velocity and offset used in ZFuncSimple

        Args:
            Offset:  offset in distance (same units of extension)
            Velocity: slope (essentially, effective approach/retract rate).
        Returns:
            Nothing
        """
        self.Velocity = Velocity
        self.Offset = Offset
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
        bin_idx_for_each_point = np.digitize(self.Extension,bins=Bins)
        # force bin idx to be between 0 and N_bins-1
        bin_idx_for_each_point = np.minimum(NumTimes-1,bin_idx_for_each_point)
        n_points = ToDigitize.size
        # get a digitized matrix where
        # full[ [Bin Idx, Point Idx] ] = Value of the point 
        idx_arr = np.arange(n_points)
        full = sparse.csr_matrix((ToDigitize,(bin_idx_for_each_point,idx_arr)),
                                  shape=(NumTimes,n_points))
        # concatenate the columns together; data_by_rows[i] is "the value
        # of every point from ToDigitize in the Bins[i]"
        data_by_rows = [full.data[full.indptr[i]:full.indptr[i+1]]
                        for i in range(NumTimes)]
        return data_by_rows
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

def Exp(x):
    return np.exp(x)

def ForwardWeighted(nf,nr,vf,Wf,Wfn,DeltaA,Beta):
    """
    Returns the weighted value for the forward part of the bi-directionary free
    energy landscape
    
    Args: see EnsembleAverage
    """
    return (vf*nf*Exp(-Beta*Wf))/(nf + nr*Exp(-Beta*(Wfn - DeltaA)))

def ReverseWeighted(nf,nr,vr,Wr,Wrn,DeltaA,Beta):
    """
    Returns the weighted value for a reverse step

    Args: see EnsembleAverage
    """
    return (vr*nr*Exp(-Beta*(Wr + DeltaA)))/(nr + nf*Exp(-Beta*(Wrn + DeltaA)))

def EnsembleAverage(v_fwd,v_rev,w_fwd,w_rev,w_fwd_n,w_rev_n,Beta,DeltaA,nf,nr):
    """
    Ensemble averages a forward and (possibly empty) reverse section
    
    Args:
        v_fwd: values for the forward
        w_fwd: work for the forward process, at a single Z value
        w_fwn_n: work for the 'final' state (see GetBoltzmannWeightedAverage)
        <x>_rev_<y>: same as forward, but for thr reverse process
        Beta: 1/(k*T), where T is temprature and k is boltzmann constant
        DeltaA: output of NumericallyGetDeltaA, free eneegyr difference
        nf/nr: number of forward/reverse trials
    Returns:
        ensemble-averaged values. 
    """
    # get the weights for the fwd
    pre = lambda x: np.array(x)
    common_args = dict(Beta=Beta,DeltaA=DeltaA,nf=nf,nr=nr)
    Fwd = [ForwardWeighted(vf=pre(v),Wf=pre(W),Wfn=pre(Wfn),**common_args)
           for v,W,Wfn in zip(v_fwd,w_fwd,w_fwd_n)]
    # get the weights for the reverse, if we have any
    if (nr > 0):
        Rev = [ReverseWeighted(vr=pre(v),Wr=pre(W),Wrn=pre(Wrn),**common_args)
               for v,W,Wrn in zip(v_rev,w_rev,w_rev_n)]
    else:
        Rev = [ [0 for i in range(len(v))] for v in v_fwd]
    # Concatenate all the forward and reverse arrays
    fwd_concat = np.concatenate(Fwd)
    rev_concat = np.concatenate(Rev)
    # now we have all the values from the entire ensemble; we just average
    # across forard and reverse (separately!), then add (fine if adding zeros)
    MeanFwd = np.mean(fwd_concat)
    MeanRev = np.mean(rev_concat)
    Total = MeanFwd + MeanRev
    return Total

def GetBoltzmannWeightedAverage(Forward,Reverse,ValueFunction,WorkFunction,
                                DeltaA,PartitionDivision):
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
    # function which converts an array x like x[i,j,k]
    # is FEC i, bin j, value k, to an array like y[l,m], bin l and ensemble m
    FlatFunc = lambda objs,bins : [ [item
                                     for x in objs
                                     for item in x[i]]
                                    for i in range(bins)]
    # function to convert ibid into array like [i,j,k] where
    # i is bin i, j is FEC, k is value
    ByBinFunc = lambda objs,bins : [ [[item for item in x[i]]
                                      for x in objs]
                                      for i in range(bins)]
    nf = len(Forward)
    nr = len(Reverse)
    assert nf > 0 , "No Forward Curves"
    v_fwd = [ValueFunction(f) for f in Forward]
    NumBins = len(v_fwd[0])
    assert len(v_fwd) > 0 ,"No Bins"
    beta = np.mean([f.Beta for f in Forward])
    ValueByBins = lambda objs : ByBinFunc([ValueFunction(f) for f in objs],
                                          NumBins)
    WorkByBins = lambda objs : ByBinFunc([WorkFunction(f) for f in objs],
                                         NumBins)
    value_fwd,value_rev = ValueByBins(Forward),ValueByBins(Reverse)
    work_fwd,work_rev = WorkByBins(Forward),WorkByBins(Reverse)
    LastWork = lambda o : o.Work[-1]
    wfn = np.array([o.Work[-1] for o in Forward])
    wrn = np.array([o.Work[-1] for o in Reverse])
    ToRet = []
    for vf,vr,wf,wr in zip(value_fwd,value_rev,work_fwd,work_rev):
        # XXX need to actually get forward and reverse parts
        val = EnsembleAverage(v_fwd=vf,
                              v_rev=vr,
                              w_fwd=wf,
                              w_rev=wr,
                              w_fwd_n=wfn,
                              w_rev_n=wrn,
                              Beta=beta,
                              DeltaA=DeltaA,
                              nf=nf,
                              nr=nr)
        ToRet.append(val)
    return np.array(ToRet)/np.array(PartitionDivision)

def DistanceToRoot(DeltaA,Beta,ForwardWork,ReverseWork):
    """
    Gives the distance to the root in equation 18 (see NumericallyGetDeltaA)

    Unit Tested by : MainTesting.TestForwardBackward

    Args:
        Beta,DeltaA: See ibid
        FowardWork,ReverseWork: list of the works as defined in ibid, same
        Units as DeltaA
    Returns:
        difference between forward and reverse
    """
    nf = len(ForwardWork)
    nr = len(ReverseWork)
    # get the forward and reverse 'factor': difference should be zero
    Forward = 1/(nr + nf * np.exp(Beta * (ForwardWork-DeltaA)))
    Reverse = 1/(nf + nr * np.exp(Beta * (ReverseWork+DeltaA)))
    # we really only case about the abolute value of the expression, since
    # we want the two sides to be equal...
    return np.mean(Forward)-np.mean(Reverse)

def NumericallyGetDeltaA(Forward,Reverse,maxiter=200,**kwargs):
    """
    Numerically solves for DeltaA, as in equation 18 of 

    Hummer, G. & Szabo, A. 
    Free energy profiles from single-molecule pulling experiments. 
    PNAS 107, 21441-21446 (2010).

    Note that we use a root finder to find the difference in units of kT,
    then convert back (avoiding large floating point problems associated with
    1e-21). 

    Unit Tested by : MainTesting.TestForwardBackward

    Args:
        Forward: List of forward paths
        Reverse: List of reverse paths
        disp: arugment to root finder: default shows all the steps
        kwargs: passed to newton
    Returns:
        Free energy different, in joules
    """
    if len(Reverse) == 0:
        return 0
    # XXX should fix/ not hard code
    beta = 1/(4.1e-21)
    # get the work in terms of beta, should make it easier to converge
    Fwd = [f.Work[-1]*beta for f in Forward]
    Rev = [f.Work[-1]*beta for f in Reverse]
    MaxWorks = [np.max(np.abs(Fwd)),
                np.max(np.abs(Rev))]
    MinWorks = [np.min(Fwd),
                np.min(Rev)]
    Max = max(MaxWorks)
    Min = min(MinWorks)
    # only look between +/- the max. Note that range is guarenteed positive
    Range = Max-Min
    FMinArgs = dict(x0=(Max-Min)/2,maxiter=maxiter,**kwargs)
    # note we set beta to one, since it is easier to solve in units of kT
    ToMin = lambda A: DistanceToRoot(A,Beta=1,ForwardWork=Fwd,ReverseWork=Rev)
    xopt = newton(ToMin,**FMinArgs)
    return xopt/beta
    
def FreeEnergyAtZeroForce(UnfoldingObjs,NumBins,RefoldingObjs=[]):
    """
    Wrapper to make it easier to get the weighted histograms, etcs.

    Args:
        obj: list of FEC_Pulling_Object
        NumBins: number of bins to put things into
    Returns:
        Energy Landscape Object
    """
    # get the bounds associated with the times and extensions
    ExtBounds = GetExtensionBounds(UnfoldingObjs)
    # Create the time and position bins using a helper function
    BinIt= lambda x,n: np.linspace(start=x[0],
                                   stop=x[1],
                                   endpoint=False,
                                   num=n)
    DeltaA = NumericallyGetDeltaA(UnfoldingObjs,RefoldingObjs)
    # create the extension bins 
    ExtBins = BinIt(ExtBounds,NumBins)
    # Set up functions for getting the force and boltzmann factors
    BinDataTo = ExtBins
    BoltzmanFunc = lambda o : o.GetDigitizedBoltzmann(BinDataTo)
    ForceFunc = lambda o: o.GetDigitizedForce(BinDataTo)
    ForceSqFunc = lambda o : o._GetDigitizedGen(BinDataTo,o.Force**2)
    OnesFunc = lambda o: o.GetDigitizedOnes(BinDataTo)
    WorkFunc = lambda o : o._GetDigitizedGen(BinDataTo,o.Work)
    # get the (per-instance) boltmann factors, for weighing
    BoltzByFEC = [BoltzmanFunc(o) for o in UnfoldingObjs]
    NBins = len(BinDataTo)
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
    weight_kwargs = dict(Forward=UnfoldingObjs,
                         Reverse=RefoldingObjs,
                         WorkFunction=WorkFunc,DeltaA=DeltaA)
    Partition = GetBoltzmannWeightedAverage(ValueFunction=OnesFunc,
                                            PartitionDivision=1,
                                            **weight_kwargs)
    BoltzmannWeightedForce = \
        GetBoltzmannWeightedAverage(ValueFunction=ForceFunc,
                                    PartitionDivision=Partition,
                                    **weight_kwargs)
    BoltzmannWeightedForceSq = \
        GetBoltzmannWeightedAverage(ValueFunction=ForceSqFunc,
                                    PartitionDivision=Partition,
                                    **weight_kwargs)
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
    okIdx = np.where(np.isfinite(FreeEnergyAtZeroForce))[0]
    assert (okIdx.size > 0) , \
        "IWT failed; transform resulted in an infinite energy landscape"
    # POST: at least one point
    MinVal = np.nanmin(FreeEnergyAtZeroForce[okIdx])
    FreeEnergyAtZeroForce -= MinVal
    # write down q, using ibid, 10, argument to G0
    q = ExtBins[GoodIndex]-dA_dz/k
    return EnergyLandscape(FreeEnergyAtZeroForce,q,ExtBins,Beta)



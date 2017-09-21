# force floating point division. Can still use integer with //
from __future__ import division
# other good compatibility recquirements for python3
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys


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



def _digitized_combined(histograms):
    """
    Given histogrm[i,j,k] is point k in bin j of FEC i, returns
    a list digitized[j,p], where j is bin j, and p is an index into the data...
    """
    n_bins = len(histograms)
    # note: zip(*l) transposes, see:
    # https://stackoverflow.com/questions/6473679/transpose-list-of-lists
    # -->  histograms[i,j]  is FEC i, bin j, but...
    # zip(*histograms)[i,j] is FEc j, bin i (swapped indices)
    return [ [item for sublist in single_bin for item in sublist]
             for single_bin in zip(*histograms)]

def _digitized_f(objs,f):
    """
    returns the digitization function f applied to each member of obj
    
    Args:
       objs: list of objects
       f: function, takes in object, returns the histogram for that object
       bins: bins for the function
    Returns:
       output of f(o,bins) formatted as _digitized_combined
    """
    return _digitized_combined([f(o) for o in objs])


def _boltzmann_weighted(is_reverse,**kw):
    """
    gives the boltzmann-weighted values for the forward or reverse ensemble

    Args:
        is_reverse: boolean, if true, then this return the reverse weight 
        work: for the boltzmann factor, length N 
        values_f: values to weight, length N 
    """
    full_dictionary = dict(**kw)
    f = ReverseWeighted if (is_reverse) else ForwardWeighted
    return np.mean(f(**full_dictionary),axis=0)

def _single_direction_weighted(objs,f_work,f_value,beta,**kw):
    # get the values and work arrays; index [i,j] is all points from FEC <i>
    # in bin <j>
    value_histograms = [f_value(f) for f in objs]
    work_histograms = [f_work(f) for f in objs]
    # average all the work in the last bin for all objects; 
    # w_f[i] is the last work for FEC i 
    w_f = [histogram[-1][-1]
           if len(histogram[-1]) > 0 else 0 \
           for histogram in work_histograms] 
    # get the digitized values/work. Index [i,j] is bin i, point j
    digitized_values = _digitized_combined(value_histograms)
    digitized_work = _digitized_combined(work_histograms)
    n_bins = len(digitized_values)
    to_ret = np.zeros(n_bins)
    for i in range(n_bins):
        # get all the values and work present here. 
        values_tmp = np.array(digitized_values[i])
        # XXX should probably check this... bins shouldn't be empty
        if len(values_tmp) == 0:
            continue
        work_tmp = np.array(digitized_work[i])
        w_f_tmp = np.concatenate([w * np.ones(len(v[i])) 
                                  for w,v in zip(w_f,value_histograms)
                                  if len(v[i]) > 0])
        kw_tmp = dict(W=work_tmp,
                      Wn=w_f_tmp,
                      Beta=beta,
                      v=values_tmp,
                      **kw)
        all_weighted = _boltzmann_weighted(**kw_tmp)
        to_ret[i] = all_weighted
    return np.array(to_ret)

def _refolding_weighted(forward,reverse,f_work,f_value,**kw):
    beta = np.mean([f.Beta for f in (forward + reverse)])
    # XXX assert beta all the same?
    kw = dict(beta=beta,f_work=f_work,f_value=f_value,**kw)
    ret_fwd = _single_direction_weighted(objs=forward,is_reverse=False,**kw)
    n_bins = ret_fwd.size
    # only use the reverse if we have it 
    if (len(reverse) > 0):
        ret_rev =  _single_direction_weighted(objs=reverse,is_reverse=True,**kw)
    else:
        # if we dont have it, we just add zeros to the result, which doesn't
        # affect anything... 
        ret_rev = np.zeros(n_bins)
    return ret_fwd + ret_rev 

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
         by Hummer and Szabo, 2010, PNAS, after equation 12. Array is np.nan
         where the 'PartitionDivision' is zero
    """
    # function to convert ibid into array like [i,j,k] where
    # i is bin i, j is FEC, k is value
    nf = len(Forward)
    nr = len(Reverse)
    assert nf > 0 , "No Forward Curves"
    v_fwd = ValueFunction(Forward[0])
    NumBins = len(v_fwd)
    assert len(v_fwd) > 0 ,"No Bins"
    # POST: at least have something to work with 
    assert (PartitionDivision > 0).any() , "Partition function was zero."
    weighted = _refolding_weighted(forward=Forward,reverse=Reverse,
                                   f_work=WorkFunction,f_value=ValueFunction,
                                   DeltaA=DeltaA,nf=nf,nr=nr)
    to_ret = np.zeros(weighted.size)
    good_idx = np.where(PartitionDivision > 0)[0]
    good_partition = np.array(PartitionDivision)[good_idx]
    to_ret[good_idx] = np.array(weighted[good_idx])/good_partition
    return to_ret

    
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
    for re in RefoldingObjs:
        v = re.Velocity
        cond = v < 0 
        assert cond , \
            "Refolding data should have negative velocity, got {:.3g}".format(v)
    # POST: velocity data looks good.
    DeltaA = NumericallyGetDeltaA(UnfoldingObjs,RefoldingObjs)
    # create the extension bins 
    ExtBins = BinIt(ExtBounds,NumBins)
    BinDataTo = ExtBins
    # Set up functions for getting the force and boltzmann factors
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
                                            PartitionDivision=np.ones(NumBins),
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
    FiniteIdx = np.where(np.isfinite(VarianceForceBoltzWeighted))[0]
    GoodIndex = \
        FiniteIdx[np.where( (VarianceForceBoltzWeighted[FiniteIdx] > 0))[0]]
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
    # perform the IWT, ibid equation 10
    first_deriv_term = -(dA_dz)**2/(2*k)
    second_deriv_term = (1/(2*Beta)) * \
        np.log(Beta *VarianceForceBoltzWeighted[GoodIndex]/k)
    FreeEnergyAtZeroForce = FreeEnergy_A  + first_deriv_term + second_deriv_term
    # bottom of well is prettu much arbitrary
    okIdx = np.where(np.isfinite(FreeEnergyAtZeroForce))[0]
    assert (okIdx.size > 0) , \
        "IWT failed; transform resulted in an infinite energy landscape"
    # POST: at least one point
    MinVal = np.nanmin(FreeEnergyAtZeroForce[okIdx])
    FreeEnergyAtZeroForce -= MinVal
    # write down q, using ibid, 10, argument to G0
    q = ExtBins[GoodIndex]-dA_dz/k
    return EnergyLandscape(FreeEnergyAtZeroForce,q,ExtBins,Beta,
                           free_energy_A=FreeEnergy_A,
                           first_deriv_term=first_deriv_term,
                           second_deriv_term=second_deriv_term)
    
    

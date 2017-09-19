# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.integrate import cumtrapz
import itertools
from collections import defaultdict
from scipy.optimize import fminbound,newton,brentq
from scipy import sparse


class _WorkWeighted:
    def __init__(self,objs,work_offset):
        self.objs = objs
        self.work_offset = work_offset
    def set_variables(self,partition,f_work_weighted,f_squared_work_weighted):
        self.partition = partition
        self.f = f_work_weighted
        self.f_squared = f_squared_work_weighted

class Landscape:
    def __init__(self,q,energy,kT,
                 free_energy_A,first_deriv_term,second_deriv_term):
        sort_idx = np.argsort(q)
        f_sort = lambda x: x[sort_idx].copy()
        self.q = f_sort(q)
        self.energy = f_sort(energy)
        self.A_z = f_sort(free_energy_A)
        self.first_deriv_term = f_sort(first_deriv_term)
        self.second_deriv_term = f_sort(second_deriv_term)
        self.kT = kT
    @property
    def beta(self):
        return 1/self.kT
    @property
    def G_0(self):
        return self.energy

def ZFuncSimple(obj):
    return obj.Offset + (obj.Velocity * obj.Time)        

class FEC_Pulling_Object:
    def __init__(self,Time,Extension,Force,SpringConstant=0.4e-3,
                 Velocity=20e-9,kT=4.1e-21):
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

            ZFunc: Function which takes in an FEC_Pulling_Object (ie: this obj)
            and returns a list of z values at each time. If none, defaults
            to simple increase from first extension

        
            Velocity: in m/s, default from data from ibid.
            kT: kbT, defaults to room temperature (4.1 pN . nm)
        """
        # make copies (by value) of the arrays we need
        self.kT=kT
        self.Beta=1/kT
        self.Time = Time.copy()
        self.Extension = Extension.copy()
        self.Force = Force.copy()
        self.SpringConstant=SpringConstant
        self.SetOffsetAndVelocity(Extension[0],Velocity)
        self.WorkDigitized=None
    @property
    def Separation(self):
        return self.Extension
    def update_work(self):
        """
        Updates the internal work variable
        """
        self.SetWork(self.CalculateForceCummulativeWork())      
    def SetOffsetAndVelocity(self,Offset,Velocity):
        """
        Sets the velocity and offset used in (e.g.) ZFuncSimple. 
        Also re-calculates the work 

        Args:
            Offset:  offset in distance (same units of extension)
            Velocity: slope (essentially, effective approach/retract rate).
        Returns:
            Nothing
        """
        self.Offset = Offset
        self.Velocity = Velocity
        self.update_work()
    def GetWorkArgs(self,ZFunc):
        """
        Gets the in-order arguments for the work functions
        Args:
            ZFunc: see GetDigitizedBoltzmann
        """
        return self.SpringConstant,self.Velocity,self.Time,self.Extension
    @property
    def ZFunc(self):
        return ZFuncSimple
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
        Z = self.ZFunc(self)
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
    _ = [o.update_work() for o in PullingObjects]

def Exp(x):
    return np.exp(x)

def ForwardWeighted(nf,nr,v,W,Wn,delta_A,beta):
    """
    Returns the weighted value for the forward part of the bi-directionary free
    energy landscape
    
    Args: see EnsembleAverage
    """
    return (v*nf*Exp(-beta*W))/(nf + nr*Exp(-beta*(Wn - delta_A)))

def ReverseWeighted(nf,nr,v,W,Wn,delta_A,beta):
    """
    Returns the weighted value for a reverse step

    Args: see EnsembleAverage
    """
    #note reverse swaps n1 <-> n0, and delta_A -> -Delta A (Hummer,2010,e19)
    return (v*nr*Exp(-beta*(W + delta_A)))/(nr + nf*Exp(-beta*(Wn + delta_A)))

def _work_offset_value(works,**kw):
    return np.mean(works,**kw)

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
    # catch over and underflow errors, since these will screw things up later
    with np.errstate(over="raise",under="raise"):
        try:
            Forward = 1/(nr + nf * Exp(Beta * (ForwardWork-DeltaA)))
            Reverse = 1/(nf + nr * Exp(Beta * (ReverseWork+DeltaA)))
        except (RuntimeWarning,FloatingPointError) as e:
            print("Weierstrass: Over/underflow encountered. " + \
                  "Need fewer kT of integrated work. Try reducing data size")
            raise(e)
    # we really only case about the abolute value of the expression, since
    # we want the two sides to be equal...
    return abs(np.mean(Forward)-np.mean(Reverse))

def _fwd_and_reverse_w_f(fwd,rev):
    """
    Returns: the forward and reverse work's last point, offset to the 
    mean of the forward, or the naegation of that mean for the reverse work

    Args:
        fwd: list of forward objects
        rev: list of reverse objects
    Returns:
        tuple of <forward offset, reverse offset (negation of forward,
                  Fwd work offsets, reverse work offsets>
    """
    # get the work in terms of beta, should make it easier to converge
    w_f_fwd = np.array([f.Work[-1] for f in fwd])
    w_f_rev = np.array([f.Work[-1] for f in rev])
    # offset the forward and reverse work, to make sure we dont have any
    # floating point problems
    offset_fwd = _work_offset_value(w_f_fwd)
    offset_rev = -offset_fwd
    w_f_fwd -= offset_fwd
    w_f_rev -= offset_rev
    return offset_fwd,offset_rev,w_f_fwd,w_f_rev

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
    # POST: reverse is not zero; have at least one
    beta = Forward[0].Beta
    # multiply by beta, so we aren't dealing with incredibly small numbers
    offset_fwd,_,Fwd,Rev = _fwd_and_reverse_w_f(Forward,Reverse)
    max_r,max_f = np.max(np.abs(Rev)),np.max(np.abs(Fwd))
    max_abs = max(max_r,max_f)
    Max = max_abs
    Min = -max_abs
    # only look between +/- the max. Note that range is guarenteed positive
    Range = Max-Min
    FMinArgs = dict(maxfun=maxiter,**kwargs)
    # note we set beta to one, since it is easier to solve in units of kT
    list_v = []
    ToMin = lambda A: DistanceToRoot(A,Beta=beta,ForwardWork=Fwd,
                                     ReverseWork=Rev)
    xopt = fminbound(ToMin,x1=-Max,x2=Max,**FMinArgs)
    to_ret = (xopt)
    return to_ret + offset_fwd

def _check_inputs(objects,expected_inputs,f_input):
    """
    ensures that all of objects have a consistent z and size

    Args:
        objects: list of InverseWeierstrass objects
        expected_inputs: list of expected inputs
        f_input: function, takes in element of objects, returns list like
        expected_inputs
    Returns:
        nothing, throws an error if something was wrong
    """
    error_kw = dict(atol=0,rtol=1e-4)
    for i,u in enumerate(objects):
        actual_data = f_input(u)
        err_data = "iwt needs all objects to have the same properties.\n" + \
                   "Expected (z0,v,N,k,kT)={:s}, but object {:d} had {:s}".\
                   format(str(expected_inputs),i,str(actual_data))
        np.testing.assert_allclose(expected_inputs,actual_data,
                                   err_msg=err_data,**error_kw)
        # POST: data matches
    # POST: all data and sizes match

def _work_weighted_value(values,value_func,**kw):
    mean_arg = value_func(v=values,**kw)
    return np.mean(mean_arg,axis=0)

def get_work_weighted_object(objs,delta_A=0,**kw):
    """
    Gets all the information necessary to reconstruct 
    
    Args:
        objs: list of FEC_Pulling objects
        delta_A: the free energy difference between the forward and reverse,
        as defined near Hummer, 2010, eq 19.

        **kw: input to _work_weighted_value (value_func, and its kw)

    returns:
        _WorkWeighted instance
    """
    n_objs = len(objs)
    if (n_objs == 0):
        to_ret = _WorkWeighted([],0)
        to_ret.set_variables(0,0,0)
        return to_ret
    # POST: have at least one thing to do...
    works = np.array([u.Work for u in objs])
    force = np.array([u.Force for u in objs])
    force_sq = np.array([u.Force**2 for u in objs])
    n_size_expected = objs[0].Force.size
    assert works.shape[0] == n_objs , "Programming error"
    assert works.shape[1] == n_size_expected , "Programming error"
    # POST: i runs over K ('number of objects')
    # POST: j runs over z ('number of bins', except no binning)
    # subtract the mean work 
    offset = np.mean(works)
    works -= offset
    delta_A -= offset
    key = objs[0]
    beta = key.Beta
    k = key.SpringConstant
    Wn_raw = np.array([w[-1] for w in works])
    Wn = (np.ones(works.shape).T * Wn_raw).T
    weighted_kw = dict(delta_A=delta_A,beta=beta,W=works,Wn=Wn,**kw)
    partition = _work_weighted_value(values=1,**weighted_kw)
    assert partition.size == n_size_expected , "Programming error"
    weighted_force = \
        _work_weighted_value(values=force,**weighted_kw)/partition
    weighted_force_sq = \
        _work_weighted_value(values=force_sq,**weighted_kw)/partition
    to_ret = _WorkWeighted(objs,offset)
    to_ret.set_variables(partition=partition,
                         f_work_weighted=weighted_force,
                         f_squared_work_weighted=weighted_force_sq)
    return to_ret


def free_energy_inverse_weierstrass(unfolding,refolding=[]):
    """
    XXX DEBUGGING REPLACE

    Args:
        <un/re>folding: list of unfolding and refolding objects to use
    """
    n_f = len(unfolding)
    n_r = len(refolding)
    assert len(unfolding) > 0 , "IWT recquires at least one unfolding object"
    # POST: at least one to look at
    key = unfolding[0]
    input_check = lambda x: [x.Offset,x.Velocity,x.SpringConstant,x.Force.size,
                             x.kT]
    unfolding_inputs = input_check(key)
    # set z0 -> z0+v, v -> -v for redfolding
    z0,v = unfolding_inputs[0],unfolding_inputs[1]
    t = max(key.Time)-min(key.Time)
    zf = z0+v*t
    refolding_inputs = [zf,-v] + unfolding_inputs[2:]
    _check_inputs(unfolding,unfolding_inputs,input_check)
    _check_inputs(refolding,refolding_inputs,input_check)
    # POST: refolding and unfolding objects are OK
    # get the free energy change between the states (or zero, if none)
    delta_A = NumericallyGetDeltaA(unfolding,refolding)
    kw = dict(delta_A=delta_A,nr=n_r,nf=n_f)
    unfolding = get_work_weighted_object(unfolding,value_func=ForwardWeighted,
                                         **kw)
    refolding = get_work_weighted_object([],value_func=ReverseWeighted,
                                         **kw)
    # add up the weighted results (if no refolding, we are just adding 0)
    weighted_force     = unfolding.f         + refolding.f
    weighted_force_sq  = unfolding.f_squared + refolding.f_squared
    weighted_partition = unfolding.partition + refolding.partition
    weighted_variance = weighted_force_sq - (weighted_force**2)
    beta = key.Beta
    k = key.SpringConstant
    A_z =  (-1/beta)*np.log(weighted_partition)
    A_z_dot = weighted_force
    one_minus_A_z_dot_over_k = beta * weighted_variance/k
    first_deriv_term  = -A_z_dot**2/(2*k)
    second_deriv_term = 1/(2*beta) * np.log(one_minus_A_z_dot_over_k)
    G_0 = A_z + first_deriv_term +second_deriv_term
    z = key.ZFunc(key)
    q = z - A_z_dot/k
    to_ret = Landscape(q=q,energy=G_0,kT=1/beta,
                       free_energy_A=A_z,first_deriv_term=first_deriv_term,
                       second_deriv_term=second_deriv_term)
    return to_ret


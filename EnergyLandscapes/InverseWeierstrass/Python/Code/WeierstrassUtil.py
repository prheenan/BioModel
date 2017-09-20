# force floating point division. Can still use integer with //
from __future__ import division
# other good compatibility recquirements for python3
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys,copy

from FitUtil.EnergyLandscapes.InverseWeierstrass.Python.Code import \
    InverseWeierstrass

def _default_slice_func(obj,s):
    """
    Returns: a copy of obj, sliced to s 
    """
    to_ret = copy.deepcopy(obj)
    to_ret._slice(s)
    n_time = to_ret.Time.size
    assert ((n_time == to_ret.Force.size) and \
            (n_time == to_ret.Separation.size)) , \
        "Not all x/y values the same. Expected {:d}, got {:s}".\
        format(n_time,str([to_ret.Force.size,to_ret.Separation.size]))
    return to_ret 

def ToIWTObject(o,Offset=0,**kw):
    """
    Returns: o, truend into a IWT object
    """
    obj = InverseWeierstrass.FEC_Pulling_Object(Time=o.Time,
                                                Extension=o.Separation,
                                                Force=o.Force,
                                                SpringConstant=o.SpringConstant,
                                                Velocity=o.Velocity,
                                                Offset=Offset,
                                                **kw)
    return obj

def ToIWTObjects(TimeSepForceObjects):
    """
    Converts TimeSepForceObjects to InverseWeierstrass objects

    Args:
        TimeSepForceObjects: list of TimeSepForceObjects to transform
    """
    Objs = [ToIWTObject(o) for o in TimeSepForceObjects]
    return Objs

def split_into_iwt_objects(d,z_0,v,
                           idx_end_of_unfolding=None,idx_end_of_folding=None,
                           flip_forces=False,
                           slice_to_use=None,f_split=None,
                           slice_func=None,
                           unfold_start_idx=None,**kw):
    """
    given a 'raw' TimeSepForce object, gets the approach and retract 
    as IWT objects, accounting for the velocity and offset of the separation

    Args:
        slice_func: takes in a TimeSepForce object and a slice, returns
        the sliced data
    
        d: Single TimeSepForce object to split. A single retract/approach
        idx_end_of_unfolding: where the unfolding stops. If not given, we
        assume it happens directly in the middle (ie: default is no 'padding').

        idx_end_of_folding: where unfolding stops. If not given, we assume
        it happens at exactly twice where the folding stops
    
        fraction_for_vel: fit this much of the retract/approach
        separation versus time to determine the true velocity
        
        f_split: if not none, a function taking in data and returning 
        (idx_end_of_unfolding,idx_end_of_folding)
    returns:
        tuple of <unfolding,refolding> IWT Object
    """
    if (slice_func is None):
        slice_func = _default_slice_func
    if (f_split is not None):
        unfold_start_idx,idx_end_of_unfolding,idx_end_of_folding = f_split(d)
    if (unfold_start_idx is None):
        unfold_start_idx = 0
    if (idx_end_of_unfolding is None):
        idx_end_of_unfolding = int(np.ceil(d.Force.size/2))
    if (idx_end_of_folding is None):
        idx_end_of_folding = 2 * idx_end_of_unfolding
    if (flip_forces):
        d.Force *= -1
    # get the unfolding and unfolds
    slice_unfolding = slice(unfold_start_idx,idx_end_of_unfolding)
    unfold_tmp = slice_func(d,slice_unfolding)
    slice_folding = slice(idx_end_of_unfolding,idx_end_of_folding)
    fold_tmp = slice_func(d,slice_folding)
    # convert all the unfolding objects to IWT data
    try:
        IwtData = ToIWTObject(unfold_tmp,**kw)
        IwtData_fold = ToIWTObject(fold_tmp,**kw)
    except (AttributeError,KeyError) as e:
        # Rob messes with the notes; he also gives the velocities
        IwtData = RobTimeSepForceToIWT(unfold_tmp,ZFunc=None,**kw)
        IwtData_fold = RobTimeSepForceToIWT(fold_tmp,ZFunc=None,**kw)
    # switch the velocities of all ToIWTObject folding objects..
    # set the velocity and Z functions
    delta_t = IwtData.Time[-1]-IwtData.Time[0]
    z_f = z_0 + v * delta_t
    IwtData.SetOffsetAndVelocity(z_0,v)
    IwtData_fold.SetOffsetAndVelocity(z_f,-v)
    return IwtData,IwtData_fold    

def get_unfold_and_refold_objects(data,number_of_pairs,flip_forces=False,
                                  slice_func=None,**kwargs):
    """
    Splits a TimeSepForceObj into number_of_pairs unfold/refold pairs,
    converting into IWT Objects.
    
    Args:
        data: TimeSepForce object to use
        number_of_pairs: how many unfold/refold *pairs* there are (ie: single
        'out and back' would be one, etc
        flip_forces: if true, multiply all the forces by -1
        get_slice: how to slice the data

        slice_func: see split_into_iwt_objects
        
        kwargs: passed to split_into_iwt_objects
    Returns:
        tuple of <unfold,refold> objects
    """
    if (slice_func is None):
        slice_func =  _default_slice_func
    n = number_of_pairs
    pairs = [slice_func(data,get_slice(data,i,n)) for i in range(n) ]
    # POST: pairs has each slice (approach/retract pair) that we want
    # break up into retract and approach (ie: unfold,refold)
    unfold,refold = [],[]
    for p in pairs:
        unfold_tmp,refold_tmp = \
            split_into_iwt_objects(p,flip_forces=flip_forces,
                                   slice_func=slice_func,**kwargs)
        unfold.append(unfold_tmp)
        refold.append(refold_tmp)
    return unfold,refold        
    
    
def get_slice(data,j,n):
    """
    Gets a slice for a TimeSepForce object 'data'
    
    Args:
        j: which slice (up to n-1)
        n: maximum number of slices
    Returns:
        new slice object
    """
    Length = data.Force.size    
    data_per_curve = int(np.round(Length/n))    
    return slice(j*data_per_curve,(j+1)*data_per_curve,1)
   
def convert_to_iwt(time_sep_force,frac_vel=0.1):
    """
    Converts a TimeSepForce object into a iwt object (assuming just one 
    direction)
    
    Args:
        time_sep_force: to convert
        frac_vel: the fractional number of points to use for getting the    
        (assumed constant) velocity
    Returns:
        iwt_object 
    """
    iwt_data = ToIWTObject(time_sep_force)
    return iwt_data    
    
def convert_list_to_iwt(time_sep_force_list,**kwargs):
    """
    see convert_to_iwt, except converts an entire list
    """
    return [convert_to_iwt(d) for d in time_sep_force_list]


def RobTimeSepForceToIWT(o,ZFunc,**kw):
    """
    converts a Rob-Walder style pull into a FEC_Pulling_Object

    Args:
         o: TimeSepForce object with Robs meta information
         ZFunc: the z function (schedule) passed along
         fraction_for_vel : see set_separation_velocity_by_first_frac
    Returns:
         properly initialized FEC_Pulling_Object for use in IWT
    """
    # spring constant should be in N/m
    k = o.Meta.__dict__["K"]
    velocity = o.Meta.__dict__["RetractVelocity"]
    Obj = InverseWeierstrass.FEC_Pulling_Object(Time=o.Time,
                                                Extension=o.Separation,
                                                Force=o.Force,
                                                SpringConstant=k,
                                                Velocity=velocity,
                                                ZFunc=ZFunc,**kw)
    return Obj
    

def iwt_ramping_experiment(data,number_of_pairs,number_of_bins,kT,
                           flip_forces=False,velocity=0,**kw):
    """

    """
    unfold,refold = \
        get_unfold_and_refold_objects(data,
                                      number_of_pairs=number_of_pairs,
                                      flip_forces=flip_forces,
                                      kT=kT,**kw)
    if (velocity > 0):
        for un,re in zip(unfold,refold):
            # keep the offsets, reset the velocites
            un.SetOffsetAndVelocity(un.Offset,velocity)
            re.SetOffsetAndVelocity(re.Offset,velocity * -1)
    # POST: have the unfolding and refolding objects, get the energy landscape
    LandscapeObj =  InverseWeierstrass.\
            free_energy_inverse_weierstrass(unfold,refold)  
    return LandscapeObj



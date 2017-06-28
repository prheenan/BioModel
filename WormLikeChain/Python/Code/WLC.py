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


from FitUtil import fit_base

from FitUtil.WormLikeChain.Python.Code.WLC_Utils import \
    WlcExtensible_Helper
    
from scipy.interpolate import interp1d

def ExtensionPerForceOdjik(kbT,Lp,L0,K0,F):
    """
    Returns the extension at each given force; where invalid (ie: force <= 0),
    returns nan
    
    Args:
        see InvertedWlcForce
    Returns:
        extension at each force
    """
    # need to cast the sqrt to a real to make this work
    f_complex = F.astype(np.complex128)
    # determine where F > 0 (this is where we can use Odjik)
    safe_idx = np.where(F > 0)
    sqrt_safe = np.sqrt(kbT/(f_complex[safe_idx]*Lp))
    # everywhere we can't use Odjik (F=0), set the value to nan
    to_ret = np.ones(F.size) * np.nan
    # elsewhere, determine the actual extension
    to_ret[safe_idx] = np.real(L0 * (1 - sqrt_safe/2 + F[safe_idx]/K0))
    return to_ret

def SeventhOrderExtAndForceGrid(kbT,Lp,L0,K0,F,MaxForce=None):
    """
    Given extension data, parameters, and a force, creates a WLC-based 
    grid, including Bouchiat polynomials. This is essentially the (smooth)
    fit to the data.

    Args:
        kbt,lP,L0,f0,F: see InvertedWlcForce
        MaxForce: the maximum ofrce to use. 
    """
    # grid the force uniformly (x vs F is essentially a 1-1 map)
    N = F.size
    UpSample = 2
    if (MaxForce is None):
        MaxForce = np.nanmax(F)
    ForceGrid = np.linspace(start=0,stop=MaxForce,num=N*UpSample)
    # get the extension predictions on the uniform grid (essentially
    # this is a detailed 1-1 map of how to go between force and ext)
    ExtPred = ExtensionPerForceOdjik(kbT,Lp,L0,K0,ForceGrid)
    # return the extension and force on the *grid*. These are smooth, and
    # (for fitting purposes) should be interpolated back to the original
    # extension / force data
    return ExtPred,ForceGrid

def InterpolateFromGridToData(XGrid,YGrid,XActual,
                              bounds_error=False,kind='linear',
                              fill_value='extrapolate'):
    """
    interpolate the force from the predicted extension grid to the actual
    extensions -- which is what we care about

    Note: by default this linearly extrapolates for edge cases.
    Considering the WLC is linear at the start (hooke) and the end 
    (stretching backbone), this is probably 
    ok, but the user should be sure this behavior is desired

    Args:
        XGrid,YGrid: two arrays of the same length; we interpolate Y
        along the X grid. Probably the outputs of 
        SeventhOrderForceAndExtGrid

        XActual: wherever we want the intepolated y values.
    """
    IntepolationMap = interp1d(XGrid,YGrid,bounds_error=bounds_error,
                               kind=kind,fill_value=fill_value)
    # do the actual mapping 
    return IntepolationMap(XActual)


def _inverted_wlc_full(ext,kbT,Lp,L0,K0,F,max_force=None,odjik_as_guess=True):
    """
    Function to fit F vs ext using an ext(F). This allows us to get a 
    good initial guess for F(ext). The force is gridded, giving 
    a smooth interpolation. 

    Args:
        ext: the extension data, size N
        F: the force data, size N 
        others: See WLC_Fit.WlcNonExtensible
        max_force: to do the inversion, we need to know what maximum
        Force to use. We take the max of the force in this slice to make it 
        happen
        
        odjik_as_guess: if true, uses the odjik_as_guess as an initial (smooth)
        guess to the force for the Wang 1997 fit 
    Returns:
        tuple of <gridded x, gridded y, y value at each ext>
    """
    if (max_force is None):
        max_force = np.nanmax(F)
    ExtGrid,ForceGrid = SeventhOrderExtAndForceGrid(kbT,Lp,L0,K0,F,max_force)
    Force = InterpolateFromGridToData(ExtGrid,ForceGrid,ext)
    # reslice the grid to be within the useful range of the data
    min_x,max_x = np.nanmin(ext),np.nanmax(ext)
    # only look at points on the grid which are within the original bounds of 
    # the data 
    ext_safe_idx = np.where(np.isfinite(ExtGrid))[0]
    ext_grid_final = ExtGrid[ext_safe_idx]       
    force_grid_final = ForceGrid[ext_safe_idx]
    grid_idx_of_interest = np.where( (ext_grid_final >= min_x) &
                                     (ext_grid_final <= max_x))[0]
    ext_grid_final = ext_grid_final[grid_idx_of_interest]                      
    force_grid_final = force_grid_final[grid_idx_of_interest]
    # POST: the grid is within the original bounds. Update if need be 
    if (odjik_as_guess):
        # use the odjik as a guess to the higher-order wang fit
        wang_dict = dict(kbT=kbT,Lp=Lp,L0=L0,K0=K0)
        Force = WlcExtensible_Helper(ext=ext,F=Force,**wang_dict)
        force_grid_final = WlcExtensible_Helper(ext=ext_grid_final,
                                                F=force_grid_final,
                                                **wang_dict)
    return ext_grid_final,force_grid_final,Force

def inverted_wlc(ext,force,L0,Lp,K0,kbT,**kwargs):
    """
    convenience wrapper for _inverted_wlc_full; makes it easier to 
    fit L0 or Lp
    
    Args:
        see _inverted_wlc_full; note the order change 
    returns:
        see _inverted_wlc_full
    """
    return _inverted_wlc_full(ext=ext,kbT=kbT,Lp=Lp,L0=L0,K0=K0,F=force,
                              **kwargs)
    
def inverted_wlc_force(*args,**kwargs):
    """
    convenience wrapper for inverted_wlc; only returns the predicted force
    
    Args:
        see inverted_wlc
    returns:
        force at each separation
    """
    ext_grid,force_grid,force_predicted = inverted_wlc(*args,**kwargs)
    return force_predicted
    
def fit(separation,force,brute_dict,**kwargs):
    """
    Fit to the force versus separation curve, assuming
    an exetensible WLC
    
    Args:
        separation: extension of the molecule
        force: force applied to the molecule; positive is pulling
        brute_dict: dictionary given to optimize.brute; should include 
        (at minimum) the range for the contour length.
        
        **kwargs: passed to inverted_wlc_force
    returns:
        tuple of <predicted parameters, predicted force atg each separation>
    """
    func = lambda *args: inverted_wlc_force(separation,force,*args,**kwargs)
    brute_dict['full_output']=False
    x0 = fit_base.brute_optimize(func,force,brute_dict=brute_dict)
    _,_,predicted_force = inverted_wlc(separation,force,*x0,**kwargs)
    return x0,predicted_force

    

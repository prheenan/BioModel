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
import FitUtil.WormLikeChain.Python.Code.WLC as WLC
_kbT = 4.11e-21
_K0 = 1200e-12

class test_object:
    def __init__(self,name,max_force_N,ext=None,**kwargs):
        if (ext is None):
            L0 = dict(**kwargs)["L0"]
            ext = np.linspace(0,L0,500)
        self.name = name
        self.max_force_N = max_force_N
        self.ext = ext
        self.param_values = dict(**kwargs)

def GetBullData(StepInNm=0.05):
    """
    Returns samples from first unfold of Figure S2.a 
    http://pubs.acs.org/doi/suppl/10.1021/nn5010588

    Bull, Matthew S., Ruby May A. Sullan, Hongbin Li, and Thomas T. Perkins.
"Improved Single Molecule Force Spectroscopy Using Micromachined Cantilevers"
    """
    # get the extensions used
    maxXnm = 19
    nSteps= maxXnm/StepInNm
    x = np.linspace(0,maxXnm,num=nSteps) * 1e-9
    L0 = 0.34e-9 * 64
    """
    # note: from Supplemental, pp 14 of 
    Edwards, Devin T., Jaevyn K. Faulk et al 
    "Optimizing 1-mus-Resolution Single-Molecule Force Spectroscopy..."
    """
    Lp = 0.4e-9
    ParamValues = dict(kbT = 4.11e-21,L0 = L0,
                       Lp =  Lp,K0 = 1318.e-12)
    Name = "Bull_2014_FigureS2"
    noiseN = 6.8e-12
    expectedMax=80e-12
    return test_object(name="Bull",max_force_N=expectedMax,ext=x,**ParamValues)
    
def GetBouichatData(StepInNm=1):
    """
    Returns samples from data from Figure 1
    From "Estimating the Persistence Length of a Worm-Like Chain Molecule ..."
    C. Bouchiat, M.D. Wang, et al.
    Biophysical Journal Volume 76, Issue 1, January 1999, Pages 409-413

web.mit.edu/cortiz/www/3.052/3.052CourseReader/38_BouchiatBiophysicalJ1999.pdf
    
    Returns:
        tuple of <z,F> in SI units
    """
    # upper and lower bound is taken from Figure 1, note nm scale
    maxExtNm = 1335
    # figure out the number of steps at this interpolation
    nSteps = int(np.ceil(maxExtNm/StepInNm))
    # get all the extension values
    x = np.linspace(start=0,stop=maxExtNm,num=nSteps,endpoint=True) * 1e-9
    # write down their parameter values, figure 1 inset
    ParamValues = dict(kbT = 4.11e-21,L0 = 1317.52e-9,
                       Lp =  40.6e-9,K0 = 1318.e-12)
    # the expected maximum fitted force is also from figure 1
    expectedMax = 48e-12
    Name = "Bouchiat_1999_Figure1"
    return test_object(name=Name,max_force_N=expectedMax,ext=x,**ParamValues)


def get_fitting_parameters_with_noise(ext_pred,force_grid,params_fit,
                                      noise_amplitude_N,ranges,Ns=10,
                                      **brute_kwargs):
    """
    Gets the fitting parameters after corrupting a known signal with noise

    Args:
        ext_pred: the x values for the fitting
        force_grid: the y values for the fitting
        params_fit: to be passed to wc_contour
        noise_amplitude_N: the (assumed uniformly normal) noise
        ranges: for parameters to fit, passed to brute
        Ns: number of points to use, for brute
        **brute_kwargs: passed to scipy.optimize.brute
    Returns:
        tuple of <fit parameters, predicted y, noisy y>
    """
    noise_unitless = (np.random.normal(size=force_grid.size))
    noise_N = noise_amplitude_N * noise_unitless
    force_noise = force_grid + noise_N
    brute_dict = dict(ranges=ranges,Ns=Ns,**brute_kwargs)
    x0,y = WLC.wlc_contour(separation=ext_pred,force=force_noise,
                           brute_dict=brute_dict,
                           **params_fit)
    return x0,y,force_noise

def ssDNA_example_data():
    params_ssDNA   = dict(kbT=_kbT,K0=_K0,L0=60e-9,Lp=0.7e-9)
    to_ret =  test_object(name="ssDNA",
                          max_force_N=65e-12,
                          **params_ssDNA)
    return to_ret

def dsDNA_example_data():
    params_dsDNA   = dict(kbT=_kbT,K0=_K0,L0=500e-9,Lp=50e-9)
    to_ret = test_object(name="dsDNA",
                         max_force_N=65e-12,
                         **params_dsDNA)
    return to_ret

def protein_example_data():
    params_protein = dict(kbT=_kbT,K0=_K0,L0=40e-9,Lp=0.3e-9)
    to_ret = test_object(name="protein",
                         max_force_N=100e-12,
                         **params_protein)
    return to_ret

def get_ext_and_force(test_obj):
    ext = test_obj.ext
    param_values = test_obj.param_values
    force = np.linspace(0,test_obj.max_force_N,ext.size)
    ext_pred,force_grid = WLC.SeventhOrderExtAndForceGrid(F=force,
                                                          **param_values)
    idx_finite = np.where(np.isfinite(ext_pred))[0]
    idx_good = np.where( (ext_pred[idx_finite] >= min(ext)) & 
                         (ext_pred[idx_finite] <= max(ext)) )[0]
    ext_pred = ext_pred[idx_finite[idx_good]]
    force_grid = force_grid[idx_finite[idx_good]]
    return ext_pred,force_grid
    
        

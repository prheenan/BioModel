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

sys.path.append("../../../../../")

import FitUtil.WormLikeChain.Python.Code.WLC as WLC
from GeneralUtil.python import PlotUtilities,GenUtilities

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
                                      noise_amplitude_N,ranges,Ns=40,
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


def test_parameter_set(test_obj,debug_plot_base=None,
                       Lp_relative_tolerance= 5e-2,
                       L0_relative_tolerance = 1e-2,noise_ampl_N=None):
    """
    Tests the given parameter set. throws an error if it failss

    Args:
        test_obj: the test object to use 
        debug_plot_base: if not none, saves plot(s) starting with file path
     
        L0_relative_tolerance: maximum acceptable fitting tolerance (e.g. 0.01
        is 1%)
    
        noise_ampl_N: a *list* of noises (in newtons) to use 

    Returns:
        nothing, throws an error if it fails
    """
    if (noise_ampl_N is None):
        noise_ampl_N = [0,1e-12,5e-12,10e-12]
    param_values,max_force_N = test_obj.param_values,test_obj.max_force_N
    # determine the noiseless curve
    L0 = param_values["L0"]
    ext = test_obj.ext
    force = np.linspace(0,test_obj.max_force_N,ext.size)
    ext_pred,force_grid = WLC.SeventhOrderExtAndForceGrid(F=force,
                                                          **param_values)
    idx_finite = np.where(np.isfinite(ext_pred))[0]
    idx_good = np.where( (ext_pred[idx_finite] >= min(ext)) & 
                         (ext_pred[idx_finite] <= max(ext)) )[0]
    ext_pred = ext_pred[idx_finite[idx_good]]
    force_grid = force_grid[idx_finite[idx_good]]
    x_grid,y_grid,y_pred = WLC.inverted_wlc(ext=ext_pred,
                                            force=force_grid,
                                            **param_values)
    # do the fitting; we use a series of noise amplitudes to check for
    # robustness. 
    params_L0_fit = dict([  [k,v] for k,v in param_values.items() if k != "L0"])
    params_L0_and_Lp_fit = dict([  [k,v] for k,v in param_values.items() 
                                   if (k != "L0" and k != "Lp")])
    factor_L0 = 5
    factor_Lp = 10
    for noise_tmp in noise_ampl_N:
        common_kwargs = dict(ext_pred=ext_pred,
                             force_grid=force_grid,
                             noise_amplitude_N=noise_tmp)
        max_x = np.nanmax(ext_pred)
        # # Fit L0
        # make the dictionary with all the fitting information
        ranges_L0 = [ (max_x/factor_L0,factor_L0*max_x) ]
        fit_dict_L0 = dict(params_fit=params_L0_fit,
                           ranges=ranges_L0,
                           **common_kwargs)
        x0,y,force_noise = get_fitting_parameters_with_noise(**fit_dict_L0)
        # ensure the error is within the bounds
        L0_relative_error = (abs((x0-L0)/L0))[0]
        assert L0_relative_error < L0_relative_tolerance , \
            "Error {:.2g} not in tolerance".format(L0_relative_error)
        # # Fit L0 and Lp
        ranges_L0_and_Lp = ranges_L0 + [ (0,max_x/factor_Lp) ]
        Lp = param_values["Lp"]
        fit_dict_L0_and_Lp = dict(params_fit=params_L0_and_Lp_fit,
                                  ranges=ranges_L0_and_Lp,
                                  **common_kwargs)
        """
        print(ranges_L0_and_Lp)
        x0,y,force_noise = \
                get_fitting_parameters_with_noise(**fit_dict_L0_and_Lp)
        # ensure the error is within the bounds
        L0_relative_error = (abs((x0-L0)/L0))[0]
        assert L0_relative_error < L0_relative_tolerance , \
            "Error {:.2g} not in tolerance".format(L0_relative_error)
        # check Lp, also
        Lp_relative_error = (abs((x0-Lp)/Lp))[1]
        assert Lp_relative_error < Lp_relative_tolerance , \
            "Error {:.2g} not in tolerance".format(Lp_relative_error)
        """


    # POST: all errors in bounds
    if (debug_plot_base is not None):
        fig = PlotUtilities.figure(figsize=(4,7))
        ext_pred_plot = ext_pred * 1e9
        force_noise_plot = force_noise * 1e12
        y_plot = y * 1e12
        y_pred_plot = y_pred * 1e12
        plt.subplot(2,1,1)
        plt.plot(ext_pred_plot,force_noise_plot,'k-',alpha=0.3,label="Data")
        plt.plot(ext_pred_plot,y_plot,'b--',label="Prediction")    
        PlotUtilities.lazyLabel("","Force (pN)","")
        plt.subplot(2,1,2)
        plt.plot(ext_pred_plot,y_pred_plot,'b--',label="Noiseless Curve")
        plt.plot(ext_pred_plot,force_noise_plot,'k-',alpha=0.3,
                 label="Noisy Curve")
        PlotUtilities.lazyLabel("Separation (nm)","Force (pN)","")
        base_dir = "./out/"
        GenUtilities.ensureDirExists(base_dir)
        PlotUtilities.savefig(fig,base_dir + debug_plot_base + ".png")

def run():
    """
    Runs some unit testing on the WLC fitting. 
    """
    """
    # note: from Supplemental, pp 14 of 
    Edwards, Devin T., Jaevyn K. Faulk et al 
    "Optimizing 1-mus-Resolution Single-Molecule Force Spectroscopy..."
    """
    kbT = 4.11e-21
    K0 = 1200e-12
    np.random.seed(42)
    params_dsDNA   = dict(kbT=kbT,K0=K0,L0=500e-9,Lp=50e-9)
    params_ssDNA   = dict(kbT=kbT,K0=K0,L0=60e-9,Lp=0.7e-9)
    params_protein = dict(kbT=kbT,K0=K0,L0=40e-9,Lp=0.3e-9)
    test_objects = [GetBouichatData(),
                    GetBullData(),
                    # try a few 'fake' common cases
                    test_object(name="ssDNA",
                                max_force_N=65e-12,
                                **params_ssDNA),
                    test_object(name="dsDNA",
                                max_force_N=65e-12,
                                **params_dsDNA),
                    test_object(name="protein",
                                max_force_N=100e-12,
                                **params_protein)]
    for t in test_objects:
        test_parameter_set(t,debug_plot_base=t.name)
        

if __name__ == "__main__":
    run()

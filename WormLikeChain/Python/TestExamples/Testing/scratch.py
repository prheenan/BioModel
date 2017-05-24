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
from GeneralUtil.python import PlotUtilities

class test_object:
    def __init__(self,name,max_force_N,**kwargs):
        self.name = name
        self.max_force_N = max_force_N
        self.param_values = dict(**kwargs)


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
    # make the uniform noise go from -1 to 1
    uniform_noise = 2 * (np.random.uniform(size=force_grid.size) - 0.5)
    noise = noise_amplitude_N * uniform_noise 
    force_noise = force_grid + noise
    brute_dict = dict(ranges=ranges,Ns=Ns,**brute_kwargs)
    x0,y = WLC.wlc_contour(separation=ext_pred,force=force_noise,
                           brute_dict=brute_dict,
                           **params_fit)
    return x0,y,force_noise


def test_parameter_set(param_values,max_force_N,debug_plot_base=None,
                       L0_relative_tolerance = 10e-2,noise_ampl_N=None):
    """
    Tests the given parameter set. throws an error if it failss

    Args:
        param_values: the expected values of the parameter. Used to generate
        data

        max_force_N: the maximum force the force-extension curve should go to

        debug_plot_base: if not none, saves plot(s) starting with file path
     
        L0_relative_tolerance: maximum acceptable fitting tolerance (e.g. 0.01
        is 1%)
    
        noise_ampl_N: a *list* of noises (in newtons) to use 

    Returns:
        nothing, throws an error if it fails
    """
    if (noise_ampl_N is None):
        noise_ampl_N = [0,1e-12,5e-12,10e-12,20e-12]
    # determine the noiseless curve
    L0 = param_values["L0"]
    print(L0,"L0")
    ext = np.linspace(L0/100,L0*1.2,num=1000)
    force = np.linspace(0,max_force_N)
    ext_pred,force_grid = WLC.SeventhOrderExtAndForceGrid(F=force,
                                                          **param_values)
    x_grid,y_grid,y_pred = WLC.inverted_wlc(ext=ext_pred,
                                            force=force_grid,
                                            **param_values)
    # do the fitting; we use a series of noise amplitudes to check for
    # robustness. 
    params_fit = dict([  [k,v] for k,v in param_values.items() if k != "L0"])
    for noise_tmp in noise_ampl_N:
        # make the dictionary with all the fitting information
        ranges = [ [np.nanmax(ext_pred)/5,5*np.nanmax(ext_pred)]]
        fit_dict = dict(ext_pred=ext_pred,
                        force_grid=force_grid,
                        params_fit=params_fit,
                        ranges = ranges,
                        noise_amplitude_N=noise_tmp)
        x0,y,force_noise = get_fitting_parameters_with_noise(**fit_dict)
        # ensure the error is within the bounds
        L0_relative_error = (abs((x0-L0)/L0))[0]
        assert L0_relative_error < L0_relative_tolerance , \
            "Error {:.2g} not in tolerance".format(L0_relative_error)
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
        PlotUtilities.savefig(fig,debug_plot_base + ".png")

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
    params_dsDNA   = dict(kbT=kbT,K0=K0,L0=500e-9,Lp=50e-9)
    params_ssDNA   = dict(kbT=kbT,K0=K0,L0=60e-9,Lp=0.7e-9)
    params_protein = dict(kbT=kbT,K0=K0,L0=60e-9,Lp=0.3e-9)
    test_objects = [test_object(name="ssDNA",
                                max_force_N=65e-12,
                                **params_ssDNA),
                    test_object(name="dsDNA",
                                max_force_N=65e-12,
                                **params_dsDNA),
                    ]
    for t in test_objects:
        test_parameter_set(param_values=t.param_values,
                           max_force_N=t.max_force_N,
                           debug_plot_base=t.name)

if __name__ == "__main__":
    run()

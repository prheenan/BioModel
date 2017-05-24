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
from FitUtil.WormLikeChain.Python.TestExamples import util
from GeneralUtil.python import PlotUtilities,GenUtilities


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
    ext_pred,force_grid = util.get_ext_and_force(test_obj)
    x_grid,y_grid,y_pred = WLC.inverted_wlc(ext=ext_pred,
                                            force=force_grid,
                                            **param_values)
    # do the fitting; we use a series of noise amplitudes to check for
    # robustness. 
    params_L0_fit = dict([  [k,v] for k,v in param_values.items() if k != "L0"])
    params_L0_and_Lp_fit = dict([  [k,v] for k,v in param_values.items() 
                                   if (k != "L0" and k != "Lp")])
    factor_L0 = 2
    factor_Lp = 5
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
        x0,y,force_noise = util.get_fitting_parameters_with_noise(**fit_dict_L0)
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
        XXX fit Lp -- right now things are too slow. 
        x0,y,force_noise = \
                util.get_fitting_parameters_with_noise(**fit_dict_L0_and_Lp)
        # ensure the error is within the bounds
        L0_relative_error = (abs((x0-L0)/L0))[0]
        assert L0_relative_error < L0_relative_tolerance , \
            "Error {:.2g} not in tolerance".format(L0_relative_error)
        # check Lp, also
        Lp_relative_error = (abs((x0-Lp)/Lp))[1]
        assert Lp_relative_error < Lp_relative_tolerance , \
            "Error {:.2g} not in tolerance".format(Lp_relative_error)
        print(L0_relative_error,Lp_relative_error)
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
    np.random.seed(42)
    test_objects = [util.GetBouichatData(),
                    util.GetBullData(),
                    # try a few 'fake' common cases
                    util.ssDNA_example_data(),
                    util.dsDNA_example_data(),
                    util.protein_example_data()]
    for t in test_objects:
        test_parameter_set(t,debug_plot_base=t.name)
        

if __name__ == "__main__":
    run()

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
    def __init__(self,max_force_N,**kwargs):
        self.max_force_N = max_force_N
        self.param_values = dict(**kwargs)


def get_fitting_parameters_with_noise(ext_pred,force_grid,params_fit,
                                      noise_amplitude_N):
    # make the uniform noise go from -1 to 1
    uniform_noise = 2 * (np.random.uniform(size=force_grid.size) - 0.5)
    noise = noise_amplitude_N * uniform_noise 
    force_noise = force_grid + noise
    ranges = [ [np.nanmax(ext_pred)/5,5*np.nanmax(ext_pred)]]
    brute_dict = dict(ranges=ranges,Ns=40)
    x0,y = WLC.wlc_contour(separation=ext_pred,force=force_noise,
                           brute_dict=brute_dict,
                           **params_fit)
    return x0,y,force_noise


def test_parameter_set(param_values,max_force_N,debug_plot_base=None,
                       L0_relative_tolerance = 1e-2):
    L0 = param_values["L0"]
    ext = np.linspace(L0/100,L0,num=1000)
    force = np.linspace(0,max_force_N)
    ext_pred,force_grid = WLC.SeventhOrderExtAndForceGrid(F=force,
                                                          **param_values)
    x_grid,y_grid,y_pred = WLC.inverted_wlc(ext=ext_pred,
                                            force=force_grid,
                                            **param_values)
    noise_ampl_N = 20e-12
    params_fit = dict([  [k,v] for k,v in param_values.items() if k != "L0"])
    fit_dict = dict(ext_pred=ext_pred,
                    force_grid=force_grid,
                    params_fit=params_fit,
                    noise_amplitude_N=noise_ampl_N)
    x0,y,force_noise = get_fitting_parameters_with_noise(**fit_dict)
    L0_relative_error = abs((x0-L0)/L0)
    assert L0_relative_error < L0_relative_tolerance , \
        "Error {:.2g} not in tolerance".format(L0_relative_error)
    if (debug_plot_base is not None):
        fig = PlotUtilities.figure(figsize=(4,7))
        ext_pred_plot = ext_pred * 1e9
        force_noise_plot = force_noise * 1e12
        y_plot = y * 1e12
        y_pred_plot = y_pred * 1e12
        plt.subplot(2,1,1)
        plt.plot(ext_pred_plot,force_noise_plot,'k-',alpha=0.3)
        plt.plot(ext_pred_plot,y_plot,'b--')    
        PlotUtilities.lazyLabel("","Force (pN)","")
        plt.subplot(2,1,2)
        plt.plot(ext_pred_plot,y_pred_plot,'b--')
        plt.plot(ext_pred_plot,force_noise_plot,'k-',alpha=0.3)
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
    Lp = 50e-9
    L0 = Lp * 10
    kbT = 4.11e-21
    K0 = 1200e-12
    params_DNA = dict(kbT=4.11e-21,L0=500e-9,Lp=50e-9,K0=1200e-12)
    test_objects = [test_object(max_force_N=65e-12,**params_DNA)]
    for t in test_objects:
        test_parameter_set(param_values=t.param_values,
                           max_force_N=t.max_force_N,
                           debug_plot_base="./out")

if __name__ == "__main__":
    run()

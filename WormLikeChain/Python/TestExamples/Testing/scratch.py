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

def test_parameter_set(param_values,max_force_N,debug_plot_base=None):
    L0 = param_values["L0"]
    ext = np.linspace(L0/100,L0,num=1000)
    force = np.linspace(0,max_force_N)
    ext_pred,force_grid = WLC.SeventhOrderExtAndForceGrid(F=force,
                                                          **param_values)
    force_amplitude_pN = 20e-12
    # make the uniform noise go from -1 to 1
    uniform_noise = 2 * (np.random.uniform(size=force_grid.size) - 0.5)
    noise = force_amplitude_pN * uniform_noise 
    force_noise = force_grid + noise
    ranges = [ [max(ext)/5,5*max(ext)]]
    brute_dict = dict(ranges=ranges,Ns=40)
    x_grid,y_grid,y_pred = WLC.inverted_wlc(ext=ext_pred,
                                            force=force_grid,
                                            **param_values)
    ParamsFit = dict([  [k,v] for k,v in param_values.items() if k != "L0"])
    x0,y = WLC.wlc_contour(separation=ext_pred,force=force_noise,
                             brute_dict=brute_dict,
                             **ParamsFit)
    print( (x0-L0)/L0 * 100)
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
    ParamValues = dict(kbT=kbT,L0=L0,Lp=Lp,K0=K0)
    test_parameter_set(param_values=ParamValues,max_force_N=65e-12,
                       debug_plot_base="./out")

if __name__ == "__main__":
    run()

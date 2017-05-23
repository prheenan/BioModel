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
    ParamValues = dict(kbT =kbT,
                       Lp =  Lp,K0 = K0)
    ext = np.linspace(L0/100,L0,num=1000)
    force = np.linspace(0,20e-12)
    ext_pred,force_grid = WLC.SeventhOrderExtAndForceGrid(kbT,Lp,L0,K0,F=force)
    force_grid += force_grid * 0.05 *(np.random.normal(size=force_grid.size))
    ranges = [ [max(ext)/5,max(ext)]]
    brute_dict = dict(ranges=ranges)
    x_grid,y_grid,y_pred = WLC.inverted_wlc(ext=ext,force=force_grid,L0=L0,
                                            **ParamValues)
    x0,x,y = WLC.wlc_contour(separation=ext,force=y_pred,
                             brute_dict=brute_dict,
                             **ParamValues)                                            
    print(x0,x,y)
    plt.subplot(2,1,1)
    #plt.plot(ext_pred,force_grid,'k-',alpha=0.3)
    plt.plot(x,y,'b--')    
    plt.subplot(2,1,2)
    plt.plot(ext,y_pred,'b--')
    plt.plot(x_grid,y_grid,'k-',alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run()

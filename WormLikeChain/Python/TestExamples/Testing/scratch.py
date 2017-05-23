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
    ParamValues = dict(kbT =kbT,L0 = L0,
                       Lp =  Lp,K0 = K0)
    ext = np.linspace(L0/100,L0,num=1000)
    force = np.linspace(0,20e-12)
    ExtPred,ForceGrid = WLC.SeventhOrderExtAndForceGrid(kbT,Lp,L0,K0,F=force)
    ForceGrid += ForceGrid * 0.05 *(np.random.normal(size=ForceGrid.size))
    x_grid,y_grid,y_pred = WLC.inverted_wlc(ext=ext,force=ForceGrid,
                                            **ParamValues)
    plt.subplot(2,1,1)
    plt.plot(ExtPred,ForceGrid,'k-',alpha=0.3)
    plt.plot(ext,y_pred,'b--')    
    plt.subplot(2,1,2)
    plt.plot(ext,y_pred,'b--')
    plt.plot(x_grid,y_grid,'k-',alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run()

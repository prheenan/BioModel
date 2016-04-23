# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

baseDir = "../"
sys.path.append(baseDir)
sys.path.append("../../../")
import Code.WLC_Fit as WLC_Fit

def GetBouichatData():
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
    x = np.arange(0,1335,0.5) * 1e-9
    # write down their parameter values, figure 1 inset
    params = WLC_Fit.WlcParamValues(kbT = 4.11e-21,L0 = 1317.52e-9,
                                    Lp =  40.6e-9,K0 = 1318.e-12)
    # for the non-extensible model, really only want to fit up to
    # some high percentage of the contour length
    values = dict([(k,v.Value) for k,v in params.GetParamDict().items()])
    y = WLC_Fit.WlcExtensible(x,**values)
    # note, by the inset in figure 1 inset / 3 error bars, 2pN is an upper
    # bound on the error we have everywhere
    noiseAmplitude = 2e-12
    return x,y,params,noiseAmplitude

def run():
    """
    Runs some unit testing on the WLC fitting
    """
    x,y,params,noiseAmplitude = GetBouichatData()
    yPure = y.copy()
    y += noiseAmplitude * (np.random.rand(*y.shape)-0.5)*2
    # get an extensible and non-extensible model, choose whether to varying L0
    # and Lp
    extensibleFit = WLC_Fit.ExtensibleWlcFit(x,y,VaryL0=True,
                                             VaryLp=True,VaryK0=False)
    # make sure the parameters match
    assert extensibleFit.Info.ParamVals.CloseTo(params)
    nonExtensibleFit = WLC_Fit.NonExtensibleWlcFit(x,y,VaryL0=True,
                                                   VaryLp=False)
    print("Extensible Parameters")
    print(extensibleFit)
    print("Non-extensible Parameters")
    print(nonExtensibleFit)
    mFit = extensibleFit.Prediction
    mFitNon = nonExtensibleFit.Prediction
    # plot everything
    toNm = 1e9
    toPn = 1e12
    plt.subplot(2,1,1)
    plt.plot(x*toNm,yPure*toPn,color='b',label="Data, No Noise")
    plt.xlabel("Distance (nm)")
    plt.ylabel("Force (pN)")
    plt.legend(loc='upper left')
    plt.title("Extensible WLC better approximates DNA data")
    plt.subplot(2,1,2)
    plt.plot(x*toNm,y*toPn,label="Data,With Noise")
    plt.plot(x*toNm,mFit*toPn,'r-',label="Extensible",linewidth=2.0)
    plt.plot(x*toNm,mFitNon*toPn,'b--',label="Non Extensible")
    plt.ylim([min(y*toPn),max(y*toPn)])
    plt.xlabel("Distance (nm)")
    plt.ylabel("Force (pN)")
    plt.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    run()

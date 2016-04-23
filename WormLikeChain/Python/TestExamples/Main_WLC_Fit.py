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
from scipy.interpolate import interp1d

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
    x = np.arange(0,1375,1) * 1e-9
    # write down their parameter values, figure 1 inset
    kbT = 4.11e-21
    L0 = 1317.52e-9
    Lp =  40.6e-9
    K0 = 1318e-12
    # for the non-extensible model, really only want to fit up to
    # some high percentage of the contour length
    offset = x[0]
    n = x.size
    if (max(x) > L0):
        maxIdx = np.argmin(np.abs(L0*0.95-x))
        sliceV = slice(0,maxIdx,1)
    else:
        sliceV = slice(0,n,1)
    xToFit= x[sliceV]
    yPartial = WLC_Fit.WlcNonExtensible(xToFit,kbT,Lp,L0)
    # extrapolate the y back
    f = interp1d(xToFit,yPartial,kind='linear',bounds_error=False,
                 fill_value='extrapolate')
    nIters = 500
    for i in range(nIters):
        sliceV = slice(0,n,1)
        xToFit = x[sliceV]
        y = f(xToFit)
        prev = y.copy()
        plt.plot(yPartial)
        plt.plot(y,'r--')
        plt.show()
        y = WLC_Fit.WlcExtensible(xToFit,kbT,Lp,L0,K0,prev)
    plt.plot(x*1e9,y*1e12)
    plt.show()
    return x,y

def run():
    """
    Runs some unit testing on the WLC fitting
    """

    x,y = GetBouichatData()
    xInterp = x
    yInterp = y
    noiseAmplitude = 1e-12
    yInterp += noiseAmplitude * (np.random.rand(*yInterp.shape)-0.5)
    # get an extensible and non-extensible model, choose whether to varying L0
    # and Lp
    extensibleFit = WLC_Fit.ExtensibleWlcFit(xInterp,yInterp,VaryL0=True,
                                             VaryLp=False,VaryK0=False)
    nonExtensibleFit = WLC_Fit.NonExtensibleWlcFit(xInterp,yInterp,VaryL0=True,
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
    plt.plot(xInterp*toNm,yInterp*toPn,label="Data")
    plt.plot(xInterp*toNm,mFit*toPn,'r-',label="Extensible",linewidth=3.0)
    plt.plot(xInterp*toNm,mFitNon*toPn,'b--',label="Non Extensible")
    plt.ylim([min(yInterp*toPn),max(yInterp*toPn)])
    plt.xlabel("Distance (nm)")
    plt.ylabel("Force (pN)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run()

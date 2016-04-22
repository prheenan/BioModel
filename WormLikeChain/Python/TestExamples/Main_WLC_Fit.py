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

def run():
    """
    Runs some unit testing on the WLC fitting
    """
    # set up the maximum x (distance) and y (force), in SI
    maxX = 601e-9 
    yUnits = 1e-12
    # add noise
    noiseAmplitude = yUnits*30
    x = np.linspace(0,maxX,15)
    xInterp = np.linspace(0,maxX,100)
    y = np.array([0,0,0,0,0,0.5,1,1.5,3,4,9,30,50,100,250]) * yUnits
    yInterp = np.interp(xInterp,x,y)
    yInterp += noiseAmplitude * (np.random.rand(*yInterp.shape)-0.5)
    # get an extensible and non-extensible model, choose whether to varying L0
    # and Lp
    extensibleFit = WLC_Fit.ExtensibleWlcFit(xInterp,yInterp,VaryL0=True,
                                             VaryLp=True,VaryK0=False)
    print(extensibleFit)
    nonExtensibleFit = WLC_Fit.NonExtensibleWlcFit(xInterp,yInterp,VaryL0=True,
                                                   VaryLp=False)
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

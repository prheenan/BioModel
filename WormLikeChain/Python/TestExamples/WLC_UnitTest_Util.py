# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

import Code.WLC_Fit as WLC_Fit


def PlotWLCFit(DataObj,fit):
    """
    Makes a fairly simple plot of the given Data and WLC fit

    Args:
         DataObj: The WLC_UnitTest_Data.ModelData of the data
         fit : the return from a fitting function; expecting 
         WLC_Fit.FitReturnInfo obj
    Returns:
         Figure reference
    """
    ylim = DataObj.ylim
    mFit = fit.Prediction
    name = DataObj.name
    x = DataObj.ext
    y = DataObj.ForceWithNoise
    name = DataObj.name
    yPure = DataObj.force
    # plot everything
    toNm = 1e9
    toPn = 1e12
    fig = plt.figure()
    ylimPn = lambda: plt.ylim(toPn * ylim)
    plt.subplot(2,1,1)
    plt.plot(x*toNm,yPure*toPn,color='b',
             label="Data, {:s} (No Noise)".format(name))
    plt.xlabel("Distance (nm)")
    plt.ylabel("Force (pN)")
    ylimPn()
    plt.legend(loc='upper left')
    plt.title("Extensible WLC better approximates FEC at high force")
    plt.subplot(2,1,2)
    plt.plot(x*toNm,y*toPn,color='k',alpha=0.3,label="Data,With Noise")
    plt.plot(x*toNm,mFit*toPn,'r-',label="Extensible",linewidth=1.5)
    ylimPn()
    plt.xlabel("Distance (nm)")
    plt.ylabel("Force (pN)")
    plt.legend(loc='upper left')
    plt.tight_layout()
    return fig

def CheckDataObj(DataObj,OutName=None):
    """
    Given a ModelData object, checks its parameters match what we would
    expect, then plots / shows, depending on OutName

    Args:
        DataObj : instance of ModelData, keeping all the data we want
        OutName : If not None, saves a plot.
    """
    x = DataObj.ext
    y = DataObj.force
    params = DataObj.params
    y = DataObj.ForceWithNoise
    print("Fitting Data From {:s}...".format(DataObj.name))
    # get an extensible and non-extensible model, choose whether to varying L0
    # and Lp
    extensibleFit = WLC_Fit.ExtensibleWlcFit(x,y,VaryL0=True,
                                             VaryLp=True,VaryK0=False)
    # make sure the parameters match what the model says it should 
    assert extensibleFit.Info.ParamVals.CloseTo(params)
    nonExtensibleFit = WLC_Fit.NonExtensibleWlcFit(x,y,VaryL0=True,
                                                   VaryLp=False)
    print("Extensible Parameters")
    print(extensibleFit)
    print("Non-extensible Parameters")
    print(nonExtensibleFit)
    mFitNon = nonExtensibleFit.Prediction
    if (OutName is not None):
        toPn = 1e12
        toNm = 1e9
        fig = PlotWLCFit(DataObj,extensibleFit)
        # add the extensible model to the end
        plt.plot(x*toNm,mFitNon*toPn,'b--',label="Non Extensible")
        fig.savefig(OutName + ".png")

def TestDataWithSteps(Steps,DataFunction):
    for step in Steps:
        DataObj = DataFunction(step)
        CheckDataObj(DataObj,OutName=DataObj.name + \
                     "DeltaX={:.2f}".format(step))
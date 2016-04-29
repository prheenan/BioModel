# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

import Code.WLC_Fit as WLC_Fit
from WLC_UnitTest_Data import GetBouichatData

def GetSampleForceExtension(**kwargs):
    """
    Returns Extension,Force for an example data set, for testing

    Returns:
        Extension,Force (both as arrays), then the actual data object
    """
    # get the (example) data
    Data = GetBouichatData(**kwargs)
    # get the force and extension (both in SI units, Newtons and meters)
    # these are just arrays.
    Force = Data.ForceWithNoise
    Extension = Data.ext
    return Extension,Force,Data

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
    params = DataObj.params
    paramDict= dict((k,v.Value) for k,v in params.GetParamDict().items())
    # fit to the noisy data
    y = DataObj.ForceWithNoise
    print("Fitting Data From {:s}...".format(DataObj.name))
    # get an extensible and non-extensible model, choose whether to varying L0
    # and Lp
    extensibleFit = WLC_Fit.ExtensibleWlcFit(x,y,VaryL0=True,
                                             VaryLp=True,VaryK0=False,
                                             Values=paramDict)
    print("Extensible Parameters")
    print(extensibleFit)
    # make sure the parameters match what the model says it should
    ParamVals  = extensibleFit.Info.ParamVals
    assert ParamVals.CloseTo(params)
    # try to get the non-extensible fit (possible it fails)
    try:
        nonExtensibleFit = WLC_Fit.NonExtensibleWlcFit(x,y,VaryL0=True,
                                                       VaryLp=False,
                                                       Values=paramDict)
        # print off the results
        print("Non-extensible Parameters")
        print(nonExtensibleFit)
    except ValueError:
        # non-extensible model wouldnt cut it
        nonExtensibleFit = None
    # make a plot, if e want one
    if (OutName is not None):
        toPn = 1e12
        toNm = 1e9
        fig = PlotWLCFit(DataObj,extensibleFit)
        # add the non-extensible model to the end
        if (nonExtensibleFit is not None):
            mFitNon = nonExtensibleFit.Prediction
            plt.plot(x*toNm,mFitNon*toPn,'b--',label="Non Extensible")
            plt.legend(loc='upper left')
        fig.savefig(OutName + ".png")

def TestDataWithSteps(Steps,DataFunction):
    for i,step in enumerate(Steps):
        DataObj = DataFunction(step)
        CheckDataObj(DataObj,OutName=DataObj.name + \
                     "_{:d}_DeltaX={:.4g}nm".format(i,step))

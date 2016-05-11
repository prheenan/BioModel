# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys


sys.path.append("../../../../../")
from EnergyLandscapes.Lifetime_Dudko2008.Python.TestExamples.Util import \
    Example_Data
from EnergyLandscapes.Lifetime_Dudko2008.Python.Code.Dudko2008Lifetime import \
    DudkoFit,DudkoModel

from EnergyLandscapes.Lifetime_Dudko2008.Python.Code.Dudko_Helper import \
    GetTimeIntegral

def run():
    """

    """
    data = Example_Data.Dudko2008Fig1_Probabilities()
    edges = data.Edges
    probArr = data.Probabilities
    maxProb = np.max(np.concatenate(probArr))
    n = len(probArr)
    # try to make  the lifetime
    styles = [dict(color='b',marker='s'),
              dict(color='g',marker='v'),
              dict(color='y',marker='d'),
              dict(color='r',marker='*')]
    times = []
    voltages = []
    # XXX TODO: break up rupture forces etc. 
    for j in range(n):
        mTmp = probArr[j]
        idxWhere = np.where(mTmp > 0)
        mTmp = mTmp[idxWhere]
        mForces = edges[idxWhere]
        tmpLoad = data.LoadingRates[j]
        loads = np.ones(mForces.size) * tmpLoad
        lifetimes = GetTimeIntegral(mTmp,mForces,loads)
        plt.semilogy(mForces,lifetimes,'bs',markersize=9,**styles[j])
        plt.ylabel("Unzipping time, T(V)[s]")
        plt.xlabel("Voltage, V [mV]")
        # XXX last is 0?
        times.extend(lifetimes[:-1])
        voltages.extend(mForces[:-1])
    times = np.array(times)
    voltages = np.array(voltages)
    idxSort = np.argsort(voltages)
    Values = data.Params
    x = np.linspace(0,200)
    y = DudkoModel(x,**Values)
    plt.plot(x,y)
    plt.ylim([1e-4,25])
    plt.xlim([0,205])
    fit = DudkoFit(voltages[idxSort],times[idxSort],Values=Values)
    yFit = DudkoModel(x,**fit.Info.ParamVals.GetValueDict())
    params = fit.Info.ParamVals.GetValueDict()
    data.Validate(fit.Prediction,params)
    plt.plot(x,yFit,'r--')
    plt.show()
    nFit = len(voltages) * 50
    xArr = np.linspace(min(voltages)*0.9,max(voltages)*1.1,nFit)
    predicted = fit.Predict(xArr)
    for i,prob in enumerate(probArr):
        mStyle = styles[i]
        plt.subplot(n/2,n/2,(i+1))
        plt.bar(edges,prob,width=10,linewidth=0,color=mStyle['color'])
        plt.ylim([0,maxProb*1.05])
        if (i == 2):
            plt.xlabel("unzipping voltage [mV]")
            plt.ylabel("probability")
    plt.show()


if __name__ == "__main__":
    run()

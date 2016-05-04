# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys


sys.path.append("../../../../../")
from EnergyLandscapes.Lifetime_Dudko2008.Python.TestExamples.Util import \
    Example_Data

def GetTimeIntegral(probabilities,forces,loads):
    """
    Getting the lifetimes by equation 2, from Dudko2008. I do *not* use
    the method they suggest (10), since we can do the numerical integral a 
    little better by using trapezoids

    Args:
        probabilities: array of probabilites; element [i] is probablity 
        to rupture at force forces[i]. y axis of (e.g) Dudko2008 Figure 1a

        forces: array of rupture 'forces' or similiar (e.g. x axis of ibid
        is voltage). Element [i] corresponds to rupture associated with probs[i]

        Load: array of 'loading rate' (units of forces/time), e.g. label
        of ibid 
    """
    lifetimes = [np.trapz(y=probabilities[i:]/(probabilities[i]*loads[i]),
                          x=forces[i:])
                 for i in range(probabilities.size)]
    return lifetimes

def run():
    """

    """
    data = Example_Data.Dudko2008Fig1_Probabilities()
    edges = data.Edges
    probArr = data.Probabilities
    maxProb = np.max(np.concatenate(probArr))
    n = len(probArr)
    # try to make  the lifetime
    j =-1
    mTmp = probArr[j]
    idxWhere = np.where(mTmp > 0)
    mTmp = mTmp[idxWhere]
    mForces = edges[idxWhere]
    tmpLoad = data.LoadingRates[j]
    print(tmpLoad)
    loads = np.ones(mForces.size) * tmpLoad
    lifetimes = GetTimeIntegral(mTmp,mForces,loads)
    plt.semilogy(mForces,lifetimes,'bs')
    plt.ylabel("Unzipping time, T(V)[s]")
    plt.xlabel("Voltage, V [mV]")
    plt.show()
    styles = [dict(color='b'),
              dict(color='g'),
              dict(color='y'),
              dict(color='r')]
    for i,prob in enumerate(probArr):
        mStyle = styles[i]
        plt.subplot(n/2,n/2,(i+1))
        plt.bar(edges,prob,width=10,linewidth=0,**mStyle)
        plt.ylim([0,maxProb*1.05])
        if (i == 2):
            plt.xlabel("unzipping voltage [mV]")
            plt.ylabel("probability")
    plt.show()


if __name__ == "__main__":
    run()

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
    styles = [dict(color='b'),
              dict(color='g'),
              dict(color='y'),
              dict(color='r')]
    times = []
    voltages = []
    for j in range(n):
        mTmp = probArr[j]
        idxWhere = np.where(mTmp > 0)
        mTmp = mTmp[idxWhere]
        mForces = edges[idxWhere]
        tmpLoad = data.LoadingRates[j]
        loads = np.ones(mForces.size) * tmpLoad
        lifetimes = GetTimeIntegral(mTmp,mForces,loads)
        plt.semilogy(mForces,lifetimes,'bs',**styles[j])
        plt.ylabel("Unzipping time, T(V)[s]")
        plt.xlabel("Voltage, V [mV]")
        # XXX last point has time of 0?
        times.extend(lifetimes[:-1])
        voltages.extend(mForces[:-1])
    times = np.array(times)
    voltages = np.array(voltages)
    idxSort = np.argsort(voltages)
    Values = dict(tau0=1e-3,
                  v=1/2.,
                  x_tx=11e-3,
                  DeltaG_tx=12 * 4.1e-21,
                  kbT=4.1e-21)
    kbT = 4.1e-21
    x = np.linspace(0,200)
    """
    Numbers from 
    Dudko, Olga K., Gerhard Hummer, and Attila Szabo.
    "Theory, Analysis, and Interpretation of Single-Molecule Force 
    Spectroscopy Experiments."
    Proceedings of the National Academy of Sciences 105, no. 41 
    (October 14, 2008)

    General interpretation (ie: conversion from x_tx  to V_tx) is from 4190 
    ("Theory") section of :

    Dudko, Olga K., et al
    "Extracting Kinetics from Single-Molecule Force Spectroscopy:
    Nanopore Unzipping of DNA Hairpins." 
    Biophysical Journal 92 (June 15, 2007)
    """
    y = DudkoModel(x,
                   tau0=14.3,
                   v=1/2,
                   x_tx=kbT/11.1,
                   DeltaG_tx=11.9*kbT,
                   kbT=kbT)
    plt.plot(x,y)
    print(y)
    plt.show()
    exit(1)
    fit = DudkoFit(voltages[idxSort],times[idxSort],Values=Values)
    print(fit.Info)
    nFit = len(voltages) * 50
    xArr = np.linspace(min(voltages)*0.9,max(voltages)*1.1,nFit)
    predicted = fit.Predict(xArr)
    plt.plot(xArr,predicted)
    plt.show()
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

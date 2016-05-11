# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

from EnergyLandscapes.Lifetime_Dudko2008.Python.Code.Dudko_Helper import \
    DudkoParamValues

from EnergyLandscapes.Lifetime_Dudko2008.Python.Code.Dudko_Helper import \
    GetTimeIntegral

from EnergyLandscapes.Lifetime_Dudko2008.Python.Code.Dudko2008Lifetime import \
    DudkoFit,DudkoModel

class PlotOpt:
    def __init__(self,xlabel,ylabel,ylim):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.ylim = ylim

class ExampleData:
    def __init__(self,BinEdges,BinValues,LoadingRates,Params,PlotOpt,rtol=0.1):
        """
        Records all the initial data. 
        """
        self.Probabilities = BinValues
        self.Edges = BinEdges
        self.LoadingRates = LoadingRates
        self.ParamObj = DudkoParamValues(Values=Params)
        self.rtol=rtol
        self.PlotOpt=PlotOpt
    @property
    def Params(self):
        """
        Returns the (expected) parameters as a dictionary of <name:values>
        """
        return self.ParamObj.GetValueDict()
    def Validate(self,PredictedY,PredictedParams):
        """
        Validates the predicted y and parameters. Throws an assertion error
        if it failes

        Args:
            PredictedY: the y values (lifetimes) predicted by the dudko method
            PredictedParams: the predicted parameters, as a dictionary
        """
        for key_exp,val_exp in self.Params.items():
            predicted = PredictedParams[key_exp]
            np.testing.assert_allclose(predicted,val_exp,
                                       atol=0,rtol=self.rtol)
    def GetLifetimesAndVoltages(self):
        """
        Gets the Lifetimes and voltages for each loading rate, assuming the 
        loading rate is constant (See dudko2008)

        Returns:
            tuple of <time,voltages>, where element [i] of time or voltage
            corresponds to LoadingRates[i]
        """
        times = []
        voltages = []
        n = len(self.LoadingRates)
        for j in range(n):
            mTmp = self.Probabilities[j]
            idxWhere = np.where(mTmp > 0)
            mTmp = mTmp[idxWhere]
            mForces = self.Edges[idxWhere]
            tmpLoad = self.LoadingRates[j]
            loads = np.ones(mForces.size) * tmpLoad
            lifetimes = GetTimeIntegral(mTmp,mForces,loads)
            # XXX last is 0?
            times.append(lifetimes[:-1])
            voltages.append(mForces[:-1])
        return times,voltages

def Dudko2008Fig2_Probabilities():
    """
    Function to get the dudko values from Figure 2 (Forces)

    Dudko, Olga K., Gerhard Hummer, and Attila Szabo.
    "Theory, Analysis, and Interpretation of Single-Molecule Force 
    Spectroscopy Experiments."
    Proceedings of the National Academy of Sciences 105, no. 41 
    (October 14, 2008)
    """
    # # Write down figure 2 histograms
    # first: edges are (roughly) 7pN apart, with 20 total (10mV/step)
    edges =np.linspace(0,140,21) 
    # below, I record the (rough) probabilities in each bin from zero.
    # I convert the probabilities to a 'percentage' (e.g. 0.01 -> 1),
    # to make it easier to write out
    # 200nm/s, y values
    fig2a_200nm_s = [0,0,4,16,36,6,34,4]
    # 4.5V/s
    fig2a_400nm_s = [0,0,2,4,20,40,50,38,1]
    # 12 V/s
    fig2a_2000_nm_s  = [0,0,0,2,10,2,24,36,38,12,8]
    # 18 V/s
    fig2a_4000_nm_s  = [0,0,0,1,10,12,24,30,40,30,20,6]
    plotsAndSpeeds = [ [fig2a_200nm_s,200],
                       [fig2a_400nm_s,400],
                       [fig2a_2000nm_s,2000],
                       [fig2a_4000nm_s,4000]]
    allHist,speeds = GetLoadsAndSpeed(edges, plotsAndSpeeds)
    # convert the loading rate by equation [4]
    # XXX ..
    
def GetLoadsAndSpeed(edges, plotsAndSpeeds):
    """
    Given a list of loading rate and count measurements, gets the normalized
    histograms

    Args:
        edge: all of the 'x' values for the histogram (E.g. force or voltage
        bin edges) 

        plotsAndSpeeds: list: each element is a tuple; element 0 of the tuple 
        is a (not necessarily normalized) distribution of rupture probabilities,
        starting from 0, and continuing until 0
    returns:
        tuple of <list of histograms (one per loading rate),
    list of loading rates>
    """
    allHist = []
    speeds = []
    # get the normalized histograms
    for hist,speed in plotsAndSpeeds:
        hist = np.array(hist)
        histFull = np.zeros(edges.size)
        histFull[:hist.size] = hist
        # normalize everything to one
        sumV = sum(histFull)
        histFull  /= sumV
        allHist.append(histFull)
        speeds.append(speed)
    return allHist,speeds
    
def Dudko2008Fig1_Probabilities():
    """
    Function to get the dudko values from Figure 1 (nanopore)

    Dudko, Olga K., Gerhard Hummer, and Attila Szabo.
    "Theory, Analysis, and Interpretation of Single-Molecule Force 
    Spectroscopy Experiments."
    Proceedings of the National Academy of Sciences 105, no. 41 
    (October 14, 2008)

    """
    # # Write down figure 1 histograms
    # first: edges are (roughly) 10mV apart, with 20 total (10mV/step)
    edges =np.linspace(0,200,21) 
    # below, I record the (rough) probabilities in each bin from zero.
    # I convert the probabilities to a 'percentage' (e.g. 0.01 -> 1),
    # to make it easier to write out
    # 0.83V/s, y values
    fig1a_0p83 = [0,0,0,1,1,3,5,8,14,18,24,18,8]
    # 4.5V/s
    fig1a_4p5 = [0,0,0,0,0,0,0,2,4,12,12,17,20,17,12,5,2]
    # 12 V/s
    fig1a_12  = [0,0,0,0,0,0,0,2,2,5,7,8,12,12,16,15,
                 10,8,6,4,4]
    # 18 V/s
    fig1a_18  = [0,0,0,0,0,0,0,1,2,4,6,6,7,12,10,15,14,13,10,8,6]
    # record all the histograms, normalizing appropriately,
    # recording everything in millivolts
    plotsAndSpeeds = [ [fig1a_0p83,0.83e3],
                       [fig1a_4p5,4.5e3],
                       [fig1a_12,12e3],
                       [fig1a_18 ,18e3]]
    allHist,speeds = GetLoadsAndSpeed(edges, plotsAndSpeeds)
    # now get the expected parameters from fitting
    """
    Actual numbers from
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
    kbT = 4.1e-21
    Params = dict(tau0=14.3,
                   v=1/2,
                  # see pp 4190, since F=V (ie: we use voltage analogously)
                  # Beta * x_tx = 1/V_tx
                  # and V_tx = 11.1mV (everything is in mV)
                   x_tx=kbT/11.1,
                   DeltaG_tx=11.9*kbT,
                   kbT=kbT)
    mOpt = PlotOpt(ylabel="Unzipping time, T(V)[s]",
                   xlabel="Voltage (mV)",
                   ylim=[1e-4,25])
    return ExampleData(edges,allHist,speeds,Params,mOpt)


def GetStyle(i):
    """
    Gets the style associated with the Dudko2008 plot

    Args:
        i: which style to get
    Returns: 
        dictionary of style parmaeters
    """
    styles = [dict(color='b',marker='s'),
              dict(color='g',marker='v'),
              dict(color='y',marker='d'),
              dict(color='r',marker='*')]
    n = len(styles)
    return styles[i%n]


def PlotLifetimesAndFit(data):
    """
    Makes a plot of lifetime versus force (or voltage)

    Args:
        data: the data to use, instance of ExampleData
    Returns:
        the figure handle made
    """
    # get the per-loading rate lifetimes
    times,voltages = data.GetLifetimesAndVoltages()
    fig = plt.figure()
    # plot each loading rate 
    for i,(mTimes,mForces) in enumerate(zip(times,voltages)):
        style = GetStyle(i)
        plt.semilogy(mForces,mTimes,linewidth=0,markersize=9,**style)
        plt.ylabel(data.PlotOpt.ylabel)
        plt.xlabel(data.PlotOpt.xlabel)
    # concatenate everything for the fit 
    times = np.concatenate(np.array(times))
    voltages = np.concatenate(np.array(voltages))
    idxSort = np.argsort(voltages)
    Values = data.Params
    maxX = max(voltages)
    x = np.linspace(0,maxX)
    y = DudkoModel(x,**Values)
    # plot the expected values on top
    plt.plot(x,y,label="Dudko Parameters")
    plt.ylim(data.PlotOpt.ylim)
    plt.xlim([0,maxX])
    plt.title("Lifetime versus 'force' not well-modeled by Bell")
    fit = DudkoFit(voltages[idxSort],times[idxSort],Values=Values)
    yFit = DudkoModel(x,**fit.Info.ParamVals.GetValueDict())
    # get the predicted (fitted) parametrs
    params = fit.Info.ParamVals.GetValueDict()
    # validate that the Dudko model gets things correct. 
    data.Validate(fit.Prediction,params)
    plt.plot(x,yFit,'r--',label="Fitted Parameters")
    plt.legend()
    return fig

def PlotHistograms(data):
    """
    Makes a histogram of probability versus force

    Args:
        data: see PlotLifetimesAndFit
    Returns:
        the figure handle made
    """
    edges = data.Edges
    probArr = data.Probabilities
    times,voltages = data.GetLifetimesAndVoltages()
    # concateate all the times and volages
    times = np.concatenate(times)
    voltages = np.concatenate(voltages)
    maxProb = np.max(np.concatenate(probArr))
    n = len(probArr)
    nFit = len(voltages) * 50
    fig = plt.figure()
    for i,prob in enumerate(probArr):
        mStyle = GetStyle(i)
        plt.subplot(n/2,n/2,(i+1))
        plt.bar(edges,prob,width=10,linewidth=0,color=mStyle['color'])
        plt.ylim([0,maxProb*1.05])
        if (i == 2):
            plt.xlabel(data.PlotOpt.xlabel)
            plt.ylabel("probability")
    return fig

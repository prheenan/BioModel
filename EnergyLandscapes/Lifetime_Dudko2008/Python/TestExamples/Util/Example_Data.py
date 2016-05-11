# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

from EnergyLandscapes.Lifetime_Dudko2008.Python.Code.Dudko_Helper import \
    DudkoParamValues

class ExampleData:
    def __init__(self,BinEdges,BinValues,LoadingRates,Params,rtol=0.1):
        """
        Records all the initial data. 
        """
        self.Probabilities = BinValues
        self.Edges = BinEdges
        self.LoadingRates = LoadingRates
        self.ParamObj = DudkoParamValues(Values=Params)
        self.rtol=rtol
    @property
    def Params(self):
        return self.ParamObj.GetValueDict()
    def Validate(self,PredictedY,PredictedParams):
        for key_exp,val_exp in self.Params.items():
            predicted = PredictedParams[key_exp]
            np.testing.assert_allclose(predicted,val_exp,
                                       atol=0,rtol=self.rtol)
            
def Dudko2008Fig1_Probabilities():
    """
    Function to get the dudko values
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
    return ExampleData(edges,allHist,speeds,Params)


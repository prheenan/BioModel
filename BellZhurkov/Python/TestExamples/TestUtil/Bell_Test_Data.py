# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

def Woodside2014FoldingAndUnfoldingData():
    """
    From: 
    Woodside, Michael T., and Steven M. Block. 
"Reconstructing Folding Energy Landscapes by Single-Molecule Force Spectroscopy"
    Annual Review of Biophysics 43, no. 1 (2014): 19-39. 
    doi:10.1146/annurev-biophys-051013-022754.
    """
    # write down figure 6a, pp 28
    Forces = [11.25,
              11.50,
              11.75,
              12.0,
              12.25,
              12.75,
              13.25,
              13.5,
              13.75,
              14.25,
              14.75]
    Forces= np.array(Forces)
    # write down the folding and unfolding rates as decaying/increasing
    # exponentials, based on their starting value in the graph
    ForceDiff = max(Forces)-min(Forces)
    # we have
    # y ~ exp(+/- t/tau)
    # so y_f/y_i = exp(+/- (t_f-t_i)/tau)
    # so tau = +/- np.log(yf/y_i)/(tf-t_i). aparently the signs work themselves
    # out the way I use tau below
    yf = 300
    yi = 8
    # get the 'taus' for the exponential decay (really inverse forces)
    tauFold = np.log(yf/yi)/(ForceDiff)
    tauUnfold = np.log(yf/yi)/(ForceDiff)
    # get the offset force array; use this to set up the data, which we
    # assume decays or increases from our 'zero point' (first measured data)
    forceOffset = (Forces-Forces[0])
    # folding is decaying / decaying / *has* minus sign
    Folding = yf * np.exp( -forceOffset/tauFold )
    # unfolding is increasing / growing / *doesnt have* minus sign
    Unfolding = yi * np.exp( forceOffset/tauUnfold)
    return Forces*1e-12,Folding,Unfolding

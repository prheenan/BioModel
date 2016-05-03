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
    ForceDiff = Forces[-1]-Forces[0]
    FoldSlope = np.log((35-7)/(ForceDiff))
    UnfoldSlope = np.log((35-8)/(ForceDiff))
    Folding = 25 * np.exp( -(Forces-Forces[0])/FoldSlope )
    Unfolding = 8 * np.exp( (Forces-Forces[0])/UnfoldSlope)
    return Forces*1e-12,Folding,Unfolding

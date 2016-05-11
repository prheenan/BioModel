# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys


sys.path.append("../../../../../")
from EnergyLandscapes.Lifetime_Dudko2008.Python.TestExamples.Util import \
    Example_Data

def PlotFit(data,BaseName):
    fig = Example_Data.PlotHistograms(data)
    fig.savefig(BaseName + "_Histogram.png")
    fig = Example_Data.PlotLifetimesAndFit(data)
    fig.savefig(BaseName + "_Lifetimes.png")
    
def run():
    """

    """
    # figure 1 from dudko 2008
    data = Example_Data.Dudko2008Fig1_Probabilities()
    PlotFit(data,"../Out/Dudko2008_Fig1")
    # figure 2 frm dudko 2008
    data = Example_Data.Dudko2008Fig2_Probabilities()
    PlotFit(data,"../Out/Dudko2008_Fig2")




if __name__ == "__main__":
    run()

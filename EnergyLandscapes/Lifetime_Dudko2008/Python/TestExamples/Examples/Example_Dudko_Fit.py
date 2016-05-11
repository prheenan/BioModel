# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys


sys.path.append("../../../../../")
from EnergyLandscapes.Lifetime_Dudko2008.Python.TestExamples.Util import \
    Example_Data



def run():
    """

    """
    data = Example_Data.Dudko2008Fig1_Probabilities()
    fig = Example_Data.PlotLifetimesAndFit(data)
    fig.savefig("../out/Dudko2008Fig1_Lifetimes.png")
    fig = Example_Data.PlotHistograms(data)
    fig.savefig("../out/Dudko2008Fig1_Histogram.png")


if __name__ == "__main__":
    run()

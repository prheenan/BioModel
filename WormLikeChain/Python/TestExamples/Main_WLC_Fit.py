# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

baseDir = "../"
sys.path.append(baseDir)
sys.path.append("../../../")
from Testing import Main_WLC_Testing
from Examples import Main_WLC_Examples

def run():
    """
    Runs some unit testing on the WLC fitting. Some booleans here:

    RunExamples: If true, shows plots of various fits

    RunTests: If true, runs unit testing using known data/parameters,
    saving multiple plots in this directory
    """
    RunExamples = True
    RunTests = True
    if (RunExamples):
        Main_WLC_Examples.run()
    if (RunTests):
        Main_WLC_Testing.run()


if __name__ == "__main__":
    run()

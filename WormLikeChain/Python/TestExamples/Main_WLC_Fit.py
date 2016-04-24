# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

baseDir = "../"
sys.path.append(baseDir)
sys.path.append("../../../")

from WLC_UnitTest_Util import TestDataWithSteps,PlotWLCFit
from WLC_UnitTest_Data import GetBouichatData
import Code.WLC_Fit as WLC_Fit


def RunBouchiatDataTests():
    """
    Runs the unit tests for Bouchiat, and generates the plots
    """
    # really, the only thing we have control over is how much we interpolate
    # over the given literature values
    StepNm = [ 0.05,0.1,0.5,1,2]
    toTest =  [ [StepNm,GetBouichatData]]
    for Steps,Function in toTest:
        TestDataWithSteps(Steps,Function)

def RunWLCExample():
    """
    Simple example of how the WLC stuff works. Start here to get a feel
    for how this works!
    """
    # get the (example) data
    Data = GetBouichatData()
    # get the force and extension (both in SI units, Newtons and meters)
    # these are just arrays.
    Force = Data.ForceWithNoise
    Extension = Data.ext
    # assume we want to just get the contour length,
    # everything else is fixed. Set up initial guesses for the params
    # (see 'WLC_UnitTest_Data.GetBouichatData' for where these are coming from)
    InitialGuesses = dict(kbT=4.11e-21, # pN * nm (Room Temp)
                          L0 =1310e-09, # nm, just a guess, we fit this
                          Lp =40.6e-09, # nm  , again 
                          K0 =1318e-12) # pN
    extensibleFit = WLC_Fit.ExtensibleWlcFit(Extension,Force,VaryL0=True,
                                             VaryLp=False,VaryK0=False,
                                             **InitialGuesses)
    # what we have is the fit object; we can get/print the parameters
    PlotWLCFit(Data,extensibleFit)
    plt.show()

def run():
    """
    Runs some unit testing on the WLC fitting
    """
    RunExample = True
    RunTests = True
    if (RunExample):
        RunWLCExample()
        exit(1)
    if (RunTests):
        RunBouchiatDataTests()

if __name__ == "__main__":
    run()

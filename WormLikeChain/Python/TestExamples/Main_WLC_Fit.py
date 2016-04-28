# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

baseDir = "../"
sys.path.append(baseDir)
sys.path.append("../../../")

from WLC_UnitTest_Util import TestDataWithSteps,PlotWLCFit,\
    GetSampleForceExtension
from WLC_UnitTest_Data import GetBouichatData
import Code.WLC_Fit as WLC_Fit


def RunBouchiatDataTests():
    """
    Runs the unit tests for Bouchiat, and generates the plots
    """
    # really, the only thing we have control over is how much we interpolate
    # over the given literature values
    # note we reverse it so we 'fail fast'
    StepNm = [ 0.005,0.01,0.05,0.1,0.25,0.5,0.75,1,2,5,10][::-1]
    toTest =  [ [StepNm,GetBouichatData]]
    for Steps,Function in toTest:
        TestDataWithSteps(Steps,Function)

def RunWLCExample():
    """
    Simple example of how the WLC stuff works. Start here to get a feel
    for how this works!
    """
    # get the (sample) extension and force
    Extension,Force,Data = GetSampleForceExtension(StepInNm=5)
    ## for this example, everything besides contour length is fixed.
    #Set up initial guesses for the params
    # (see 'WLC_UnitTest_Data.GetBouichatData' for where these are coming from)
    InitialGuesses = dict(kbT=4.11e-21, # pN * nm (Room Temp)
                          L0 =1310e-09, # nm
                          Lp =40.6e-09, # nm  , again 
                          K0 =1318e-12) # pN
    extensibleFit = WLC_Fit.ExtensibleWlcFit(Extension,Force,VaryL0=True,
                                             VaryLp=True,VaryK0=False,
                                             Values=InitialGuesses)
    # what we have is the fit object; we can get/print the parameters
    PlotWLCFit(Data,extensibleFit)
    plt.show()

def BoundedWLCExample():
    """
    Suppose you dont have a good idea for the exact parameters of your model.
    The Bounded WLC, given rough ranges, can automatically figure it out
    for you. 

    Even the rough ranges can be inferred from your data. That is what is 
    happening here (see Code/WLC_Fit.BoundedWlcFit and 
    Code/WLC_HelperClass.GetReasonableBounds for details). 
    """
    # get the (sample) extension and force
    Extension,Force,Data = GetSampleForceExtension()
    extensibleFit = WLC_Fit.BoundedWlcFit(Extension,Force,VaryL0=True,
                                          VaryLp=True)
    # what we have is the fit object; we can get/print the parameters
    PlotWLCFit(Data,extensibleFit)
    plt.show()

    

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
        RunWLCExample()
        BoundedWLCExample()
    if (RunTests):
        RunBouchiatDataTests()

if __name__ == "__main__":
    run()

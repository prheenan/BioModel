# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

baseDir = "../../"
sys.path.append(baseDir)
sys.path.append("../../../../../")

from TestExamples.Util.WLC_UnitTest_Util import TestDataWithSteps,PlotWLCFit,\
    GetSampleForceExtension
from TestExamples.Util.WLC_UnitTest_Data import GetBouichatData,GetBullData
import Code.WLC_Fit as WLC_Fit
# for basin hopping, need to initialize to HOP
from FitUtil.FitUtils.Python.FitClasses import Initialization
import Code.WLC_ComplexValued_Fit as WLC_ComplexValued_Fit


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

def RunInvertedWLCExample():
    # get the (sample) extension and force
    Extension,Force,Data = GetSampleForceExtension(StepInNm=5)
    ## for this example, everything besides contour length is fixed.
    #Set up initial guesses for the params
    # (see 'WLC_UnitTest_Data.GetBouichatData' for where these are coming from)
    InitialGuesses = dict(kbT=4.11e-21, # pN * nm (Room Temp)
                          L0 =1310e-09, # nm
                          Lp =40.6e-09, # nm  , again 
                          K0 =1318e-12) # pN
    extensibleFit = WLC_ComplexValued_Fit.InvertedWlcForce(Extension,
                                                           F=Force,
                                                           **InitialGuesses)
    plt.plot(Extension,extensibleFit,linestyle='--')
    plt.plot(Extension,Force,'r.')
    plt.show()


    
def BoundedWLCExample():
    """
    Suppose you dont have a good idea for the exact parameters of your model.
    The Bounded WLC, given rough ranges, can automatically figure it out
    for you. 

    Even the rough ranges can be inferred from your data, *assuming
    you only have a single WLC, and you have almost or just above a full
    contour length. That is what is happening here 
    (see Code/WLC_Fit.BoundedWlcFit and 
    Code/WLC_HelperClass.GetReasonableBounds for details). 
    """
    # get the (sample) extension and force
    Extension,Force,Data = GetSampleForceExtension()
    extensibleFit = WLC_Fit.BoundedWlcFit(Extension,Force,VaryL0=True,
                                          VaryLp=True,Ns=20)
    # what we have is the fit object; we can get/print the parameters
    PlotWLCFit(Data,extensibleFit)
    plt.show()

def RunBasinHoping():
    """
    Runs the basin hoping routine, which is randomized and
    probably slower than BoundedWLC, but uses a simulated-annealing like 
    method forimitialization
    """
    # get the (sample) extension and force
    Extension,Force,Data = GetSampleForceExtension()
    InitialGuesses = Data.params.GetValueDict()
    # make an initialization object for the basin hopping; set disp to false
    # to prevent command-line spam
    InitialObj = Initialization(Type=Initialization.HOP,disp=False)
    extensibleFit = WLC_Fit.ExtensibleWlcFit(Extension,Force,VaryL0=True,
                                             Values=InitialGuesses,
                                             InitialObj=InitialObj)
    # what we have is the fit object; we can get/print the parameters
    PlotWLCFit(Data,extensibleFit)
    plt.show()
    

def run():
    """
    Runs some examples on the WLC fitting
    """
    RunInvertedWLCExample()
    RunWLCExample()
    BoundedWLCExample()
    RunBasinHoping()

if __name__ == "__main__":
    run()

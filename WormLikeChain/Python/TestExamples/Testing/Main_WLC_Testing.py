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

def BouchiatDataTests():
    """
    Returns the parameters for the Bouchiat Data Tests
    """
    StepNm = [0.005,0.01,0.05,0.1,0.25,0.5,0.75,1,2,5][::-1]
    toTest =  GetBouichatData
    return StepNm,toTest

def BullDataTests():
    """
    Returns the parameters for the Bull Data Test
    """
    StepNm = [0.0001,0.0005,0.001,0.005,0.025,0.01,0.025,0.05,0.1][::-1]
    toTest =  GetBullData
    return StepNm,toTest
 
def RunDataTests():
    """
    Runs the unit tests, and generates the plots
    """
    # really, the only thing we have control over is how much we interpolate
    # over the given literature values
    # note we reverse it so we 'fail fast'
    IndiviudalFuncs = [BouchiatDataTests,BullDataTests]
    StepsArray = []
    FunctionsArray = []
    for testFunc in IndiviudalFuncs:
        tmpStep,tmpFunc = testFunc()
        StepsArray.append(tmpStep)
        FunctionsArray.append(tmpFunc)
    for Steps,Function in zip(StepsArray,FunctionsArray):
        TestDataWithSteps(Steps,Function)

def run():
    """
    Runs some unit testing on the WLC fitting. 
    """
    RunDataTests()


if __name__ == "__main__":
    run()

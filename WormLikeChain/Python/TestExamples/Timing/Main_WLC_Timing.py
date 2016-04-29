# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

baseDir = "../../"
sys.path.append(baseDir)
sys.path.append("../../../../")

from TestExamples.Util.WLC_UnitTest_Util import TestDataWithSteps,PlotWLCFit,\
    GetSampleForceExtension
from TestExamples.Util.WLC_UnitTest_Data import GetBullData,GetBouichatData
import Code.WLC_Fit as WLC_Fit

from timeit import Timer

def GetSteps(rangeV):
    numPoints = np.array([1.5e3,2e3,5e3,1e4,2e4,5e4,7e4,8e4,1e5,1.5e5,2e5,3.5e5,
                          5e5,1e6])
    steps = rangeV/numPoints
    return steps

def BouchiatDataTests():
    """
    Returns the parameters for the Bouchiat Data Tests
    """
    rangeV = 1314
    toTest =  GetBouichatData
    return GetSteps(rangeV),toTest

def BullDataTests():
    """
    Returns the parameters for the Bull Data Test
    """
    rangeV = 17
    toTest =  GetBullData
    return GetSteps(rangeV),toTest

def TimeFitting(DataObj):
    ParamVals = DataObj.params.GetValueDict()
    x = DataObj.ext
    t = Timer(lambda: WLC_Fit.WlcExtensible(x,**ParamVals))
    return t.timeit(number=3)
 
def RunTimingTests(limit):
    """
    Runs the unit tests, and generates the plots
    """
    # really, the only thing we have control over is how much we interpolate
    # over the given literature values
    # note we reverse it so we 'fail fast'
    IndiviudalFuncs = [BullDataTests,BouchiatDataTests]
    styles = [
        dict(label="Protein Input",color='r',marker='x'),
        dict(label="DNA Input",color='b',marker='o')]
    StepsArray = []
    FunctionsArray = []
    for testFunc in IndiviudalFuncs:
        tmpStep,tmpFunc = testFunc()
        StepsArray.append(tmpStep)
        FunctionsArray.append(tmpFunc)
    InputSize = []
    TimeSize = []
    for Steps,Function in zip(StepsArray,FunctionsArray):
        tmpTime = []
        tmpSize = []
        for i,s in enumerate(Steps):
            if (i == limit):
                break
            obj = Function(s)
            time = TimeFitting(obj)
            tmpSize.append(obj.n)
            tmpTime.append(time)
        InputSize.append(tmpSize)
        TimeSize.append(tmpTime)
    # make a graph of the timing results
    sortIdx = np.argsort(InputSize)
    # array for interpolating the bounds
    flatX = np.ravel(InputSize) / 1000
    interpX = np.linspace(min(flatX),max(flatX),flatX.size*50)
    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    for listSize,listTime,sty in zip(InputSize,TimeSize,styles):
        n = np.array(listSize)/1000
        t = np.array(listTime)*1000
        plt.plot(n,t,**sty)
    plt.plot(interpX,np.log(interpX)*interpX*4,'b--',label="Upper, N * log(N)")
    plt.plot(interpX,np.log(interpX)*interpX/3,'g-',label="Lower, N * log(N)")
    plt.title("Extensible WLC Runtime is well-bounded by N*Log(N)")
    plt.xlabel("Input Size (Thousands of Points)")
    plt.ylabel("RunTime (ms)")
    plt.legend(loc='upper left')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig("Timing.png")

def run():
    """
    Runs some unit testing on the WLC timing
    """
    limit = None
    RunTimingTests(limit)


if __name__ == "__main__":
    run()

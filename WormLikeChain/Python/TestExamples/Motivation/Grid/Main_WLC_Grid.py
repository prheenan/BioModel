# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

baseDir = "../../../"
sys.path.append(baseDir)
sys.path.append("../../../../../")

from TestExamples.Util.WLC_UnitTest_Util import TestDataWithSteps,PlotWLCFit,\
    GetSampleForceExtension
import Code.WLC_Fit as WLC_Fit
import Code.WLC_HelperClasses as WLC_Helper
from scipy.optimize import brute


def RunGridAnalysis():
    """
    Simple example of how the WLC stuff works. Start here to get a feel
    for how this works!
    """
    # get the (sample) extension and force
    Extension,Force,Data = GetSampleForceExtension(StepInNm=5)
    # set up the model object (essentially doing a lot of the work WLC_FIT
    # does... functionalize?)
    Model =  WLC_Helper.WLC_MODELS.EXTENSIBLE_WANG_1997
    bounds = WLC_Helper.GetReasonableBounds(Extension,Force)
    Vals = WLC_Helper.WlcParamValues(Values=Data.params.GetValueDict(),
                                     Bounds=bounds)
    toVary = WLC_Helper.WlcParamsToVary(VaryLp=True,VaryK0=False,VaryL0=True)
    inf = WLC_Helper.WlcFitInfo(Model=Model,ParamVals=Vals,VaryObj=toVary)
    fixed =  inf.GetFixedParamDict()
    vary = inf.GetVaryingParamDict()
    varyNames = [k for k,_ in vary.items()]
    mFunc = WLC_Helper.GetFunctionCall(Model,varyNames,fixed)
    toMin = WLC_Fit.GetMinimizingFunction(Extension,Force,mFunc)
    # run the brute routine wih the specified bounds
    # for all the grid points, record the error
    boundsRaw = inf.DictToValues(inf.GetVaryingBoundsDict())
    boundsBasin = [WLC_Helper.BoundsObj.ToMinimizeConventions(*b)
                   for b in boundsRaw]
    x0,fval,grid,jout = brute(toMin,ranges=boundsBasin,disp=False,
                              finish=None,full_output=True,Ns=10)
    print(jout)
    plt.show()

def run():
    """
    Runs some examples of the grid-search analysis, and why it is important
    """
    RunGridAnalysis()

if __name__ == "__main__":
    run()

# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../../../../")
from FreelyJointedChain.Python.TestExamples.Util.ExampleData import \
    Smith1996_Figure6

from FreelyJointedChain.Python.Code.FJC import FreelyJointedChainFit


def PlotSmith1996():
    Data =Smith1996_Figure6()
    # fit the data...
    Extension = Data.Extension
    Values = Data.Params.GetValueDict()
    noiseForce = Data.ForceWithNoise
    fit = FreelyJointedChainFit(Extension,noiseForce,Values=Values)
    # now fit on a (relatively) uniform grid from low forst to high force
    # use a 10x linear grid
    n = noiseForce.size*10
    forceGrid = np.linspace(noiseForce[0],noiseForce[-1],n)
    # get the predictions
    extGrid = fit.Predict(forceGrid)
    toMicrons = lambda x : x * 1e6
    toPn = lambda x: x * 1e12
    ext = toMicrons(Extension)
    noiseForcePn = toPn(noiseForce)
    ylim = lambda :  plt.ylim(toPn(Data.ylim))
    # plot the model and fit
    plt.subplot(2,1,1)
    plt.plot(ext,toPn(Data.Force))
    plt.ylabel("Force (pN)")
    ylim()
    plt.subplot(2,1,2)
    plt.plot(ext,noiseForcePn,'k',alpha=0.3)
    plt.plot(toMicrons(extGrid),toPn(forceGrid),'b--',linewidth=2)
    plt.xlabel("Extension (microns)")
    plt.ylabel("Force (pN)")
    ylim()
    plt.show()


def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    PlotSmith1996()
    

if __name__ == "__main__":
    run()

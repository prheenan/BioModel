# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

baseDir = "../../../"
sys.path.append(baseDir)
sys.path.append("../../../../../../")
from scipy import optimize

from TestExamples.Util.WLC_UnitTest_Util import TestDataWithSteps,PlotWLCFit,\
    GetSampleForceExtension
import Code.WLC_Fit as WLC_Fit
import Code.WLC_HelperClasses as WLC_Helper
from scipy.optimize import brute

# 3D plotting
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def PlotGrid(gridInfo,thresh,outName):
    """
    Given the gridInfo (essentially, what brute returns after initiialization),
    plots out the grid, saving to outName

    Args:
        gridInfo: FitInto.Info.Initialization.InitializationInfo
        thresh: the maximum error we want to plot
        outName: what to save as
    """
    # get the grid details
    minPoint = gridInfo['x0']
    minVal = gridInfo['fval']
    grid = gridInfo['grid']
    X = grid[0]
    Y = grid[1]
    # only loo at absolute errors less than the threshhold
    Z = np.abs(gridInfo['jout'])
    safeIdx = np.where(Z < thresh)
    # plot the log10 of the erro
    normZ = lambda x : np.log10(x)
    toPlot = lambda x: np.ravel(x[safeIdx])
    # flatten everything to plot as tri-surfaces
    xTri = toPlot(X)
    yTri = toPlot(Y)
    zTri = normZ(toPlot(Z))
    fig = plt.figure(dpi=400)
    ax = fig.gca(projection='3d')
    surf = ax.plot_trisurf(xTri,yTri,zTri,cmap=cm.coolwarm,
                           linewidth=0, antialiased=False,
                           label="WLC Error Landscape")
    minStr = ["{:.3g}".format(o) for o in minPoint]
    ax.plot(*minPoint,zs=[normZ(minVal)],marker='o',color='r',linestyle="None",
            label="Optimal [L0,Lp]: {:s}".format(minStr))
    plt.xlabel("L0 (Relative)")
    plt.ylabel("Lp (Relative)")
    ax.set_zlabel("Log_10 Error")
    plt.legend()
    plt.tight_layout()
    fig.savefig(outName)


def RunGridAnalysis():
    """
    Runs the grid stepping analysis, using the brute force method to get
    a good initial guess
    """
    # get the (sample) extension and force
    Extension,Force,Data = GetSampleForceExtension(StepInNm=1)
    Noisey = Data.ForceWithNoise
    GridSize =100
    fitInfo = WLC_Fit.BoundedWlcFit(Extension,Noisey,VaryL0=True,VaryLp=True,
                                    Ns=GridSize,finish=optimize.fmin)
    thresh =20
    gridInfo = fitInfo.Info.Initialization.InitializationInfo
    PlotGrid(gridInfo,thresh,"Coarse")

    

def run():
    """
    Runs some examples of the grid-search analysis, and why it is important
    """
    RunGridAnalysis()

if __name__ == "__main__":
    run()

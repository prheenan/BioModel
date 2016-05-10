# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../../../../")
import BellZhurkov.Python.TestExamples.TestUtil.Bell_Test_Data as Data
import BellZhurkov.Python.Code.BellZhurkov as BellModel

def RunWoodsideFigure6():
    """
    Reproduces  Figure 6 From: 
    Woodside, Michael T., and Steven M. Block. 
"Reconstructing Folding Energy Landscapes by Single-Molecule Force Spectroscopy"
    Annual Review of Biophysics 43, no. 1 (2014): 19-39. 
    doi:10.1146/annurev-biophys-051013-022754.

    See TestExamples.TestUtil.Bell_Test_Data.Woodside2014FoldingAndUnfoldingData
    """
    Forces,Folding,Unfolding = Data.Woodside2014FoldingAndUnfoldingData()
    # everything in CGS initially
    vary = dict(beta=False,
                k0=False,
                DeltaG=True,
                DeltaX=True)
    GuessDict = dict(beta=1/(4.1e-21),
                     k0=1,
                     DeltaX=20e-9,
                     DeltaG=0)
    opt = dict(Values=GuessDict,
               Vary=vary)
    infFold = BellModel.BellZurkovFit(Forces,Folding,**opt)
    infUnfold = BellModel.BellZurkovFit(Forces,Unfolding,**opt)
    # get predictions along a (slightly larger) x range
    xMin=11e-12
    xMax=15e-12
    # how much should we interpolate?
    numPredict = (len(Forces)+1)*50
    xRangePredict = np.linspace(xMin,xMax,numPredict)
    predictFold = infFold.Predict(xRangePredict)
    predictUnfold = infUnfold.Predict(xRangePredict)
    markerDict = dict(marker='o',
                      markersize=7,
                      linewidth=0,
                      markeredgewidth=0.0)
    lineDict = dict(linestyle='-',color='k',linewidth=1.5)
    toPn = 1e12
    ForcePn = Forces*toPn
    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    plt.plot(ForcePn,Folding,'ro',label="Folding",**markerDict)
    plt.plot(xRangePredict*toPn,predictFold,**lineDict)
    plt.plot(ForcePn,Unfolding,'bo',label="Unfolding",**markerDict)
    plt.plot(xRangePredict*toPn,predictUnfold,**lineDict)
    ax.set_yscale('log')
    # limits in PicoNewtons
    plt.xlim(xMin*toPn,xMax*toPn)
    plt.xlabel("Force (pN)")
    plt.ylabel("Rate (Hz)")
    plt.title("Woodside and Block, Figure 6a (2016)")
    plt.legend(loc='lower center')
    fig.savefig("./Woodside2016_Figure6.png")

def run():
    """
    Runs examples of the Bell-Zhurkov Model
    """
    RunWoodsideFigure6()
    
if __name__ == "__main__":
    run()

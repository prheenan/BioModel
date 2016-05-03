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
    See TestExamples.TestUtil.Bell_Test_Data.Woodside2014FoldingAndUnfoldingData
    """
    Forces,Folding,Unfolding = Data.Woodside2014FoldingAndUnfoldingData()
    """

    """
    GuessDict = dict(beta=1,
                     k0=10,
                     DeltaX=20e-9,
                     DeltaG=(max(Forces)-min(Forces)))
    inf = BellModel.GenBellZurkovFit(Forces,Folding,GuessDict)
    ax = plt.subplot(1,1,1)
    plt.plot(Forces,Folding,'ro',linewidth=0,label="Folding")
    plt.plot(Forces,inf.Prediction,'g--',linewidth=2,label="Folding")
    plt.plot(Forces,Unfolding,'bo',linewidth=0,
             label="Unfolding")
    ax.set_yscale('log')
    plt.show()

def run():
    """
    Runs examples of the Bell-Zhurkov Model
    """
    RunWoodsideFigure6()
    
if __name__ == "__main__":
    run()

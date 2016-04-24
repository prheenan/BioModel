# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

baseDir = "../"
sys.path.append(baseDir)
sys.path.append("../../../")
import Code.WLC_Fit as WLC_Fit

class ModelData:
    """
    Class to keep track of a (generated) unit test data set 
    """
    def __init__(self,ext,force,params,noise,name,ylim):
        """
        Initialize a new model object

        Args:
            ext: extension, noise-free, preedicted by model
            force: force, noise free, predicted by model 
            params: WLC_Fit.WlcParamValues used to generate the object
            noise: Noise term, to add on (ideally, should match the literature)
            name: Name of the model
        """
        self.ext = ext
        self.force = force
        self.params = params
        self.noise = noise
        self.name = name
        self.ylim = ylim

def GetBouichatData(StepInNm=0.05):
    """
    Returns samples from data from Figure 1
    From "Estimating the Persistence Length of a Worm-Like Chain Molecule ..."
    C. Bouchiat, M.D. Wang, et al.
    Biophysical Journal Volume 76, Issue 1, January 1999, Pages 409-413

web.mit.edu/cortiz/www/3.052/3.052CourseReader/38_BouchiatBiophysicalJ1999.pdf
    
    Returns:
        tuple of <z,F> in SI units
    """
    # upper and lower bound is taken from Figure 1, note nm scale
    x = np.arange(0,1335,StepInNm) * 1e-9
    # write down their parameter values, figure 1 inset
    params = WLC_Fit.WlcParamValues(kbT = 4.11e-21,L0 = 1317.52e-9,
                                    Lp =  40.6e-9,K0 = 1318.e-12)
    # for the non-extensible model, really only want to fit up to
    # some high percentage of the contour length
    values = dict([(k,v.Value) for k,v in params.GetParamDict().items()])
    y = WLC_Fit.WlcExtensible(x,**values)
    # note, by the inset in figure 1 inset / 3 error bars, 2pN is an upper
    # bound on the error we have everywhere
    noiseAmplitude = 2e-12
    # make the limits based on their plot
    ylim = np.array([-3e-12,50e-12])
    return ModelData(x,y,params,noiseAmplitude,"Bouchiat_1999_Figure1",ylim)

def CheckDataObj(DataObj,OutName=None):
    """
    Given a ModelData object, checks its parameters match what we would
    expect, then plots / shows, depending on OutName

    Args:
        DataObj : instance of ModelData, keeping all the data we want
        OutName : If not None, saves a plot.
    """
    x = DataObj.ext
    y = DataObj.force
    params = DataObj.params
    noiseAmplitude = DataObj.noise
    name = DataObj.name
    yPure = y.copy()
    ylim = DataObj.ylim
    y += noiseAmplitude * (np.random.rand(*y.shape)-0.5)*2
    print("Fitting Data From {:s}...".format(name))
    # get an extensible and non-extensible model, choose whether to varying L0
    # and Lp
    extensibleFit = WLC_Fit.ExtensibleWlcFit(x,y,VaryL0=True,
                                             VaryLp=True,VaryK0=False)
    # make sure the parameters match what the model says it should 
    assert extensibleFit.Info.ParamVals.CloseTo(params)
    nonExtensibleFit = WLC_Fit.NonExtensibleWlcFit(x,y,VaryL0=True,
                                                   VaryLp=False)
    print("Extensible Parameters")
    print(extensibleFit)
    print("Non-extensible Parameters")
    print(nonExtensibleFit)
    mFit = extensibleFit.Prediction
    mFitNon = nonExtensibleFit.Prediction
    if (OutName is not None):
        # plot everything
        toNm = 1e9
        toPn = 1e12
        fig = plt.figure()
        ylimPn = lambda: plt.ylim(toPn * ylim)
        plt.subplot(2,1,1)
        plt.plot(x*toNm,yPure*toPn,color='b',
                 label="Data, {:s} (No Noise)".format(name))
        plt.xlabel("Distance (nm)")
        plt.ylabel("Force (pN)")
        ylimPn()
        plt.legend(loc='upper left')
        plt.title("Extensible WLC better approximates FEC at high force")
        plt.subplot(2,1,2)
        plt.plot(x*toNm,y*toPn,color='k',alpha=0.3,label="Data,With Noise")
        plt.plot(x*toNm,mFit*toPn,'r-',label="Extensible",linewidth=1.5)
        plt.plot(x*toNm,mFitNon*toPn,'b--',label="Non Extensible")
        ylimPn()
        plt.xlabel("Distance (nm)")
        plt.ylabel("Force (pN)")
        plt.legend(loc='upper left')
        plt.tight_layout()
        fig.savefig(OutName + ".png")

def TestDataWithSteps(Steps,DataFunction):
    for step in Steps:
        DataObj = DataFunction(step)
        CheckDataObj(DataObj,OutName=DataObj.name + \
                     "DeltaX={:.2f}".format(step))

    
def run():
    """
    Runs some unit testing on the WLC fitting
    """
    # really, the only thing we have control over is how much we interpolate
    # over the given literature values
    StepNm = [ 0.05,0.1,0.5,1,2]
    toTest =  [ [StepNm,GetBouichatData]]
    for Steps,Function in toTest:
        TestDataWithSteps(Steps,Function)

if __name__ == "__main__":
    run()

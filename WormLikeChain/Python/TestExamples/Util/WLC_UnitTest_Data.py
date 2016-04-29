# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

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
        self.ylim = np.array(list(ylim))
        self.n = self.force.size
    @property
    def ForceWithNoise(self,seed=42):
        """
        make the noise be uniformly at random, with amplitude self.noise

        Args:
            seed: what to seed with, defaults the MOLTUAE
        Returns:
            force with added noise
        """
        np.random.seed(seed)
        return self.force + (2*np.random.rand(self.n)-1) * self.noise

def GetDataObj(x,ParamValues,noiseAmplitude,ylim,expectedMax,Name,rol=0.015):
    """
    Returns a data object, fitting the specified x with the parameters
    and noise values
    """
    params = WLC_Fit.WlcParamValues(Values=ParamValues)
    # for the non-extensible model, really only want to fit up to
    # some high percentage of the contour length
    values = dict([(k,v.Value) for k,v in params.GetParamDict().items()])
    y = WLC_Fit.WlcExtensible(x,**values)
    # note, by the inset in figure 1 inset / 3 error bars, 2pN is an upper
    # bound on the error we have everywhere
    # make the limits based on their plot
    toRet = ModelData(x,y,params,noiseAmplitude,Name,ylim)
    # the expected maximum fitted force is also from figure 1
    actualMax = np.max(toRet.force)
    # Note that this is a test on WlcExtensible, more or less.
    np.testing.assert_allclose(actualMax,expectedMax,atol=0,rtol=0.15)
    return toRet

def GetBullData(StepInNm=0.01):
    """
    Returns samples from first unfold of Figure S2.a 
    http://pubs.acs.org/doi/suppl/10.1021/nn5010588

    Bull, Matthew S., Ruby May A. Sullan, Hongbin Li, and Thomas T. Perkins.
"Improved Single Molecule Force Spectroscopy Using Micromachined Cantilevers"
    """
    # get the extensions used
    maxXnm = 19
    nSteps= maxXnm/StepInNm
    x = np.linspace(0,maxXnm,num=nSteps) * 1e-9
    L0 = 0.34e-9 * 64
    """
    # note: from Supplemental, pp 14 of 
    Edwards, Devin T., Jaevyn K. Faulk et al 
    "Optimizing 1-mus-Resolution Single-Molecule Force Spectroscopy..."
    """
    Lp = 0.4e-9
    ParamValues = dict(kbT = 4.11e-21,L0 = L0,
                       Lp =  Lp,K0 = 1318.e-12)
    Name = "Bull_2014_FigureS2"
    noiseN = 6.8e-12
    expectedMax=80e-12
    ylim = [0,expectedMax]
    return GetDataObj(x,ParamValues,noiseN,ylim,expectedMax,Name)
    
def GetBouichatData(StepInNm=0.5):
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
    maxExtNm = 1335
    # figure out the number of steps at this interpolation
    nSteps = int(np.ceil(maxExtNm/StepInNm))
    # get all the extension values
    x = np.linspace(start=0,stop=maxExtNm,num=nSteps,endpoint=True) * 1e-9
    # write down their parameter values, figure 1 inset
    ParamValues = dict(kbT = 4.11e-21,L0 = 1317.52e-9,
                       Lp =  40.6e-9,K0 = 1318.e-12)
    # note, by the inset in figure 1 inset / 3 error bars, 2pN is an upper
    # bound on the error we have everywhere
    noiseAmplitude = 2e-12
    # make the limits based on their plot
    ylim = np.array([-3e-12,52e-12])
    # the expected maximum fitted force is also from figure 1
    expectedMax = 48e-12
    Name = "Bouchiat_1999_Figure1"
    return GetDataObj(x,ParamValues,noiseAmplitude,ylim,expectedMax,Name)


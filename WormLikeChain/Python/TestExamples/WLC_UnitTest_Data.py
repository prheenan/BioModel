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
        self.ylim = ylim
        self.n = self.force.size
    @property
    def ForceWithNoise(self):
        # make the noise be uniformly at random, with amplitude self.noise
        return self.force + (2*np.random.rand(self.n)-1) * self.noise
        

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
    x = np.arange(0,1320,StepInNm) * 1e-9
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
    ylim = np.array([-3e-12,52e-12])
    return ModelData(x,y,params,noiseAmplitude,"Bouchiat_1999_Figure1",ylim)


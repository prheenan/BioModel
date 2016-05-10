# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

from FreelyJointedChain.Python.Code.FJC_Helper import FJCValues
from FreelyJointedChain.Python.Code.FJC import FJCModel

class FJC_Data:
    def __init__(self,ext,force,L0,b_kuhn,S_modulus,kbT,noise,ylim):
        param_dict = dict(L0=L0,
                          b_kuhn=b_kuhn,
                          S_modulus=S_modulus,
                          kbT=kbT)
        self.Extension = ext
        self.Force = force
        self.Params = FJCValues(Values=param_dict)
        self.noise = noise
        self.N = self.Force.size
        self.ylim = np.array(ylim)
    @property
    def ForceWithNoise(self,seed=42):
        np.random.seed(seed=seed)
        minusOneToOne = (np.random.rand(self.N)-0.5)*2
        return self.Force + minusOneToOne * self.noise

def Smith1996_Figure6(StepPn=0.01):
    """
    Returns the data from figure 6 of 

    Smith, Steven B., Yujia Cui, and Carlos Bustamente. 
    Overstretching B-DNA: The Elastic Response of Individual Double-Stranded 
    and Single-Stranded DNA Molecules."
    Science 271, no. 5250 (February 9, 1996): 795.

    Args:
        StepPn: force step in PicoNewtons
    Returns:
        FJC_Data    
    """
    EndForcePn = 80
    Num = int(np.ceil(EndForcePn/StepPn))
    toPn = 1e-12
    Force = np.linspace(0,EndForcePn,Num) * toPn
    # parameter values from after last equation in ibid
    dictV = dict(L0=27e-6,
                 b_kuhn=15e-10,
                 S_modulus=800e-12,
                 kbT=4.1e-21)
    Extension = FJCModel(Force,**dictV)
    return FJC_Data(Extension,Force,noise=2*toPn,ylim=[0,80*toPn],**dictV)
    

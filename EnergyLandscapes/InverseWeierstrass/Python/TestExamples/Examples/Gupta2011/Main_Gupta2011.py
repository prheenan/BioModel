# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append("../../../../../../")
from EnergyLandscapes.InverseWeierstrass.Python.Code import InverseWeierstrass



def ReadInData(InDir,ExtString="Ext_",ForceString="F_",FileExt=".txt",Max=2):
    AllFiles = [InDir + f for f in os.listdir(InDir)]
    # get whichever pairs of file exist in the input directory
    GetFile = lambda x,i: InDir + x + str(i) + FileExt
    ExtFiles = [GetFile(ExtString,i) for i in range(Max)]
    ForceFiles = [GetFile(ForceString,i) for i in range(Max)]
    Pairs = [ (ext,force) for ext,force in zip(ExtFiles,ForceFiles)
              if ( (ext in AllFiles) and (force in AllFiles)) ]
    # read them in as Extension,Force Pairs
    """
    data is samples at 20kHz by pp 633, 'Methods' of : 
    Gupta, Amar Nath, Abhilash Vincent, Krishna Neupane, Hao Yu, Feng Wang, 
    and Michael T. Woodside. 
    "Experimental Validation of Free-Energy-Landscape Reconstruction from 
    Non-Equilibrium Single-Molecule Force Spectroscopy Measurements." 
    Nature Physics 7, no. 8 (August 2011)
    """
    freq = 20e3
    TimeStep = 1/freq
    GetTimes = lambda x: np.linspace(start=0,stop=(x.size-1)*TimeStep,
                                     num=x.size)
    # note: data is in nm and pN, so we convert back to m and N
    Data = [(np.loadtxt(ext)*1e-9,np.loadtxt(force)*1e-12)
            for ext,force in Pairs]
    # create the data objects we will return
    ToRet = [InverseWeierstrass.FEC_Pulling_Object(GetTimes(ext),ext,
                                                   force)
             for ext,force in Data]
    return ToRet

def PlotAllFEC(Objs):
    for tmp in Objs:
        toNm = lambda x: x*1e9
        toMs = lambda x: x*1e3
        toPn = lambda x: x*1e12
        tokbT = lambda x: x/(4.1e-21)
        plt.subplot(4,1,1)
        plt.plot(toMs(tmp.Time),toNm(tmp.Extension))
        plt.ylabel("Extension (nm)")
        plt.subplot(4,1,2)
        plt.plot(toMs(tmp.Time),toPn(tmp.Force))
        plt.xlabel("Time (ms)")
        plt.ylabel("Force (pN)")
        plt.subplot(4,1,3)
        plt.plot(toNm(tmp.Extension),toPn(tmp.Force))
        plt.ylabel("Force (pN)")
        plt.subplot(4,1,4)
        plt.plot(toNm(tmp.Extension),tokbT(tmp.Work))
        plt.xlabel("Extension (nm)")
        plt.ylabel("Work (kbT)")
        plt.tight_layout()

def Analyze(Objs,NumTimeBins=75,NumPositionBins=75):
    """
    Args:
        Objs: list of InverseWeierstrass.FEC_Pulling_Object objects 
        NumTimeBins: For discretizing time, how many times to use
        NumPositionBins: For discretizing position, how many times to use
    """
    # get all the works
    InverseWeierstrass.SetAllWorkOfObjects(Objs)
    InverseWeierstrass.FreeEnergyWeightedHistogramByObject(Objs,\
                                            NumTimeBins=NumTimeBins,
                                            NumPositionBins=NumPositionBins)
    nObj = len(Objs)
    fig = plt.figure(figsize=(6,12))
    PlotAllFEC(Objs)
    fig.savefig("FEC.png")

    
def run():
    """
    Runs the IWT on the Woodside data
    """
    # read the data into objects
    Objs = ReadInData(InDir="./Example_Force_Extension_Data/",Max=100)
    Analyze(Objs)

if __name__ == "__main__":
    run()

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
    ForceOffsetN = -2e-12
    Data = [(np.loadtxt(ext)*1e-9,
             (np.loadtxt(force)*1e-12-ForceOffsetN))
            for ext,force in Pairs]
    # create the data objects we will return
    ToRet = [InverseWeierstrass.FEC_Pulling_Object(GetTimes(ext),ext,
                                                   force,
                                                   # velocity not given, eyeball
                                                   # bassed on 2nm in 1
                                                   # data point
                                                   Velocity=(2e-9)*freq)
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

def Analyze(Objs,NumBins=40):
    """
    Args:
        Objs: list of InverseWeierstrass.FEC_Pulling_Object objects 
        NumBins: For discretizing <XXX>, how many bins to use
    """
    # get all the works
    InverseWeierstrass.SetAllWorkOfObjects(Objs)
    Landscape = InverseWeierstrass.\
                FreeEnergyAtZeroForce(Objs,NumBins=NumBins)
    FreeEnergyAtZeroForce = Landscape.EnergyLandscape
    ExtBins = Landscape.ExtensionBins
    q = Landscape.Extensions
    Beta = Landscape.Beta
    # for plotting, only look at finite.
    GoodIdx = np.where(np.isfinite(FreeEnergyAtZeroForce))[0]
    # shift the free energy to F_1/2
    # approximately 20pN, see plots
    F0 = 19.5e-12
    FreeEnergyAtF0_kbT = ((FreeEnergyAtZeroForce-ExtBins*F0)*Beta)[GoodIdx]
    FreeEnergyAtF0_kbT -= np.min(FreeEnergyAtF0_kbT)
    n=3
    fig = plt.figure(figsize=(5,7))
    # plot energies in units of 1/Beta (kT), force in pN, dist in nm
    plt.subplot(n,1,1)
    xlim = lambda : plt.xlim([750,940])
    for o in Objs:
        plt.plot(o.Extension*1e9,o.Force*1e12)
    xlim()
    # Plot the free energy versus txtension as well
    plt.ylabel("Force [pN]")
    plt.xlabel("Extension [nm]")
    plt.title("Transforming {:d} FECs to a landscape".format(len(Objs)))
    FreeEnergyExt = (q * 1e9)[GoodIdx]
    plt.subplot(n,1,2)
    plt.plot(ExtBins * 1e9,FreeEnergyAtZeroForce*Beta)
    xlim()
    plt.ylim([0,250])
    plt.ylabel("G at Zero Force (kT)")
    plt.xlabel("Extension [nm]")
    plt.subplot(n,1,3)
    # sort everything by the free energy extension
    SortIdx = np.argsort(FreeEnergyExt)
    FreeEnergy = FreeEnergyAtF0_kbT[SortIdx]
    plt.plot(FreeEnergyExt[SortIdx] * 1e9,FreeEnergy)
    plt.ylabel("G at F-1/2 (kT)")
    plt.xlabel("Distance around Barrier (nm)")
    plt.tight_layout()
    plt.ylim([-0.5,10])
    plt.show()
    fig.savefig("./LandscapeReconstruction.png")
    ## make a plot of just a single force extension curve
    fig = plt.figure()
    tmp = Objs[0]
    ExtTmp = tmp.Extension*1e9
    ForceTmp = tmp.Force*1e12
    xlim = lambda : plt.xlim([min(ExtTmp),max(ExtTmp)])
    ylim = lambda : plt.ylim([min(ForceTmp),max(ForceTmp)])
    plt.plot(ExtTmp,ForceTmp,linewidth=3.0)
    plt.xlabel("Extension [nm]")
    plt.ylabel("Force [pN]")
    xlim()
    ylim()
    plt.tight_layout()
    fig.savefig("ForceExtensionCurve.png")
    GetYSpan = lambda x: ForceTmp[np.argmin(np.abs(ExtTmp-x))]
    # add in lines to show position binning
    for b in ExtBins:
        x = b*1e9
        YVals = [0,GetYSpan(x)]
        plt.plot((x,x),YVals,color='r')
    xlim()
    ylim()
    plt.tight_layout()
    fig.savefig("ForceExtensionCurve_Binned.png")
    # add in integral, showing work
    Colors = ['r','k']
    NColors = len(Colors)
    for i,b in enumerate(ExtBins[:-1]):
        x1 = b*1e9
        x2 = ExtBins[i+1]*1e9
        yVals = [GetYSpan(x1),GetYSpan(x2)]
        plt.fill_between([x1,x2],yVals,color=Colors[i % NColors],alpha=0.3)
    xlim()
    ylim()
    plt.tight_layout()
    fig.savefig("ForceExtensionCurve_Work.png")

    
def run():
    """
    Runs the IWT on the Woodside data
    """
    # read the data into objects
    Objs = ReadInData(InDir="./Example_Force_Extension_Data/",Max=100)
    Analyze(Objs)

if __name__ == "__main__":
    run()

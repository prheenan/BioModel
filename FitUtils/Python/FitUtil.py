# use OS for file IO and the like
import os
# use numpy for array operations
import numpy as np
# Use CSV for writing human readable files
import csv
from scipy.optimize import curve_fit
# path!
import ntpath
# for argument parsing easily
import argparse
# for getting formatted times
import time

# Stats
def RSQ(predicted,actual):
    # given predicted and actual values, get the RSQ
    meanObs = np.mean(actual)
    SS_Res = np.sum((predicted-actual)**2)
    SS_Tot = np.sum((actual-meanObs)**2)
    return 1 - SS_Res/SS_Tot

def lineIntersect(slope1,intercept1,slope2,intercept2):
    return (intercept1-intercept2)/(slope2-slope1)

# assumes that aThenBX are lists for the two lines
def lineIntersectParam(aThenB1,aThenB2):
    return lineIntersect(aThenB1[0],aThenB1[1],aThenB2[0],aThenB2[1])

def linModel(xData,a,b):
    # y = ax+b
    return xData*a+b

def GenFit(x,y,model=linModel,**kwargs):
    params,Cov = curve_fit(f=model,xdata=x,ydata=y,**kwargs)
    # the square root of the diagonal elements are the standard deviations
    paramsStd = np.sqrt(np.diag(Cov))
    predicted = model(x,*params)
    return params,paramsStd,predicted

def fitInfo(x,y,units=['',''],model=linModel,varStr=['a','b'],
            modelStr="y=a*x+b"
            ,degFit=1,fmtStr=".3g",full=False,simplify=True,**kwargs):
    # get all the information you could want about the fit.
    # XXX TODO: add in support for non linear models.
    # x: observed x
    # y: observed y
    # units: units of the variables in varStr
    # varStr: parameters of the fit. goes from high degree to low 
    # modelStr: describing the model.
    # degFit: degree of the model
    # fmtStr: formating of the data
    # full : if we should return all the data
    params,paramsStd,predicted = GenFit(x,y,model,**kwargs)
    R_SQ = RSQ(predicted,y)
    # if RSQ is very close to 1 (XXX add parameter?) don't display, since
    # we are likely not interested in an actual fit...
    if (not simplify or (R_SQ-1) > 1.e-6):
        modelStr += "\nRSQ: {:.3f}".format(R_SQ)
    for label,mean,stdev,unitTmp in zip(varStr,params,paramsStd,units):
        tempMStr = ("\n{:5s}={:" + fmtStr + "}").format(label,mean)
        # if either in range or told not to simplify, add the stdev
        if (not (np.isfinite(stdev) or stdev<0 or stdev == float('inf'))
            or not simplify):
            tempMStr += "+/-{:.1g}".format(stdev)
        modelStr += tempMStr
        # add the units (if we have any)
        if (len(unitTmp) > 0):
             modelStr += "[{:s}]".format(unitTmp)
    if (full):
        return predicted,modelStr,params,paramsStd,RSQ
    else:
        return predicted,modelStr


def TaylorSeries(x,y,deg=1,ZeroX=False,ZeroY=False,**kwargs):
    """
    Args:
        x: x to fit
        y: y to fit
        deg: degree of the fit
        ZeroX: if true, offsets the series so that x[0] -> 0
        ZeroY: if true, offsets the series so that y[0] -> 0
    Returns:
        return of polyfit
    """
    offsetX = x[0] if ZeroX else 0
    offsetY = y[0] if ZeroY else 0
    return np.polyfit(x-offsetX,y-offsetY,deg=deg)

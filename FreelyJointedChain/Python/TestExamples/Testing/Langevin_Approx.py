# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../../../../")
from FreelyJointedChain.Python.Code.FJC import Langevin


def PlotLangevinConvergenceAndAssertCorrect():
    """
    Tests that the Langevin function is within error to the 'exact' value.
     
    Note this is necessary because the 'exact' value is analyticaly well-defined
    but numerially unstable at small x, since 1/x goes to infinity 

    We *expect* this to throw some warnings, since it is likely the 'exact' 
    vaule is screwy.
    
    """
    maxX = 2
    stepX = 1e-5
    x = np.arange(0,maxX,stepX)
    # relative tolerance; we consider things below this to be zero
    tol = 1e-12
    exact = (1./np.tanh(x) - 1/x)
    # could divide by zero; these should all be zero
    exact[np.where(~np.isfinite(exact))] = 0
    approx = Langevin(x)
    approx[np.where(approx<tol)] = 0
    # assert that we are within 1ppm for all values of xa
    np.testing.assert_allclose(exact,approx,atol=0,rtol=1e-6)
        
    # make sure the functions match
    relError = np.abs((exact-approx)/approx)
    relError[np.where(approx < tol)] = exact-approx
    xlim = lambda : plt.xlim(-0.5,max(x)*1.05)
    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.title("9th-order Langevin provides robust approximation at small x")
    plt.plot(x,exact,'r-',linewidth=3.0,label="Exact")
    plt.plot(x,approx,label="Approximate",linestyle='--')
    plt.ylim(0,max(approx)*1.05)
    plt.ylabel("Function Value")
    plt.legend(loc='lower right')
    xlim()
    # plot the relative error
    plt.subplot(2,1,2)
    plt.plot(x,relError)
    plt.title("Relative Error in Langevin")
    plt.ylabel("Relative Error")
    plt.xlabel("x")
    xlim()
    # set the x limits
    minV = -max(relError)/5
    plt.ylim([minV,max(relError)*1.05])
    fig.savefig("./Out/LangevinConvergence.png")


def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    PlotLangevinConvergenceAndAssertCorrect()

if __name__ == "__main__":
    run()

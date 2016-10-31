# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../../")
from Code import InverseWeierstrass
from scipy.integrate import cumtrapz 

def AddNoise(signal,SNR,function=None):
    size = signal.size
    # by default, white noise uniformly distributed about 0 with the SNR
    if (function is None):
        amplitude = np.sqrt(1/SNR)
        function = lambda x: (np.random.rand(size)-0.5)*2*amplitude*x
    return signal + function(signal)

def GetEnsemble(cantilever_spring_pN_nm=10,
                force_spring_constant_pN_nm=(22-15)/(65-59),
                force_offset_pN=15,
                snr=(10)**2,
                num_points=50,
                num_ensemble = 5,
                z0_nm=59,
                z1_nm=65,
                DeltaA_kT=10):
    """
    Gets an ensemble of FEC with the given statistics.

    Assumes 'triangle wave', start at F0, go to F0+k*x linearly fwd, then lower

    Args:
        cantilever_spring_pN_nm: how stiff the cantilevr is in pN/nm
        force_spring_constant_pN_nm: assuming we increase and decrease linearly,
        the spring constant for that local region

        force_offset_pN: where we start in force at the starting extension
        snr: signal-to-noise ratio (default to white noise)

        num_points: how many extension,force points
        num_ensemble: how many copies in the ensemble
        z0_nm, z1_nm: starting and ending extensions in nm
    Returns:
        tuple of <list of forward objects,list of reverse objects,
        free energy different in J>
    """
    ext_nm = np.linspace(z0_nm,z1_nm,num=num_points)
    # get the force by a simpe spring constant
    force_pN = (ext_nm-ext_nm[0]) * force_spring_constant_pN_nm + \
               force_offset_pN
    # okay, convert everything to 'real' units
    force_N =  force_pN * 1e-12
    ext_m = ext_nm * 1e-9
    cantilever_spring_N_m =  cantilever_spring_pN_nm * 1e-3
    # get the reverse extensions
    ext_rev_m =  ext_m[::-1].copy()
    # add in noise to the force, make the ensembles
    fwd_objs,rev_objs = [],[]
    # reverse the force and add the free energy difference to it...
    reversed_force = force_N[::-1].copy()
    DeltaA =cumtrapz(x=ext_m,y=force_N,initial=0)
    for i in range(num_ensemble):
        # add noise to each member of the ensemble separately
        force_N_noise = AddNoise(force_N,snr)
        force_N_noise_rev =  AddNoise(reversed_force,snr)
        fwd=InverseWeierstrass.\
            FEC_Pulling_Object(None,ext_m,force_N_noise,
                               SpringConstant=cantilever_spring_N_m,
                               ZFunc = lambda: ext_m)
        rev=InverseWeierstrass.\
             FEC_Pulling_Object(None,ext_rev_m,force_N_noise_rev,
                                SpringConstant=cantilever_spring_N_m,
                                ZFunc = lambda: ext_rev_m)
        fwd_objs.append(fwd)
        rev_objs.append(rev)
    InverseWeierstrass.SetAllWorkOfObjects(fwd_objs)
    InverseWeierstrass.SetAllWorkOfObjects(rev_objs)
    return fwd_objs,rev_objs,DeltaA
    

def TestForwardBackward():
    """
    Tests that the forward and backward schemes work fine 
    """
    tolerance_deltaA = 0.01
    np.random.seed(42)
    num_points=500
    points_per_bin=1
    num_bins = num_points/points_per_bin
    fwd_objs,rev_objs,deltaA_true= GetEnsemble(num_points=num_points)
    delta_A_calc = InverseWeierstrass.NumericallyGetDeltaA(fwd_objs,
                                                           rev_objs)
    kT = 4.1e-21
    deltaA = deltaA_true[-1]
    np.testing.assert_allclose(deltaA/kT,delta_A_calc/kT,atol=0,
                               rtol=tolerance_deltaA)
    # POST: correct DeltaA to within tolerance. For small DeltaA,
    # Exp(DeltaA +/- Err)/Exp(DeltaA)=Exp(+/- Err)~1+Err, so
    # we should have very small errors in the energy landscape
    landscape = InverseWeierstrass.FreeEnergyAtZeroForce(fwd_objs,num_bins,[])
    landscape_rev = \
        InverseWeierstrass.FreeEnergyAtZeroForce(fwd_objs,num_bins,rev_objs)
    plt.plot(landscape.Extensions,landscape.EnergyLandscape)
    plt.plot(landscape_rev.Extensions,landscape_rev.EnergyLandscape)
    plt.show()

    
def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    TestForwardBackward()

if __name__ == "__main__":
    run()

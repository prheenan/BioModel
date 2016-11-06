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
                num_ensemble=5,
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
    
def TestWeighting():
    """
    Tests the forward and reverse weighing function from equation 18 
    of (noting this generalized accd to equations 2 and discussion after 12)
    Hummer, G. & Szabo, A. 
    Free energy profiles from single-molecule pulling experiments. 
    PNAS 107, 21441-21446 (2010)
    """
    Fwd = InverseWeierstrass.ForwardWeighted
    Rev = InverseWeierstrass.ReverseWeighted
    # test one and zero conditions for forward
    fwd_is_one = dict(nf=1,vf=1,Wfn=0,Wf=0,Beta=0,DeltaA=0,nr=0)
    fwd_is_zero = dict(nf=1,vf=0,Wfn=0,Wf=0,Beta=0,DeltaA=0,nr=0)
    np.testing.assert_allclose(1,Fwd(**fwd_is_one))
    np.testing.assert_allclose(0,Fwd(**fwd_is_zero))
    # test one and zero conditions for revese
    rev_is_one = dict(nr=1,vr=1,Wrn=0,Wr=0,Beta=0,DeltaA=0,nf=0)
    rev_is_zero = dict(nr=1,vr=0,Wrn=0,Wr=0,Beta=0,DeltaA=0,nf=0)
    np.testing.assert_allclose(1,Rev(**rev_is_one))
    np.testing.assert_allclose(0,Rev(**rev_is_zero))
    # POST: very simple conditions work. now try ones with still no deltaA
    np.testing.assert_allclose(np.exp(-1)/2,
                               Fwd(vf=1,nf=1,nr=1,Wfn=0,Wf=1,Beta=1,DeltaA=0))
    np.testing.assert_allclose(np.exp(1)/2,
                               Rev(vr=1,nf=1,nr=1,Wrn=0,Wr=-1,Beta=1,DeltaA=0))
    # POST: no delta A works, check with DeltaA
    np.testing.assert_allclose(2*np.exp(-1)/(2+3*np.exp(-2)),
                               Fwd(vf=1,nf=2,nr=3,Wfn=1,Wf=1,Beta=1,DeltaA=-1))
    # XXX reverse is broken? typo between hummer and etc...
    np.testing.assert_allclose(2*np.exp(1)/(2+3*np.exp(2)),
                               Rev(vr=1,nf=3,nr=2,Wrn=-3,Wr=-2,Beta=1,DeltaA=1))
    # POST: also works with DeltaA... pretty convincing imo

def TestBidirectionalEnsemble(seed=42,tolerance_deltaA=0.01,snr=10,
                              tol_energy_atol_kT = 0.5,num_points=1000,
                              num_bins=50,
                              tol_energy_rtol = 0,ensemble_kwargs=dict()):
    """
    Tests that a (simple) Bi directional ensemble gives the correct energy

    Args:
        seed: what to seed the PRNG with
        tolerances_deltaA: what tolerance to allow for energy differences
        snr: signal to noise ratio for the enesemble FECs
        num_points: number of points in each FEC
        num_bins: number of bins (energy landscape points)

        tol_energy_atol_kT/ tol_energy_rtol: the absolute and relative energy 
        tolerance; all point on the forward and reverse curves should be the 
        same up to these tolerances
        
        ensemble_kwargs: passed to the ensemble
    Returns:  
        nothing, but fails if DeltaA is calculated wrong, of if the forward
        and revese parts dont quite work...
    """
    # keep a common seed (could iterate)
    np.random.seed(seed)
    fwd_objs,rev_objs,deltaA_true= GetEnsemble(num_points=num_points,
                                               snr=snr,**ensemble_kwargs)
    delta_A_calc = InverseWeierstrass.NumericallyGetDeltaA(fwd_objs,
                                                           rev_objs)
    kT = 4.1e-21
    deltaA = deltaA_true[-1]
    diff = deltaA-delta_A_calc
    np.testing.assert_allclose(deltaA/kT,delta_A_calc/kT,atol=0,
                               rtol=tolerance_deltaA)
    # POST: correct DeltaA to within tolerance. For small DeltaA,
    # Exp(DeltaA +/- Err)/Exp(DeltaA)=Exp(+/- Err)~1+Err, so
    # we should have very small errors in the energy landscape
    # the forward and reverse landscapes should be pretty much identical (?)
    landscape = InverseWeierstrass.FreeEnergyAtZeroForce(fwd_objs,num_bins,[])
    landscape_rev = InverseWeierstrass.\
        FreeEnergyAtZeroForce(fwd_objs,num_bins,rev_objs)
    np.testing.\
        assert_allclose(landscape.EnergyLandscape/kT,
                        landscape_rev.EnergyLandscape/kT,
                        atol=tol_energy_atol_kT,
                        rtol=tol_energy_rtol)
    plt.plot(landscape.EnergyLandscape/kT-landscape_rev.EnergyLandscape/kT)
    plt.show()


    
    
def TestForwardBackward():
    """
    Tests that the forward and backward schemes work fine 

    Asserts;
        (1) DeltaA is calculated correctly wrt forward and backwards (within
        some small tolerance) for a one-state system (noisy spring)
        (2) the "forward-only" and bi-directional trajectories give the same 
        energy landscape for the same setup as (1)
    """
    # 'normal' snr should have normal toleance
    TestBidirectionalEnsemble(snr=10,
                              tol_energy_atol_kT=0.5,
                              tol_energy_rtol=0)
    # high SNR, should match OK...
    TestBidirectionalEnsemble(snr=200,
                              tol_energy_atol_kT=0.2,
                              tol_energy_rtol=0)
    # infinite SNR, should match pretty much exactly
    TestBidirectionalEnsemble(snr=np.inf,
                              tol_energy_atol_kT=0.0,
                              tol_energy_rtol=1e-9)


    
def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    TestWeighting()
    TestForwardBackward()

if __name__ == "__main__":
    run()

# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../../../../../../")
from FitUtil.EnergyLandscapes.InverseWeierstrass.Python.Code import \
    InverseWeierstrass
from scipy.integrate import cumtrapz
import copy
from GeneralUtil.python import PlotUtilities

def AddNoise(signal,snr,function=None):
    size = signal.size
    # by default, white noise uniformly distributed about 0 with the SNR
    if (function is None):
        amplitude = np.sqrt(1/snr)
        function = lambda x: (np.random.rand(size)-0.5)*2*amplitude*np.mean(x)
    return signal + function(signal)

def Force_pN(x_nm,k_pN_nm,f_offset_pN):
    """
    Gets and extension and force offset force function 

    Args:
        x_nm: extension, in nm
        k_pN_nm: spring constant, in pN/nm
        f_offset_pN: force offset, in pN
    Returns: 
        Simple harmonic force in pN
    """
    return (x_nm-x_nm[0]) * k_pN_nm + f_offset_pN

def GetEnsemble(cantilever_spring_pN_nm=10,
                force_spring_constant_pN_nm=(22-15)/(65-59),
                reverse_force_spring_constant_pN_nm=None,
                reverse_force_offset_pN=None,
                force_offset_pN=15,
                snr=(10)**2,
                num_points=50,
                num_ensemble=5,
                z0_nm=59,
                z1_nm=65,
                noise_function=None):
    """
    Gets an ensemble of FEC with the given statistics.

    Assumes 'triangle wave', start at F0, go to F0+k*x linearly fwd, then lower

    Args:
        cantilever_spring_pN_nm: how stiff the cantilevr is in pN/nm
        force_spring_constant_pN_nm: assuming we increase and decrease linearly,
        the spring constant for that local region

        force_offset_pN: where we start in force at the starting extension
        snr: signal-to-noise ratio (default to white noise). 
     
        reverse_force_spring_constant_pN_nm: if not None, used to generate
        the reverse. otherwise, just use forward (only pulling spring 

        reverse_force_offset_pN: offset for the reverse force, only used 
        if we have a different spring constant (recquired)
    
        num_points: how many extension,force points
        num_ensemble: how many copies in the ensemble
        z0_nm, z1_nm: starting and ending extensions in nm

        noise_function: passed to AddNoise
    Returns:
        tuple of <list of forward objects,list of reverse objects,
        free energy different in J>
    """
    ext_nm = np.linspace(z0_nm,z1_nm,num=num_points)
    # get the force by a simpe spring constant
    force_pN = Force_pN(ext_nm,force_spring_constant_pN_nm,force_offset_pN)
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
    reversed_ext_nm = ext_nm[::-1]
    DeltaA =cumtrapz(x=ext_m,y=force_N,initial=0)
    noise_args = dict(snr=snr,
                      function=noise_function)
    for i in range(num_ensemble):
        # add noise to each member of the ensemble separately
        force_N_noise = AddNoise(force_N,**noise_args)
        if (reverse_force_spring_constant_pN_nm is None):
            force_N_noise_rev =  AddNoise(reversed_force,**noise_args)
        else:
            assert reverse_force_offset_pN is not None ,\
                   "Must provide a reverse offset if using"
            force_rev_pN = Force_pN(reversed_ext_nm,
                                    force_spring_constant_pN_nm,
                                    reverse_force_offset_pN)
            force_N_noise_rev = AddNoise(force_rev_pN*1e-12,**noise_args)
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
                              tol_energy_atol_kT=1,
                              tol_energy_rtol=0)
    # high SNR, should match OK...
    TestBidirectionalEnsemble(snr=200,
                              tol_energy_atol_kT=0.2,
                              tol_energy_rtol=0)
    # infinite SNR, should match pretty much exactly
    TestBidirectionalEnsemble(snr=np.inf,
                              tol_energy_atol_kT=0.0,
                              tol_energy_rtol=1e-9)

    
def Swap(switch_m,tau_m,swap_from,swap_to,sign):
    """
    Simple way of swapping between states based on a 'switching location'
    in extension (weighed exponentially from there)
      
    Args:
       switch_m: where the exponential is offset in m
       tau_m: decay constant in meters 
       swap_from: object to swap from (initial state)
       swap_to: object to swp to (final state) 
       sign: for the argument of the exponential
    Returns:
       new object, which transitions between the two states
    """
    ext = swap_from.Extension
    uniform_random = np.random.uniform(size=ext.size)
    probability = np.minimum(1,np.exp(sign*(ext-switch_m)/tau_m))
    New = copy.deepcopy(swap_from)
    Idx = np.where(probability >= uniform_random)[0]
    New.Force[Idx] = swap_to.Force[::-1][Idx]
    return New


def TestHummer2010():
    """
    Recreates the simulation from Figure 3 of 

    Hummer, G. & Szabo, A. 
    Free energy profiles from single-molecule pulling experiments. 
    PNAS 107, 21441-21446 (2010).
    """
    # estmate nose amplitude, Figure 3 A ibid
    snr = (10/2)**2
    #estimate stifness of system in forward and reverse
    k_fwd = (15-8)/(225-200)
    k_rev = (20-14)/(265-237)
    # noise for force
    noise_N = 2e-12
    # noise is uniform
    noise_function = lambda x: (np.random.rand(x.size) - 0.5) * 2 * noise_N
    # get the 'normal' ensemble (no state switching)
    # note: hummer and Szabo do a complicated simulation of the bead + linkers
    # I can't do that, so I just assume we have a super stiff spring (meaning
    # the ensemble extensions are close to the molecular)
    ensemble_kwargs = dict(cantilever_spring_pN_nm=100,
                           force_spring_constant_pN_nm=k_fwd,
                           reverse_force_spring_constant_pN_nm=k_rev,
                           reverse_force_offset_pN=20,
                           force_offset_pN=8,
                           num_ensemble=10,
                           z0_nm=195,
                           z1_nm=262,
                           snr=snr,
                           num_points=500,
                           noise_function=noise_function)
    fwd_objs,rev_objs,DeltaA = GetEnsemble(**ensemble_kwargs)
    # really stupid way of flipping: just exponential increase/decrease from
    # 'switch' location on forward and reverse 
    # determine the 
    fwd_switch_m = 225e-9
    rev_switch_m = 240e-9
    tau = 2e-9
    state_fwd = []
    state_rev = []
    debug = False
    for fwd,rev in zip(fwd_objs,rev_objs):
        N = fwd.Force.size
        fwd_switch = Swap(switch_m=fwd_switch_m,swap_from=fwd,
                          swap_to=rev,sign=1,tau_m=tau)
        rev_switch = Swap(switch_m=rev_switch_m,swap_from=rev,
                          swap_to=fwd,sign=-1,tau_m=tau)
        state_fwd.append(fwd_switch)
        state_rev.append(rev_switch)
        if (debug):
            plt.plot(fwd_switch.Extension,fwd_switch.Force,color='r',alpha=0.3)
            plt.plot(rev_switch.Extension,rev_switch.Force,color='b',alpha=0.3)
            plt.show()
    # POST: fwd and reverse have the forward and reverse trajectories 
    # go ahead and made the energy landscapes
    num_bins=100
    landscape = InverseWeierstrass.FreeEnergyAtZeroForce(state_fwd,num_bins,[])
    landscape_rev = InverseWeierstrass.\
                    FreeEnergyAtZeroForce(state_fwd,num_bins,state_rev)
    kT = 4.1e-21
    # See figure 3b inset, inid, for f_(1/2)... but they actually use 14pN (
    # test)
    f_one_half = 14e-12
    # for some reason, they offset the energies?... Figure 3A
    energy_offset_kT = 20
    landscape_fwd_kT = landscape.EnergyLandscape/kT + energy_offset_kT
    landscape_rev_kT = landscape_rev.EnergyLandscape/kT + energy_offset_kT
    ext_fwd = landscape_rev.Extensions
    ext_rev = landscape.Extensions
    # should be very close before XXXnm
    split_point_meters = 230e-9
    CloseIdxFwd = np.where(ext_fwd < split_point_meters)[0]
    CloseIdxRev = np.where(ext_rev < split_point_meters)[0]
    limit = min(CloseIdxFwd.size,CloseIdxRev.size)
    assert limit > num_bins/4 , "Should have roughly half of data before 230nm"
    # want landscapees before 230nm to be within 10% of each other
    np.testing.assert_allclose(landscape_fwd_kT[CloseIdxFwd[:limit]],
                               landscape_rev_kT[CloseIdxRev[:limit]],
                               rtol=0.1)
    # POST: 'early' region is fine
    # check the bound on the last points (just estimate these by eye)
    forward_maximum_energy_kT = 300
    reverse_maximum_energy_kT = 250
    np.testing.assert_allclose(landscape_fwd_kT[-1],forward_maximum_energy_kT,
                               rtol=0.05)
    np.testing.assert_allclose(landscape_rev_kT[-1],reverse_maximum_energy_kT,
                               rtol=0.05)
    # POST: endpoints match Figure 3 bounds
    landscape_fonehalf_kT = (landscape_rev_kT*kT-ext_rev* f_one_half)/kT
    # get the relative landscape hummer and szabo plot (their min is about
    # 2.5kT offset from zero)
    offset_kT_tilted = 2.5
    landscape_fonehalf_kT_rel =  \
        landscape_fonehalf_kT - min( landscape_fonehalf_kT) + offset_kT_tilted
    # make sure the barrier height is about right
    idx_barrier = np.where( (ext_rev > 220e-9) &
                            (ext_rev < 240e-9) )
    barrier_region = landscape_fonehalf_kT_rel[idx_barrier]
    expected_barrier_height_kT = 5
    barrier_delta = np.max(barrier_region)-np.min(landscape_fonehalf_kT_rel)
    np.testing.assert_allclose(barrier_delta,
                               expected_barrier_height_kT,atol=1)
    # POST: height should be quite close to Figure 3
    ToX = lambda x: x*1e9
    xlim = lambda: plt.xlim([190,265])
    fig = PlotUtilities.figure(figsize=(4,7))
    plt.subplot(2,1,1)
    plt.plot(ToX(ext_fwd),landscape_rev_kT,color='r',alpha=0.6,
             linestyle='-',linewidth=3,label="Bi-directional")
    plt.plot(ToX(ext_rev),landscape_fwd_kT,color='g',
             linestyle='--',label="Only Forward")
    plt.ylim([0,300])
    xlim()
    PlotUtilities.lazyLabel("","Free Energy (kT)",
                            "Hummer 2010, Figure 3")
    plt.subplot(2,1,2)
    plt.plot(ToX(ext_rev),landscape_fonehalf_kT_rel,color='r')
    plt.ylim([0,25])
    xlim()
    PlotUtilities.lazyLabel("Extension q (nm)","Energy at F_(1/2) (kT)","")
    PlotUtilities.savefig(fig,"out.png")

    
def run():
    """
    Runs all IWT unit tests
    """
    np.seterr(all='raise')
    np.random.seed(42)
    TestWeighting()
    TestForwardBackward()
    TestHummer2010()



if __name__ == "__main__":
    run()

# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../../../../../../")
from FitUtil.EnergyLandscapes.InverseWeierstrass.Python.Code import \
    InverseWeierstrass,WeierstrassUtil
from scipy.integrate import cumtrapz
import copy
from GeneralUtil.python import CheckpointUtilities,GenUtilities,PlotUtilities
from scipy.interpolate import interp1d
import Simulation

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
                num_ensemble=100,
                z0_nm=59,
                z1_nm=65,
                velocity_m_per_s=40e-9,
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
    ext_nm = np.linspace(z0_nm,z1_nm,num=num_points,endpoint=True)
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
    reversed_ext_nm = ext_nm[::-1].copy()
    DeltaA =cumtrapz(x=ext_m,y=force_N,initial=0)
    noise_args = dict(snr=snr,
                      function=noise_function)
    tau = (max(ext_nm)-min(ext_nm)) * 1e-9/velocity_m_per_s
    time = np.linspace(0,tau,ext_nm.size)
    for i in range(num_ensemble):
        # add noise to each member of the ensemble separately
        force_N_noise = AddNoise(force_N,**noise_args)
        if (reverse_force_spring_constant_pN_nm is None):
            force_N_noise_rev =  AddNoise(reversed_force,**noise_args)
        else:
            assert reverse_force_offset_pN is not None ,\
                   "Must provide a reverse offset if using"
            force_rev_pN = Force_pN(reversed_ext_nm,
                                    reverse_force_spring_constant_pN_nm,
                                    reverse_force_offset_pN)
            force_N_noise_rev = AddNoise(force_rev_pN*1e-12,**noise_args)       
        fwd=InverseWeierstrass.\
            FEC_Pulling_Object(time,ext_m,force_N_noise,
                               Velocity=velocity_m_per_s,
                               SpringConstant=cantilever_spring_N_m)
        rev=InverseWeierstrass.\
             FEC_Pulling_Object(time,ext_rev_m,force_N_noise_rev,
                                Velocity=-1 * velocity_m_per_s,
                                SpringConstant=cantilever_spring_N_m)
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
    fwd_is_one = dict(nf=1,v=1,Wn=0,W=0,Beta=0,DeltaA=0,nr=0)
    fwd_is_zero = dict(nf=1,v=0,Wn=0,W=0,Beta=0,DeltaA=0,nr=0)
    np.testing.assert_allclose(1,Fwd(**fwd_is_one))
    np.testing.assert_allclose(0,Fwd(**fwd_is_zero))
    # test one and zero conditions for revese
    rev_is_one = dict(nr=1,v=1,Wn=0,W=0,Beta=0,DeltaA=0,nf=0)
    rev_is_zero = dict(nr=1,v=0,Wn=0,W=0,Beta=0,DeltaA=0,nf=0)
    np.testing.assert_allclose(1,Rev(**rev_is_one))
    np.testing.assert_allclose(0,Rev(**rev_is_zero))
    # POST: very simple conditions work. now try ones with still no deltaA
    np.testing.assert_allclose(np.exp(-1)/2,
                               Fwd(v=1,nf=1,nr=1,Wn=0,W=1,Beta=1,DeltaA=0))
    np.testing.assert_allclose(np.exp(1)/2,
                               Rev(v=1,nf=1,nr=1,Wn=0,W=-1,Beta=1,DeltaA=0))
    # POST: no delta A works, check with DeltaA
    np.testing.assert_allclose(2*np.exp(-1)/(2+3*np.exp(-2)),
                               Fwd(v=1,nf=2,nr=3,Wn=1,W=1,Beta=1,DeltaA=-1))
    # XXX reverse is broken? typo between hummer and etc...
    np.testing.assert_allclose(2*np.exp(1)/(2+3*np.exp(2)),
                               Rev(v=1,nf=3,nr=2,Wn=-3,W=-2,Beta=1,DeltaA=1))
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


def check_hummer_by_ensemble(kT,landscape,landscape_both,f_one_half):
    # See figure 3b inset, inid, for f_(1/2)... but they actually use 14pN (
    # test)
    # for some reason, they offset the energies?... Figure 3A
    energy_offset_kT = 20
    energy_offset_fwd_only_kT = 75
    num_bins = landscape.EnergyLandscape.size
    landscape.EnergyLandscape -= min(landscape.EnergyLandscape)
    landscape_both.EnergyLandscape -= \
            min(landscape_both.EnergyLandscape)
    landscape_fwd_kT = landscape.EnergyLandscape/kT + energy_offset_fwd_only_kT
    landscape_both_kT = landscape_both.EnergyLandscape/kT + energy_offset_kT
    ext_fwd = landscape.Extensions
    ext_both = landscape_both.Extensions
    # should be very close before XXXnm
    split_point_meters = 235e-9
    CloseIdxFwd = np.where(ext_fwd < split_point_meters)[0]
    CloseIdxBoth = np.where(ext_both < split_point_meters)[0]
    limit = min(CloseIdxFwd.size,CloseIdxBoth.size)
    assert limit > num_bins/3 , "Should have roughly half of data before 230nm"
    # want landscapees before 230nm to be within 10% of each other
    np.testing.assert_allclose(landscape_fwd_kT[CloseIdxFwd[limit:]],
                               landscape_both_kT[CloseIdxBoth[limit:]],
                               rtol=0.1)
    # POST: 'early' region is fine
    # check the bound on the last points (just estimate these by eye)
    forward_maximum_energy_kT = 250
    both_maximum_energy_kT = 250
    np.testing.assert_allclose(landscape_fwd_kT[-1],forward_maximum_energy_kT,
                               rtol=0.05)
    np.testing.assert_allclose(landscape_both_kT[-1],reverse_maximum_energy_kT,
                               rtol=0.05)
    # POST: endpoints match Figure 3 bounds
    landscape_fonehalf_kT = (landscape_both_kT*kT-ext_both* f_one_half)/kT
    # get the relative landscape hummer and szabo plot (their min is about
    # 2.5kT offset from zero)
    offset_kT_tilted = 2.5
    landscape_fonehalf_kT_rel =  \
        landscape_fonehalf_kT - min( landscape_fonehalf_kT) + offset_kT_tilted
    # make sure the barrier height is about right
    idx_barrier = np.where( (ext_both > 220e-9) &
                            (ext_both < 240e-9) )
    barrier_region = landscape_fonehalf_kT_rel[idx_barrier]
    expected_barrier_height_kT = 5
    barrier_delta = np.max(barrier_region)-np.min(landscape_fonehalf_kT_rel)
    np.testing.assert_allclose(barrier_delta,
                               expected_barrier_height_kT,atol=1)

def landscape_plot(landscape,landscape_rev,landscape_rev_only,kT,f_one_half):
    ToX = lambda x: x*1e9
    xlim = lambda: plt.xlim([190,265])
    landscape_rev_kT = landscape_rev.EnergyLandscape/kT
    landscape_fwd_kT = landscape.EnergyLandscape/kT
    landscape_rev_only_kT = landscape_rev_only.EnergyLandscape/kT
    ext_fwd = landscape_rev.Extensions
    ext_rev = landscape.Extensions
    landscape_fonehalf_kT = (landscape_rev_kT*kT-ext_rev* f_one_half)/kT
    landscape_fonehalf_kT_rel = landscape_fonehalf_kT-min(landscape_fonehalf_kT)
    plt.subplot(2,1,1)
    # add in the offsets, since we dont simulate before...
    plt.plot(ToX(ext_fwd),landscape_rev_kT+20,color='r',alpha=0.6,
             linestyle='-',linewidth=3,label="Bi-directional")
    plt.plot(ToX(ext_rev),landscape_fwd_kT+75,color='b',
             linestyle='--',label="Only Forward")
    plt.plot(ToX(landscape_rev_only.Extensions),landscape_rev_only_kT+20,
             "g--",label="Only Reverse")
    plt.ylim([0,300])
    xlim()
    PlotUtilities.lazyLabel("","Free Energy (kT)",
                            "Hummer 2010, Figure 3")
    plt.subplot(2,1,2)
    plt.plot(ToX(ext_rev),landscape_fonehalf_kT_rel,color='r')
    plt.ylim([0,25])
    xlim()
    PlotUtilities.lazyLabel("Extension q (nm)","Energy at F_(1/2) (kT)","")

def _assert_digitization_correct(x_m_abs,n,obj):
    """
    checks that the digitization procedure works fine

    Args:
        x_m_abs: the 'absolute' x value in meters expected
        n: the number of bins to use for digitixation
        obj: the obhect to digitize 

    Returns:
        nothing, throws an error if something went wrong
    """
    bins = np.linspace(min(x_m_abs),max(x_m_abs),endpoint=True,num=n)
    digitized_ext = obj._GetDigitizedGen(Bins=bins,ToDigitize=obj.Extension)
    combined_digitized_data = sorted(np.concatenate(digitized_ext))
    np.testing.assert_allclose(combined_digitized_data,sorted(x_m_abs)), \
        "error, digitization lost data"
    # POST: exactly the correct data points were digitized. check that 
    # they wount up in the right bin
    for i,arr in enumerate(digitized_ext):
        # data with data between bin[i] and [i+1] is binned into i.
        # endpoints: data in the last bin (N-1) should be > bin N-2
        # endpoints: data in the first bin (0) should be < bin 1
        if (i < n-1):
            assert (arr <= bins[i]).all(), "upper bound broken"
        else:
            # bin n-1 should be entirely greater than n-2
            assert (arr > bins[i-1]).all(), "upper bound broken"            
        if (i > 0):
            assert (arr >= bins[i-1]).all(), "lower bound broken"            
        else:
            # bin 0 should be entirely less than 0 
            assert (arr < bins[i+1]).all(), "lower bound broken"
    # POST: all bins work 

def _assert_data_correct(obj,x_nm,offset_pN,k_pN_per_nm,
                         assert_dict=dict(atol=1e-30,rtol=1e-9)):
    """
    asserts that obj has the correct force and work (assuming simple ramp)

    Args:
        obj: IWT pulling object
        x_nm: x values in nanometers of the pulling experiment
        offset_pN : where the pulling experiment starts from
        k_pN_per_nm : stiffness of the force vs extension
        assert_dict: extra options passed to np.testing.assert_allclose
    Returns:
        nothnig, throws an error if something goes wrong
    """
    # # check that the force and work match
    work_joules = lambda x,f : cumtrapz(x=x,y=f,initial=0)
    x_m_abs = x_nm * 1e-9
    x_m = x_m_abs - x_m_abs[0]
    force_fwd = (k_pN_per_nm * 1e-3) * x_m + (offset_pN * 1e-12)
    work_tmp = work_joules(x=x_m,f=force_fwd)
    np.testing.assert_allclose(obj.Work,work_tmp,**assert_dict)
    np.testing.assert_allclose(obj.Force,force_fwd,**assert_dict)
    np.testing.assert_allclose(obj.Extension,x_m_abs,**assert_dict)
    # # XXX check that the digitization routine works well 
    _assert_digitization_correct(x_m_abs=x_m_abs,n=50,obj=obj)
    
def assert_noiseless_ensemble_correct(z0_nm,z1_nm,fwd_objs,rev_objs,
                                      fwd_offset_pN,rev_offset_pN,
                                      k_fwd,k_rev,**kw):
    """
    Asserts that the fwd and reversed objects have the correct
    data

    Args:
       <z0/z1>_nm: the start and end of the line ramp
       <fwd/rev>_objs: the forward and reverse iwt pulling objects 
       <fwd/rev>_offset_pN: the offsets for the t=0 part of the ramp
       k_<fwd/rev>: the forward or reverse spring const, pN/nm
        **kw: passed to _assert_data_correct
    
    Returns: 
       nothing, throws an error if something goes wrong 
    """
    N = fwd_objs[0].Force.size
    x_fwd = np.linspace(z0_nm,z1_nm,num=N,endpoint=True)
    x_rev = x_fwd[::-1].copy()
    for f,r in zip(fwd_objs,rev_objs):
        _assert_data_correct(f,x_nm=x_fwd,offset_pN=fwd_offset_pN,
                             k_pN_per_nm=k_fwd,**kw)
        _assert_data_correct(r,x_nm=x_rev,offset_pN=rev_offset_pN,
                             k_pN_per_nm=k_rev,**kw)

def assert_noisy_ensemble_correct(fwd,rev):
    """
    Assuming that the digitization functions work well on the noiseless
    distribution, this tests that the ensemble functions work fine..
    
    Args:
        fwd,rev: the forward and reverse objects to test
    Returns:
    """
    n = 50
    for f,r in zip(fwd,rev):
        _assert_digitization_correct(f.Extension,n=n,obj=f)
        _assert_digitization_correct(r.Extension,n=n,obj=r)
    # POST: the digitization is OK. 
    # check that the ensemble binning is OK.

def get_simulated_ensemble(n,**kw):
    to_ret = []
    for _ in range(n):
        t,q,z,f,p = Simulation.hummer_force_extension_curve(**kw)
        velocity = p["velocity"]
        spring_constant = p['k']
        beta = p['beta']
        kT = 1/beta
        good_idx = np.where( (z > 325e-9) & (z < 425e-9))
        initial_dict = dict(Time=t[good_idx],Extension=q[good_idx],
                            Force=f[good_idx],Velocity=velocity,kT=kT,
                            SpringConstant=spring_constant)
        tmp = InverseWeierstrass.FEC_Pulling_Object(**initial_dict)
        to_ret.append(tmp)
    return to_ret

def HummerData(cache_dir="./cache",seed=42):
    # estmate nose amplitude, Figure 3 A ibid
    snr = (10/2)**2
    #estimate stifness of system in forward and reverse
    k_fwd = (15-8)/(225-200)
    k_rev = (20-14)/(265-237)
    # noise for force
    noise_N = 2e-12
    # dont add in noise until after swapping 
    noise_function = lambda x: np.zeros(x.size)
    # get the 'normal' ensemble (no state switching)
    # note: hummer and Szabo do a complicated simulation of the bead + linkers
    # I can't do that, so I just assume we have a super stiff spring (meaning
    # the ensemble extensions are close to the molecular)
    z0_nm = 195
    z1_nm = 262
    N = 500
    fwd_offset_pN = 7
    rev_offset_pN = 20
    ensemble_kwargs = dict(cantilever_spring_pN_nm=100,
                           force_spring_constant_pN_nm=k_fwd,
                           reverse_force_spring_constant_pN_nm=k_rev,
                           reverse_force_offset_pN=rev_offset_pN,
                           force_offset_pN=fwd_offset_pN,
                           num_ensemble=100,
                           z0_nm=z0_nm,
                           z1_nm=z1_nm,
                           snr=snr,
                           num_points=N,
                           noise_function=noise_function)
    fwd_objs,rev_objs,DeltaA = GetEnsemble(**ensemble_kwargs)
    # make sure all the data are correct
    assert_noiseless_ensemble_correct(z0_nm,z1_nm,fwd_objs,rev_objs,
                                      fwd_offset_pN,rev_offset_pN,
                                      k_fwd,k_rev)
    n = 200
    np.random.seed(seed)
    cache_fwd,cache_rev = [cache_dir + s +"/" for s in ["_fwd","_rev"]]
    GenUtilities.ensureDirExists(cache_fwd)
    GenUtilities.ensureDirExists(cache_rev)
    func_fwd = lambda : get_simulated_ensemble(n)
    func_rev = lambda : get_simulated_ensemble(n,reverse=True)
    fwd = CheckpointUtilities.multi_load(cache_fwd,func_fwd)
    rev = CheckpointUtilities.multi_load(cache_rev,func_rev)
    # POST: the ensemble data (without noise) are OK 
    # read in the simulateddata 
    #assert_noisy_ensemble_correct(state_fwd,state_rev)
    return fwd,rev

def landscape_plot(landscape,landscape_rev,landscape_rev_only,kT,f_one_half):
    ToX = lambda x: x*1e9
    xlim = lambda: plt.xlim([190,265])
    landscape_rev_kT = landscape_rev.EnergyLandscape/kT
    landscape_fwd_kT = landscape.EnergyLandscape/kT
    landscape_rev_only_kT = landscape_rev_only.EnergyLandscape/kT
    ext_fwd = landscape_rev.Extensions
    ext_rev = landscape.Extensions
    landscape_fonehalf_kT = (landscape_rev_kT*kT-ext_rev* f_one_half)/kT
    landscape_fonehalf_kT_rel = landscape_fonehalf_kT-min(landscape_fonehalf_kT)
    plt.subplot(2,1,1)
    # add in the offsets, since we dont simulate before...
    plt.plot(ToX(ext_fwd),landscape_rev_kT+20,color='r',alpha=0.6,
             linestyle='-',linewidth=3,label="Bi-directional")
    plt.plot(ToX(ext_rev),landscape_fwd_kT+75,color='b',
             linestyle='--',label="Only Forward")
    plt.plot(ToX(landscape_rev_only.Extensions),landscape_rev_only_kT+20,
             "g--",label="Only Reverse")
    plt.ylim([0,300])
    xlim()
    PlotUtilities.lazyLabel("","Free Energy (kT)",
                            "Hummer 2010, Figure 3")
    plt.subplot(2,1,2)
    plt.plot(ToX(ext_rev),landscape_fonehalf_kT_rel,color='r')
    plt.ylim([0,25])
    xlim()
    PlotUtilities.lazyLabel("Extension q (nm)","Energy at F_(1/2) (kT)","")


def assert_work_correct(obj,expected,atol=1e-100,rtol=1e-12):
    plt.semilogy(obj.Work)
    plt.semilogy(expected)
    plt.show()
    np.testing.assert_allclose(obj.Work,expected)

def check_iwt_obj(exp,act,**tolerance_kwargs):
    """
    checks that the 'act' iwt object matches the expected 'exp' object. kwargs
    are passed to np.testing.assert_allclose. This is a 'logical' match.
    """
    np.testing.assert_allclose(act.Time,exp.Time,**tolerance_kwargs)
    # check the refolding data matches
    np.testing.assert_allclose(act.Time,exp.Time,**tolerance_kwargs)
    # make sure the fitting set the offset and velocity propertly
    actual_params = [act.Offset,act.Velocity]
    expected_params = [exp.Offset,exp.Velocity]
    np.testing.assert_allclose(actual_params,expected_params)
    # make sure the work matches
    np.testing.assert_allclose(act.Work,exp.Work)


def TestHummer2010():
    """
    Recreates the simulation from Figure 3 of 

    Hummer, G. & Szabo, A. 
    Free energy profiles from single-molecule pulling experiments. 
    PNAS 107, 21441-21446 (2010).
    """

    # POST: fwd and reverse have the forward and reverse trajectories 
    # go ahead and made the energy landscapes
    num_bins=100
    kT = 4.1e-21
    f_one_half = 14e-12
    state_fwd,state_rev = HummerData()
    # make copy of the data; we check this below to make sure we dont 
    # mess with it
    state_fwd_o,state_rev_o = copy.deepcopy(state_fwd),copy.deepcopy(state_rev)
    landscape = InverseWeierstrass.FreeEnergyAtZeroForce(state_fwd,num_bins,[])
    landscape_both = InverseWeierstrass.\
            FreeEnergyAtZeroForce(state_fwd,num_bins,state_rev)
    landscape_rev_only = InverseWeierstrass.\
                    FreeEnergyAtZeroForce(state_rev,num_bins,[])
    plt.plot(landscape.EnergyLandscape/4.1e-21,color='b')
    plt.plot(landscape_both.EnergyLandscape/4.1e-21,color='r')
    plt.plot(landscape_rev_only.EnergyLandscape/4.1e-21,color='g')
    plt.show()
    # POST: height should be quite close to Figure 3
    fig = PlotUtilities.figure(figsize=(4,7))
    landscape_plot(landscape,landscape_both,landscape_rev_only,kT,f_one_half)
    PlotUtilities.savefig(fig,"out.png")
    check_hummer_by_ensemble(kT,landscape,landscape_both,f_one_half=f_one_half)
    # POST: ensemble works well.
    # combine all the forward and reverse states
    single = copy.deepcopy(state_fwd[0])
    single.Force = []
    single.Extension = []
    single.Time = []
    for fwd,rev in zip(state_fwd,state_rev):
        single.Force += list(fwd.Force) + list(rev.Force)
        single.Extension += list(fwd.Extension) + list(rev.Extension)
        single.Time += list(fwd.Time) + list(rev.Time)
    # combine all the data
    N = len(state_fwd)
    single.Force = np.array(single.Force)
    single.Extension = np.array(single.Extension)
    single.Time = np.array(single.Time)
    kwargs = dict(number_of_pairs=N,
                  flip_forces=False,
                  fraction_for_vel=0.3,kT=4.1e-21)
    unfold,refold = WeierstrassUtil.get_unfold_and_refold_objects(single,
                                                                  **kwargs)
    tolerance_kwargs = dict(atol=0,rtol=1e-6)
    for un,re,un_org,re_org in zip(unfold,refold,state_fwd,state_rev):
        check_iwt_obj(un_org,un,**tolerance_kwargs)
        check_iwt_obj(re_org,re,**tolerance_kwargs)
    # make sure we didn't mess with the 'original', generated data
    # (for the purposes of IWT)
    landscape_rev_2 = InverseWeierstrass.\
            FreeEnergyAtZeroForce(state_fwd,num_bins,state_rev)
    np.testing.assert_allclose(landscape_rev.EnergyLandscape,
                               landscape_rev_2.EnergyLandscape)
    # POST:the new landscape matches the original one. make sure the data is ok
    for fwd,rev,fwd_orig,rev_orig in \
        zip(state_fwd,state_rev,state_fwd_o,state_rev_o):
        check_iwt_obj(fwd,fwd_orig,**tolerance_kwargs)
        check_iwt_obj(rev,rev_orig,**tolerance_kwargs)
    # POST: should be able to get the same landscape; data havent been corrupted
    # check that the sliced data is OK. 
    landscape_bidirectional = InverseWeierstrass.\
        FreeEnergyAtZeroForce(state_fwd,num_bins,state_rev)
    np.testing.assert_allclose(landscape_rev.EnergyLandscape,
                               landscape_bidirectional.EnergyLandscape)
    # check that the command-line style calling works 
    expected_landscape = landscape_rev.EnergyLandscape
    assert_correct = lambda actual: \
        np.testing.assert_allclose(actual.EnergyLandscape,
                                   expected_landscape)
    cmd_line = WeierstrassUtil.iwt_ramping_experiment(single,
                                                      number_of_bins=num_bins,
                                                      **kwargs)
    assert_correct(cmd_line)     
    # check that we get the same answer when we flip the data, and ask it
    # to be flipped
    single_rev = copy.deepcopy(single)
    single_rev.Force *= -1
    kwargs_flipped = dict(**kwargs)
    kwargs_flipped['flip_forces'] = True
    cmd_line_flipped = WeierstrassUtil.\
        iwt_ramping_experiment(single_rev,
                               number_of_bins=num_bins,
                               **kwargs_flipped)
    assert_correct(cmd_line_flipped)    
    # check that we get the same answer if we specify the velocity parameter
    cmd_line_velocity = WeierstrassUtil.\
                        iwt_ramping_experiment(single,
                                               number_of_bins=num_bins,
                                               velocity=single.Velocity,
                                               **kwargs)
    assert_correct(cmd_line_velocity)    
    # check that we get an incorrect answer if we mess up the velocity 
    cmd_line_incorrect = WeierstrassUtil.\
        iwt_ramping_experiment(single,
                               velocity=single.Velocity*2,
                               number_of_bins=num_bins,
                               **kwargs)
    # the first point will be zero; after that the one with wonky velocity
    # shouldn't agree                                
    assert_landscapes_disagree(cmd_line_incorrect,expected_landscape)
    # check that if we only flip one, things are also bad 
    cmd_line_incorrect = WeierstrassUtil.\
        iwt_ramping_experiment(single_rev,
                               number_of_bins=num_bins,
                               **kwargs)    
    assert_landscapes_disagree(cmd_line_incorrect,expected_landscape)
                               
    
def assert_landscapes_disagree(new_obj,expected_landscape):
    # the landscapes are normalized to zero; so we ignore the first point 
    assert ((new_obj.EnergyLandscape[1:] != expected_landscape[1:]).all())
                                               
    
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

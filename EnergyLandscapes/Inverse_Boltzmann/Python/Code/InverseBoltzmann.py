# force floating point division. Can still use integer with //
from __future__ import division
# other good compatibility recquirements for python3
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
import scipy

def inverse_boltzmann(P_q,kT=4.1e-21):
    """
    returns the inverse boltzman; given a probability of being at a given
    extension 

    Args:
        P_q: the probability at each extension
        kT: boltzmann's constant
    
    Returns:
        G_eqm(q), from Gupta, A. N. et al, Nat Phys 7, 631-634, 2011.
    """
    return -kT * np.ln(probability_of_extension)

def deconvolution_iteration(S_q,P_q,r_0=1,p_k=None):
    """
    See: Gebhardt, J. et al., PNAS 107, 2013-2018 (2010). SI,  pp2,
    'Deconvolution procedure.'

    See also: Woodside, M. T. et al. Science, 2006. SI, pp3-4,
    'Deconvolution procedure.', eq. 2 (where the notation is somewhat clearer)
    
    Args:
        S_q: the smoothing function at each q, size N
        P_q: the measured probability at each q, size N
        r_0: the convergence amplitude        
        p_k: the current 'true' distribution, size N. if None, XXX
    Returns:
        p_(k+1) in eq 2 of Woodside, Science,2006, SI, Eq.2 (pp4)
    """
    r = r_0 * (1 - 2*np.abs(p_k-1/2))
    S_q_convolved_with_p_k = fftconvolve(p_k,S_q,mode='same')
    p_k_plus_one =  p_k + r * (P_q - S_q_convolved_with_p_k) 
    p_k_plus_one = np.maximum(0,p_k_plus_one)
    return p_k_plus_one


def f_assert_prob(x,msg):
    """
    assets that a given probability density (in units of 1/<something>) is 
    reasonable. specifically, checks the following:

    (1) Is every element greater than zero 
    (2) Is the sum between (0,inf]? Since units aren't given, we can't 
    assume it will sume to 1 (but it should integrate...)

    Args:
        x: the probability distribution to check
        msg: the message to pring on an error
    Returns:
        nothing, throws an error if something goes wrong
    """
    # make sure every element is between 0 and 1
    msg = msg + " Offending Array:\n{:s}".format(str(x))
    is_finite = np.isfinite(x).all()
    assert (is_finite) , msg
    # POST: is finite 
    assert ( (x >= 0) ).all() , msg
    # since we dont know 'x', we only know the probabilities should
    # sum to somewhere >0. (e.g. if probability is in units of 
    # 1/meters, we would need to integrate to get 1; we can't just sum)
    sum_x = sum(x) 
    assert ( (sum_x > 0)).all() , msg
    

def deconvolve(p_0,S_q,P_q,r_0=1,n_iters=50,delta_tol=1e-6,return_full=False):
    """
    deconvolve the probability distrubtion until whichever is first:

    - n_iters have been performed
    - the change from iteration i to (i+1) is less than delta_tol for
    *all* probability elements

    Args:
        p_0: initial guess. 
        r_0, S_q, P_q: see deconvolution_iteration
        n_iters: maximum number of iterations to perform
        delta_tol: if all probabilities change less than delta_tol, we quit
        return_full: if true, returns more detailed information

        **kwargs: passed to deconvolution_iteration; but should not specify p_k
    Returns:
        if return_full, tuple of (final probability, list of all probabilities).
        Otherwise, just final probability
    """
    # check the inputs
    f_assert_prob(p_0,"Initial guess (p_0) not normalized (in [0,1]).")
    f_assert_prob(S_q,"Smoothing func (S_q) not normalized (in [0,1]).")
    f_assert_prob(P_q,\
        "Measured distribution (P_q) not normalized (in [0,1]).")
    assert (r_0 > 0) , "Convergence rate must be >0"
    assert p_0.size == S_q.size , \
        "Smoothing function (S_q) must be same size as guess (p_0)"
    assert p_0.size == P_q.size , \
        "Measured distribution (P_q) must be same size as guess (p_0)"
    # POST: probabilities should be OK. should be able to run.
    all_probs = [p_0]
    p_k = p_0
    # iteratively update the probability 
    for i in range(n_iters):
        p_next = deconvolution_iteration(p_k=p_k,S_q=S_q,P_q=P_q,r_0=r_0)
        error = np.abs(p_next - p_k)
        p_k = p_next
        if return_full:
            # append the current iteration
            all_probs.append(p_k)
        if (error < delta_tol).all():
             # then the error condition has been reached
             break
        assert (p_k >= 0).all() , \
            "Deconvolution error, p_k histogram became negative. Check XXX"

    if (return_full):
        return p_k,all_probs
    else:
        return p_k
        
def gaussian_psf(gaussian_stdev,extension_bins,loc=None):
    """
    returns a gaussian point-spread function 

    Args:
         gaussian_stdev: the standard deviation
         extension_bins: where we want the psf
         loc: the mean of the gaussian. defaults to <extension_bins>
    Returns:
         Probability distribution S_q at each point in extension_bins
    """
    if (loc is None):
        loc = np.mean(extension_bins)
    S_q = scipy.stats.norm.pdf(extension_bins,loc=loc,scale=gaussian_stdev)
    return S_q

def gaussian_deconvolve(gaussian_stdev,extension_bins,P_q,p_0=None,**kwargs):
    """
    Returns the deconvolution of P_q with a gaussian of a set width centered 
    at the middle of extension_bins.
    
    If something goes wrong, throws an assertion error 
    
    Args:
        gaussian_stdev_meters: the standard deviation of the gaussian, same 
        units as extension
        
        extension_bins: the x values corresponding to P_q
        
        P_q: see deconvolution_iteration
        
        **kwargs: passed to deconvolve
        
    Returns:
        the normalized, deconvoluted probability at each extension 
    """
    S_q = gaussian_psf(gaussian_stdev,extension_bins)
    if (p_0 is None):
        p_0 = np.ones(S_q.size)     
    p_0 /= np.trapz(y=p_0,x=extension_bins)
    p_k = deconvolve(p_0=p_0,S_q=S_q,P_q=P_q,**kwargs)
    is_finite = (sum(np.isfinite(p_k)) == p_k.size)
    is_positive = (p_k >= 0).all()
    is_normalizable = (sum(p_k) > 0)
    assert is_finite , "Deconvolution resulted in infinite sum."
    assert is_positive , \
        "Deconvolution resulted in negative probability distribution."
    assert is_normalizable , \
        "Couldn't normalize the probability distribution (sum was 0)."
    # POST: p_k is normalizable 
    # re-normalize.
    p_k /= np.trapz(y=p_k,x=extension_bins)
    return p_k   

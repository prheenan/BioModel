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
    assert ( (x >= 0) ).all() , msg
    # since we dont know 'x', we only know the probabilities should
    # sum to somewhere >0. (e.g. if probability is in units of 
    # 1/meters, we would need to integrate to get 1)
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
    f_assert_prob(p_0,"Initial guess p_0 not normalized properly (in [0,1])")
    f_assert_prob(S_q,"Smoothing func S_q not normalized properly (in [0,1])")
    f_assert_prob(P_q,
                  "Measured probability P_q not normalized properly (in [0,1])")
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
        if return_full:
            # append the current iteration
            all_probs.append(p_next)
        if (error < delta_tol).all():
             # then the error condition has been reached
             break
        p_k = p_next
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

def gaussian_deconvolve(gaussian_stdev,extension_bins,P_q,**kwargs):
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
    p_0 = np.ones(S_q.size)
    p_0 /= np.trapz(y=p_0,x=extension_bins)
    p_k = deconvolve(p_0=p_0,S_q=S_q,P_q=P_q,**kwargs)
    # make sure the probability we found is valid 
    assert (np.where(np.isfinite(p_k))[0].size == p_k.size) and \
           (sum(p_k) > 0) , \
           "Deconvolution error resulted in non-normalizable sum"
    # POST: p_k is normalizable 
    # re-normalize.
    p_k /= np.trapz(y=p_k,x=extension_bins)
    return p_k   

def interpolate_and_deconvolve_gaussian_psf(gaussian_stdev,extension_bins,P_q,
                                            interpolate_kwargs=dict(),
                                            **deconvolve_kwargs):
    """
    Ease-of-use function for deconvolving a gaussian point-spread function.

    Args:
         see gaussian_deconvolve, except...
         interpolate_kwargs: passed to get_interpolated_probability
    Returns:
         tuple of <interpolated ext, interpolated probability, deconvolved and
         interpolated probability> 
    """
    # get the interpolated probabilities
    interp_ext,interp_prob = get_interpolated_probability(ext=extension_bins,
                                                          raw_prob=P_q,
                                                          **interpolate_kwargs)
    deconv_interpolated_probability = \
                gaussian_deconvolve(extension_bins=interp_ext,
                                    P_q=interp_prob,
                                    gaussian_stdev=gaussian_stdev,
                                    **deconvolve_kwargs)
    return interp_ext,interp_prob,deconv_interpolated_probability

def get_extension_bins_and_distribution(extension,bins):
    """
    returns the (normalized) probability ditribution of extension bini

    Args:
         extension: array to digitize
         bins: passed to np.histogram
    Returns:
         tuple of <left side of bins, histogram distribution> 
    """
    distribution,bins = np.histogram(a=extension,bins=bins,normed=True)
    bins = bins[:-1]
    return bins,distribution


def get_interpolated_probability(ext,raw_prob,
                                 upscale_factor=10,kind='linear',
                                 interp_ext=None,**kwargs):
    """
    returns an interpolated probability (possibly on a different x grid)

    Args:
        ext: the x values for raw_prob, size N 
        raw_prob: the y values to interpolate, size N
        upscale_factor: if interp_ext is none, just gets this many more points
        linearly, interpolated along ext (so size upscale_factor* N). If 
        no interpolation is desired, just set this to one
    
        interp_ext: if not none, the grid to interpolate along. 

        kind,**kwargs: passed to interp1d
      
    """
    if (interp_ext is None):
        if (upscale_factor > 1):
            # we have something to do!
            interp_ext = np.linspace(start=min(ext),
                                     stop=max(ext),
                                     num=ext.size*upscale_factor)
        else:
            # no interpolation desired
            interp_ext = ext
    interp_smoothed_prob_f = scipy.interpolate.interp1d(x=ext, 
                                                        y=raw_prob, 
                                                        kind=kind,**kwargs)
    # get the probability at each extension, normalize
    interp_smoothed_prob = interp_smoothed_prob_f(interp_ext)
    interp_smoothed_prob = np.maximum(0,interp_smoothed_prob)
    interp_smoothed_prob /= np.trapz(y=interp_smoothed_prob,x=interp_ext)
    return interp_ext,interp_smoothed_prob

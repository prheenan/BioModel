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

def deconvolution_iteration(r_0,S_q,P_q,p_k=None):
    """
    See: Gebhardt, J. et al., PNAS 107, 2013-2018 (2010). SI,  pp2,
    'Deconvolution procedure.'

    See also: Woodside, M. T. et al. Science, 2006. SI, pp3-4,
    'Deconvolution procedure.', eq. 2 (where the notation is somewhat clearer)
    
    Args:
        r_0: the convergence amplitude
        S_q: the smoothing function at each q, size N
        P_q: the measured probability at each q, size N
        p_k: the current 'true' distribution, size N. if None, XXX
    Returns:
        p_(k+1) in eq 2 of Woodside, Science,2006, SI, Eq.2 (pp4)
    """
    r = r_0 * (1 - 2*np.abs(p_k-1/2))
    S_q_convolved_with_p_k = fftconvolve(p_k,S_q,mode='same')
    p_k_plus_one =  p_k + r * (P_q - S_q_convolved_with_p_k)
    return p_k_plus_one


def f_assert_prob(x,msg):
    """
    assets that a given probability density (in units of 1/<something>) is 
    reasonable. specifically, checks the following:

    (1) Is every element between 0 and 1
    (2) Is the sum between (0,inf]? Since units aren't given, we can't 
    assume it will sume to 1 (but it should integrate...)

    Args:
        x: the probability distribution to check
        msg: the message to pring on an error
    Returns:
        nothing, throws an error if something goes wrong
    """
    # make sure every element is between 0 and 1
    assert ( (x >= 0) & (x <= 1)).all() , msg
    # since we dont know 'x', we only know the probabilities should
    # sum to somewhere >0. (e.g. if probability is in units of 
    # 1/meters, we would need to integrate to get 1)
    sum_x = sum(x) 
    assert ( (sum_x > 0)).all() , msg
    

def deconvolve(p_0,r_0,S_q,P_q,n_iters=50,delta_tol=1e-6,return_full=False):
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

def woodside_2006_smoothing_function(extensions,n=1000):
    """
    Returns the woodside 2006 smoothing function

    Args:
        extensions: the extensions at which the smoothing function is desired
        n: the number of points to get
    Returns: 
        the woodside_2006 Science paper smoothing function
    """
    # use Woodside, M. T. et al. Science, 2006. SI, Figure S2 for the PSF
    # Since we are convolving, (I think) the average doesnt matter
    woodside_mu = 513.6
    woodside_fwhm = (515.5-512)
    # see: https://en.wikipedia.org/wiki/Full_width_at_half_maximum
    woodside_stdev = woodside_fwhm/2.355
    n_bins = extensions.size
    loc = np.mean(extensions)
    scale = woodside_stdev
    S_q = scipy.stats.norm.pdf(extensions, loc=loc, scale=scale)
    return S_q

def test_smoothing(mean_fwhm_weights,max_nm=30,step_nm=0.5):
    extensions = np.arange(0,max_nm,step=step_nm)
    probabilities = [scipy.stats.norm.pdf(extensions,
                                          loc=mu, scale=fwhm/2.355)
                     for mu,fwhm,_ in mean_fwhm_weights]
    weights = [w[-1] for w in mean_fwhm_weights]
    full_pdf = np.average(probabilities,axis=0,weights=weights)
    full_pdf /= np.trapz(y=full_pdf,x=extensions)
    probability_normalized = np.maximum(0,full_pdf)
    probability_normalized = np.minimum(1,probability_normalized)
    P_q  = probability_normalized
    # XXXX fix extensions = probability_grid.size
    S_q = woodside_2006_smoothing_function(extensions=extensions)
    r_0 = 2
    p_0 = np.ones(P_q.size)
    p_0 /= np.trapz(y=p_0,x=extensions)
    iterations = 1000
    p_final,p_list = deconvolve(p_0=p_0,
                                r_0=r_0,S_q=S_q,P_q=P_q,n_iters=iterations,
                                return_full=True,delta_tol=1e-6)
    #... renormalize probability? XXX figure out why this is necessary
    p_final /= np.trapz(y=p_final,x=extensions)
    extension_grid_plot = extensions-min(extensions)
    # determine the ratios
    split_point_nm = 15
    split_idx = np.where(extensions>split_point_nm)[0][0]
    split_ratios = [np.max(p_final[:split_idx])/np.max(P_q[:split_idx]),
                    np.max(p_final[split_idx:])/np.max(P_q[split_idx:])]
    # XXX check that the split ratios are close to what we expect. 
    plt.plot(extension_grid_plot,S_q,'b:')
    plt.plot(extension_grid_plot,p_final,'g--')
    plt.plot(extension_grid_plot,P_q,linewidth=3,color='r')
    plt.show()    

def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    # use Woodside, M. T. et al. Science, 2006. FIgure S3B for the extension
    # histogram.
    mean_fwhm_nm = [ [10,4,1], 
                     [22.5,4.25,1.6]]
    test_smoothing(mean_fwhm_nm)


if __name__ == "__main__":
    run()

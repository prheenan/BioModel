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
        
def gaussian_psf(gaussian_stdev,extension_bins):
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

def test_smoothing(mean_fwhm_weights,split_ratios,max_nm=None,step_nm=0.5,
                   split_point_nm = 15,fwhm_smoothing=3.5):
    """
    given a number of distributions, runs the deconvolution algorithm on them 
    """
    if (max_nm is None):
        max_nm = max([mu+3*fwhm for mu,fwhm,_ in mean_fwhm_weights])
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
    r_0 = 2
    p_0 = np.ones(P_q.size)
    p_0 /= np.trapz(y=p_0,x=extensions)
    iterations = 1000
    gaussian_stdev = fwhm_smoothing/2.355
    S_q = gaussian_psf(gaussian_stdev=gaussian_stdev,extension_bins=extensions)
    deconvolve_kwargs = dict(gaussian_stdev=gaussian_stdev,
                             extension_bins=extensions,
                             n_iters=iterations,
                             return_full=False,
                             delta_tol=1e-9,
                             r_0=r_0,
                             P_q=P_q)
    p_final = gaussian_deconvolve(**deconvolve_kwargs)
    extension_grid_plot = extensions-min(extensions)
    # determine the ratios
    split_idx = np.where(extensions>split_point_nm)[0][0]
    pred_split_ratios = [np.max(p_final[:split_idx])/np.max(P_q[:split_idx]),
                         np.max(p_final[split_idx:])/np.max(P_q[split_idx:])]
    np.testing.assert_allclose(pred_split_ratios,split_ratios,atol=0,rtol=1e-3)
    

def read_ext_and_probability(input_file):
    arr = np.loadtxt(input_file,delimiter=",")
    ext,raw_prob = arr[:,0],arr[:,1]
    sort_idx = np.argsort(ext)
    ext = ext[sort_idx]
    raw_prob = raw_prob[sort_idx]
    return ext,raw_prob

def run():
    """
    <Description>

    Args:
        param1: This is the first param.
    
    Returns:
        This is a description of what is returned.
    """
    # # use Woodside, M. T. et al. Science, 2006. FIgure 3 for all the tests
    # test figure 3a
    mean_fwhm_nm_fig3a = [ [11,7,1.2], 
                           [25,5,1]]
    deconv_ext,deconv_prob = read_ext_and_probability("woodside_2006_3a.csv")
    ext,raw_prob = \
        read_ext_and_probability("woodside_2006_3a_raw_probability.csv")
    interp_ext = np.linspace(min(ext),max(ext),ext.size*10)
    f_interp_prob = scipy.interpolate.interp1d(x=ext, y=raw_prob, 
                                               kind='linear')
    interp_prob = f_interp_prob(interp_ext)
    interp_smoothed_prob = scipy.interpolate.interp1d(x=deconv_ext, 
                                                      y=deconv_prob, 
                                                      kind='linear')(interp_ext)
    interp_smoothed_prob = np.maximum(0,interp_smoothed_prob)
    interp_smoothed_prob /= np.trapz(y=interp_smoothed_prob,x=interp_ext)
    # normalize probability
    interp_prob = np.maximum(0,interp_prob)
    interp_prob /= np.trapz(y=interp_prob,x=interp_ext)
    # 
    deconvolve_kwargs = dict(gaussian_stdev=5.5/2.355,
                             extension_bins=interp_ext,
                             n_iters=300,
                             return_full=False,
                             delta_tol=1e-9,
                             r_0=1,
                             P_q=interp_prob)
    p_final = gaussian_deconvolve(**deconvolve_kwargs)
    n_points_1_nm = int(np.ceil(1/(interp_ext[1]-interp_ext[0])))
    if (n_points_1_nm % 2 == 0):
        n_points_1_nm += 1
    p_final_filtered = scipy.signal.medfilt(p_final, 
                                            kernel_size=n_points_1_nm)
    # renormalize... 
    p_final_filtered /= np.trapz(y=p_final_filtered,x=interp_ext)
    diff = np.abs(p_final_filtered - interp_smoothed_prob)
    diff_rel = diff
    print(np.percentile(diff_rel,95))
    plt.hist(diff_rel)
    plt.xscale('log')
    plt.show()
    # plot everything
    plt.plot(ext,raw_prob,'ro')
    plt.plot(interp_ext,interp_prob,'b')
    plt.plot(interp_ext,interp_smoothed_prob,'b')
    plt.plot(interp_ext,p_final_filtered)
    plt.show()

if __name__ == "__main__":
    run()

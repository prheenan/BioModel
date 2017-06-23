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
    S_q_convolved_with_p_k = fftconvolve(S_q,p_k,mode='same')
    p_k_plus_one =  p_k + r * (P_q - S_q_convolved_with_p_k)
    return p_k_plus_one


def deconvolve(p_0,n_iters=50,delta_tol=1e-6,return_full=False,**kwargs):
    """
    deconvolve the probability distrubtion until whichever is first:

    - n_iters have been performed
    - the change from iteration i to (i+1) is less than delta_tol for
    *all* probability elements

    Args:
        p_0: initial guess. 
        n_iters: maximum number of iterations to perform
        delta_tol: if all probabilities change less than delta_tol, we quit
        return_full: if true, returns more detailed information

        **kwargs: passed to deconvolution_iteration; but should not specify p_k
    Returns:
        if return_full, tuple of (final probability, list of all probabilities).
        Otherwise, just final probability
    """
    all_probs = [p_0]
    p_k = p_0
    # iteratively update the probability 
    for i in range(n_iters):
        p_next = deconvolution_iteration(p_k=p_k,**kwargs)
        error = np.abs(p_next - p_k)
        if return_full:
            # append the current iteration
            all_probs.append(p_next)
        if (error < delta_tol).all():
             # then the error condition has been reached
             break
        p_k = p_next
    if (return_full):
        return p_k,all_probs
    else:
        return p_k

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
    extension_probability_tuple = [ [0,0],
                                    [5,0.01],
                                    [7,0.04],
                                    [10,0.08],
                                    [13,0.04],
                                    [15,0],
                                    [17,0.01],
                                    [20,0.05],
                                    [22,0.12],
                                    [25,0.05],
                                    [27,0.01],
                                    [30,0]]
    extensions_nm = [e[0] for e in extension_probability_tuple]
    probability_unnormalized = [ e[1] for e in extension_probability_tuple]
    f2 = interp1d(extensions_nm,probability_unnormalized, kind='cubic')
    # get a better plot of probability vs extension
    n_points = 50
    extension_grid = np.linspace(min(extensions_nm),max(extensions_nm),
                                 n_points)
    probability_grid = f2(extension_grid)
    # fit a spline to the probability vs extension 
    probability_integral = np.trapz(y=probability_grid,
                                    x=extension_grid)
    probability_normalized = probability_grid/probability_integral
    probability_normalized = np.maximum(0,probability_normalized)
    probability_normalized = np.minimum(1,probability_normalized)
    plt.plot(extension_grid,probability_normalized,'b--')
    plt.show()
    P_q  = probability_normalized
    # use Woodside, M. T. et al. Science, 2006. SI, Figure S2 for the PSF
    # Since we are convolving, (I think) the average doesnt matter
    woodside_mu = 513.6
    woodside_fwhm = abs(woodside_mu-512.25)
    woodside_stdev = woodside_fwhm/2.355
    n = 1000
    n_bins = probability_grid.size
    s_psf = np.random.normal(woodside_mu,woodside_stdev,n_bins)
    S_q, _ = np.histogram(s_psf, n_bins, normed=True)
    r_0 = 1
    p_0 = np.ones(P_q.size) / P_q.size
    iterations = 1000
    p_final,p_list = deconvolve(p_0=p_0,return_full=True,delta_tol=1e-6,
                                r_0=r_0,S_q=S_q,P_q=P_q)
    #... renormalize probability? XXX figure out why this is necessary
    p_final /= np.trapz(y=p_final,x=extension_grid)
    extension_grid_plot = extension_grid-min(extension_grid)
    for p_intermediate in p_list:
        plt.plot(extension_grid_plot,p_intermediate)
    plt.plot(extension_grid_plot,p_final,'g--')
    plt.plot(extension_grid_plot,P_q,linewidth=3,color='r')
    plt.show()

if __name__ == "__main__":
    run()

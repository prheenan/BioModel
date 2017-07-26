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

from FitUtil.EnergyLandscapes.Inverse_Boltzmann.Python.Code import \
    InverseBoltzmann

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

def normalize_to_sum_1(extension,gaussian_stdev,bins):
    """
    normalize the given quantitites such that the distribution has both sum and 
    integral one

    Args
        extension: size N, same units as gaussian_stdev
        gaussian_stdev: the standard deviation of the gaussian psf
        bins: the input to np.histogram

    Returns:
        tuple of <factor extension should be multiplied by,extension multiplied,
        1-normalized bins, 1-normalized gaussian stdev,
        1-normalized extension distribution P(q), 
    """
    # get the extension distribution in whatever units the user gives us
    bins,P_q = get_extension_bins_and_distribution(extension,bins=bins)
    sum_initial = sum(P_q)
    # choose bins such that the sum is 1
    extension_factor = sum_initial
    # XXX assume p0...
    extension_unitless = extension*extension_factor
    bins *= extension_factor
    gaussian_stdev *= extension_factor
    P_q /= np.trapz(y=P_q,x=bins)
    return extension_factor,extension_unitless,bins,gaussian_stdev,P_q
    

def extension_deconvolution(gaussian_stdev,extension,bins,
                            interpolate_kwargs = dict(),
                            deconvolve_common_kwargs=dict(p_0=None,
                                                          n_iters=300,
                                                          delta_tol=1e-9,
                                                          return_full=False,
                                                          r_0=1)):
    """
    deconvolves an extension vs time trace

    Args:
        gaussian_stdev: of the (assumed gaussian) point-spread function
        extension: extension vs time, size N
        bins: passed to get_extension_bins_and_distribution (and np.histogram).
        Can be a number, of list of left bin edges

        interpolate_kwargs: see 
        InverseBoltzmann.interpolate_and_deconvolve_gaussian_psf

        deconvolve_common_kwargs: passed as kwargs to 
        interpolate_and_deconvolve_gaussian_psf
    
    Returns:
        tuple of <interpolated extensions, raw probability, 
        deconvolved probability>
    """
    extension_factor,extension_unitless,bins,gaussian_stdev,P_q = \
        normalize_to_sum_1(extension,gaussian_stdev,bins)
    deconvolve_kwargs = dict(gaussian_stdev=gaussian_stdev,
                             extension_bins = bins,
                             P_q = P_q,
                             interpolate_kwargs=interpolate_kwargs,
                             **deconvolve_common_kwargs)
    interp_ext,interp_prob,deconv_interpolated_probability = \
        InverseBoltzmann.\
        interpolate_and_deconvolve_gaussian_psf(**deconvolve_kwargs)
    # convert the extensions back from their unnormalized format, renormalize 
    # the probabilities so that they match up
    interp_ext = interp_ext * 1/extension_factor
    # 'raw' probability
    interp_prob /= np.trapz(x=interp_ext,y=interp_prob)
    # deconvolved probability 
    factor_deconv = np.trapz(x=interp_ext,y=deconv_interpolated_probability)
    deconv_interpolated_probability /= factor_deconv
    return interp_ext,interp_prob,deconv_interpolated_probability

def run_and_save_data(gaussian_stdev,extension,bins,out_file,
                      interpolate_kwargs=dict(),
                      save_kwargs=dict(fmt=str("%.15g"))):
    """
    Runs a deconvolution, saving the data out 
    """
    interp_ext,interp_prob,prob_deconc = \
            extension_deconvolution(gaussian_stdev,
                                    extension,bins,
                                    interpolate_kwargs=interpolate_kwargs)
    header = "# extension bin -- raw probability -- deconvolved probability" + \
             " (Inverse Boltzmann, (c) Patrick Heenan 2017)"
    X = np.array(((interp_ext,interp_prob,prob_deconc))).T
    np.savetxt(fname=out_file,X=X,**save_kwargs)

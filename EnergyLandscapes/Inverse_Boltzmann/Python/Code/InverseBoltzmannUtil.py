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
    # get the extension distribution in whatever units the user gives us
    bins,P_q = get_extension_bins_and_distribution(extension,bins=bins)
    # XXX assume we know initial guess...
    p_0 = np.ones(P_q.size)
    sum_initial = sum(p_0)
    # get the normalized p_0
    p_0_normalized = p_0/np.trapz(y=p_0,x=bins)
    # determine what p_0 will then sum to
    p_0_sum = sum(p_0_normalized)
    # choose bins such that the sum is 1
    extension_factor = p_0_sum
    # XXX assume p0...
    extension_unitless = extension*extension_factor
    bins *= extension_factor
    gaussian_stdev *= extension_factor
    P_q /= np.trapz(y=P_q,x=bins)
    deconvolve_kwargs = dict(gaussian_stdev=gaussian_stdev,
                             extension_bins = bins,
                             P_q = P_q,
                             interpolate_kwargs=interpolate_kwargs,
                             **deconvolve_common_kwargs)
    interp_ext,interp_prob,deconv_interpolated_probability = \
        InverseBoltzmann.\
        interpolate_and_deconvolve_gaussian_psf(**deconvolve_kwargs)
    # convert the extensions back to their unit-less format, and renormalize the
    # probabilities so that they match up
    interp_ext = interp_ext * 1/extension_factor
    # 'raw' probability
    interp_prob /= np.trapz(x=interp_ext,y=interp_prob)
    # deconvolved probability 
    factor_deconv = np.trapz(x=interp_ext,y=deconv_interpolated_probability)
    deconv_interpolated_probability /= factor_deconv
    return interp_ext,interp_prob,deconv_interpolated_probability

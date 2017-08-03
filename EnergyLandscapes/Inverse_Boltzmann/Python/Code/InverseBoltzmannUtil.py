# force floating point division. Can still use integer with //
from __future__ import division
# other good compatibility recquirements for python3
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys,scipy

from FitUtil.EnergyLandscapes.Inverse_Boltzmann.Python.Code import \
    InverseBoltzmann

def get_extension_bins_and_distribution(extension,bins):
    """
    returns the (normalized) probability ditribution of extension bins

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
                             interp_kwargs=interpolate_kwargs,
                             **deconvolve_common_kwargs)
    interp_ext,interp_prob,deconv_interpolated_probability = \
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



def upscale_factor_by_stdev(extension_bins,gaussian_stdev,n_per_bin=25):
    """
    Returns: the maximum of 1, or n_per_bin * (size of stdev in terms of 
    the median size of extension_bins)

    Args:
        n_per_bin: how many bins should be in 1 gaussian_stdev
        others: see interpolate_and_deconvolve_gaussian_psf

    Returns:
        gaussian bins
    """
    median_bin_size = np.abs(np.median(np.diff(extension_bins)))
    return max(1,n_per_bin*(median_bin_size/gaussian_stdev))

def interpolate_and_deconvolve_gaussian_psf(gaussian_stdev,extension_bins,P_q,
                                            interp_kwargs=dict(),
                                            **deconvolve_kwargs):
    """
    Ease-of-use function for deconvolving a gaussian point-spread function.

    Args:
         see gaussian_deconvolve, except...
         interp_kwargs: passed to get_interpolated_probability
    Returns:
         tuple of <interpolated ext, interpolated probability, deconvolved and
         interpolated probability> 
    """
    # get the interpolated probabilities
    interp_ext,interp_prob = get_interpolated_probability(ext=extension_bins,
                                                          raw_prob=P_q,
                                                          **interp_kwargs)
    deconv_interpolated_probability = InverseBoltzmann.\
            gaussian_deconvolve(extension_bins=interp_ext,
                                P_q=interp_prob,
                                gaussian_stdev=gaussian_stdev,
                                **deconvolve_kwargs)
    return interp_ext,interp_prob,deconv_interpolated_probability


def get_interpolated_probability(ext,raw_prob,
                                 upscale=10,kind='linear',
                                 interp_ext=None,**kwargs):
    """
    returns an interpolated probability (possibly on a different x grid)

    Args:
        ext: the x values for raw_prob, size N 
        raw_prob: the y values to interpolate, size N
        upscale: if interp_ext is none, just gets this many more points
        linearly, interpolated along ext (so size upscale_factor* N). If 
        no interpolation is desired, just set this to one
    
        interp_ext: if not none, the grid to interpolate along. 

        kind,**kwargs: passed to interp1d
      
    """
    if (interp_ext is None):
        if (upscale > 1):
            # we have something to do!
            interp_ext = np.linspace(start=min(ext),
                                     stop=max(ext),
                                     num=ext.size*upscale)
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


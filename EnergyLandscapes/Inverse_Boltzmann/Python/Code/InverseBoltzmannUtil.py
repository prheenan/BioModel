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

def _normalization_factor_for_histogram_to_sum_1(P_q):
    """
    Returns what to multiply each extension bin by such that
    P_q will sum and integrate to one (assuming renormalization of P_q
    after multiplying the extensions

    Args:
        P_q: a probability distribution
    Returns:
        see description
    """
    # We want (1) and (2):
    # (1) Int P_q dq ~ sum_q (P(q) dq) =  1
    # (2) sum_q P_q = 1
    # it follows if we choose
    # dq -> dq * sum_q P_q
    # and enforce normaliztion (1), then
    # P_q -> P_q / (sum_q P_q)
    # so that we get 2 for free 
    return sum(P_q)

def _normalize(factor,P_q,q,*args):
    """
    See enforce_normalization_sum_1(P_q,q,*args), except 'factor' is manually
    specified

    Args:
        factor: what to multiply all the extension variables by
    Returns:
        see enforce_normalization_sum_1(P_q,q,*args)
    """
    q_ret = q * factor
    to_ret = [a*factor for a in args]
    P_q_ret = P_q/np.trapz(y=P_q,x=q_ret)
    return [factor,P_q_ret,q_ret] + to_ret
    
def enforce_normalization_sum_1(P_q,q,*args):
    """
    Ensures that (1) and (2) from _normalization_factor_for_histogram_to_sum_1
    are satisfied (ie: the probability sums and integrates to one)

    Args:
        P_q: probability distribution, size N
        q: extension, size N
        *args: any addition quantities to determine; same units as q
    Returns:
        tuple of <factor,P_q_normalized,q_normalized, all of *args normalized>
    """
    factor = _normalization_factor_for_histogram_to_sum_1(P_q)
    return _normalize(factor,P_q,q,*args)

def denormalize(factor,P_q,q,*args):
    """
    See enforce_normalization_sum_1, except returns denomalized arrays
    (ie: multiplying by 1/factor instead of factor
    """
    return _normalize(1/factor,P_q,q,*args)

def normalize_to_sum_1(bins,extension,gaussian_stdev):
    """
    normalize the given quantitites such that the distribution has both sum and 
    integral one

    Args
        extension: size N, same units as gaussian_stdev
        gaussian_stdev: the standard deviation of the gaussian psf
        bins: the input to np.histogram

    Returns:
        see  enforce_normalization_sum_1, except returns gausian_Stdev also
    """
    # get the extension distribution in whatever units the user gives us
    bins,P_q = get_extension_bins_and_distribution(extension,bins=bins)
    # very important to enforce normalization.
    to_ret =  enforce_normalization_sum_1(P_q,bins,extension,gaussian_stdev)
    return to_ret
    

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
    extension_factor,P_q_u,bins_u,extension_u,gaussian_stdev_u = \
        normalize_to_sum_1(bins,extension,gaussian_stdev)
    sum_to_check = sum(P_q_u)
    int_to_check = np.trapz(y=P_q_u,x=bins_u)
    assert abs(sum_to_check - 1) < 1e-2 , \
        "Sum-normalization didn't work, got {:.4g}, not 1".format(sum_to_check)
    assert (int_to_check-1) < 1e-2 , \
        "Int-normalization didn't work, got {:.4g}, not 1".format(int_to_check)
    # POST: everything is normalized as we want (or within X%; this shouldn't
    # cause extra numerical instability ). 
    deconvolve_kwargs = dict(gaussian_stdev=gaussian_stdev_u,
                             extension_bins = bins_u,
                             P_q = P_q_u,
                             interp_kwargs=interpolate_kwargs,
                             **deconvolve_common_kwargs)
    interp_ext,interp_prob,deconv_interpolated_probability = \
        interpolate_and_deconvolve_gaussian_psf(**deconvolve_kwargs)
    # convert the extensions back from their unnormalized format, renormalize 
    # the probabilities so that they match up
    # note: we *dont* normalize interp_ext twice (so it is '_' the first time)
    _,interp_prob,_ = \
        denormalize(extension_factor,interp_prob,interp_ext)
    _,deconv_interpolated_probability,interp_ext = \
        denormalize(extension_factor,deconv_interpolated_probability,interp_ext)
    return interp_ext,interp_prob,deconv_interpolated_probability

def smart_interpolation(extension,bins,gaussian_stdev,**kw):
    """
    Args:
        exntesion, bins: see get_extension_bins_and_distribution
        gaussian_stdev: see upscale_factor_by_stdev
    Returns: 

        the 'smart' choice of interpolation factor, given data and 
        gaussian_stdev. Useful for avoiding convolution problems. 
    """
    ext_bins,_ = get_extension_bins_and_distribution(extension,bins=bins)
    interpolation_factor = \
        upscale_factor_by_stdev(extension_bins=ext_bins,
                                gaussian_stdev=gaussian_stdev,**kw)
    return interpolation_factor

def interpolate_output(output_bins,interp_ext,interp_prob,
                       prob_deconc,**kw):
    """
    given probability distributions and two x grids, interpolates back

    Args:
         output_bins: the desired x grid
         interp_ext: the interpolated x grid
         interp_prob/prob_deconc: the interpolated (but not deconvolve) 
         probability and the interpolated and deconvolved probability
    Returns:
         a tuple of the new X, and two probability distributions
    """
    # re-calculate all the results onto the interpolated grid 
    interp_prob = scipy.interpolate.interp1d(x=interp_ext,
                                             y=interp_prob,**kw)(output_bins) 
    prob_deconc = scipy.interpolate.interp1d(x=interp_ext,
                                             y=prob_deconc,**kw)(output_bins)
    interp_ext = output_bins
    interp_prob /= np.trapz(y=interp_prob,x=interp_ext)
    prob_deconc /= np.trapz(y=prob_deconc,x=interp_ext)
    return interp_ext,interp_prob,prob_deconc

def run(gaussian_stdev,extension,bins,interpolate_kwargs=dict(),
        help_smart=True):
    """
    Returns the deconvolved...

    Args:
        see extension_deconvolution, except:
        help_smart: boolean, if true, then gets upscale (for 
        extension_deconvolution) based on the gaussian standard deviation
    Returns:
        tuple of <interpolated extension, probability, and deconvolved 
        probability>
    """
    if (help_smart):
        interpolation_factor = smart_interpolation(extension,bins,
                                                   gaussian_stdev)
        interpolate_kwargs['upscale'] = interpolation_factor
    interp_ext,interp_prob,prob_deconc = \
            extension_deconvolution(gaussian_stdev,
                                    extension,bins,
                                    interpolate_kwargs=interpolate_kwargs)
    return interp_ext,interp_prob,prob_deconc
    
def save_data(out_file,interp_ext,interp_prob,prob_deconc,output_interpolated,
              bins,delimiter=",",fmt=str("%.15g"),**kw):
    """
    saves the given data, possibly interpolating back to its original grid

    Args:
        out_file: where to save (as csv)
        interp_ext,interp_prob: the non-deconvolved bins and probability
        prob_deconc: the deconvolved probability
        output_interpolated: boolean, if true, interpolates back to bins 
        remainder: passed to savetxt
    """
    if (not output_interpolated):
        # then interpolate back to the original bins
        output_bins = np.linspace(min(interp_ext),max(interp_ext),num=bins,
                                  endpoint=True)
        interp_ext,interp_prob,prob_deconc = \
                interpolate_output(output_bins,interp_ext,interp_prob,
                                   prob_deconc)
    header = "# extension bin -- raw probability -- deconvolved probability" + \
             " (Inverse Boltzmann, (c) Patrick Heenan 2017)"
    X = np.array(((interp_ext,interp_prob,prob_deconc))).T
    np.savetxt(fname=out_file,X=X,delimiter=delimiter,fmt=fmt,**kw)
    

def run_and_save_data(gaussian_stdev,extension,bins,out_file,
                      run_kwargs=dict(interpolate_kwargs=dict()),
                      save_kwargs=dict(output_interpolated=True)):
    """
    Runs a deconvolution, saving the data out 

    Args:
        see run, except for save_kwargs, see save_data
    Returns:
        nothing
    """
    interp_ext,interp_prob,prob_deconc = run(gaussian_stdev,extension,bins,
                                             **run_kwargs)
    save_data(out_file,interp_ext,interp_prob,prob_deconc,bins=bins,
              **save_kwargs)

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


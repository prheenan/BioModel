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
sys.path.append("../../../../../")
from FitUtil.EnergyLandscapes.Inverse_Boltzmann.Python.Code import \
    InverseBoltzmann,InverseBoltzmannUtil
import scipy.stats as st
from scipy.interpolate import griddata,interp1d
from scipy.integrate import cumtrapz



def read_ext_and_probability(input_file):
    """
    given an input file with columnes like <extension,raw>,
    reads and return the columns

    Args:
        input_file: full path to the file
    Returns:
        tuple of <extension, probability>, sorted low to high by extension
    """
    arr = np.loadtxt(input_file,delimiter=",")
    ext,raw_prob = arr[:,0],arr[:,1]
    sort_idx = np.argsort(ext)
    ext = ext[sort_idx]
    raw_prob = raw_prob[sort_idx]
    raw_prob = np.maximum(0,raw_prob)
    raw_prob /= np.trapz(x=ext,y=raw_prob)
    return ext,raw_prob

def spatially_filtered_probability(x,probability,x_filter):
    """
    Returns a spatially filtered probability distribution (properly normalized)
    
    Args:
        x: the x values, size N
        probability: the y values, size N.
        x_filter: the amount to filter, same units as ex 
    Returns:
        probability median-filtered and normalized so the integral over x is one
        and all values are >= 0 
    """
    n_points_x = int(np.ceil(x_filter/(x[1]-x[0])))
    if (n_points_x % 2 == 0):
        n_points_x += 1
    p_final_filtered = scipy.signal.medfilt(probability, 
                                            kernel_size=n_points_x)
    # renormalize... 
    p_final_filtered = np.maximum(0,p_final_filtered)
    p_final_filtered /= np.trapz(y=p_final_filtered,x=x)
    return p_final_filtered

def assert_probabilities_close(actual,expected,percentiles,tolerances):
    """
    asserts that <percentiles> of <actual-expected> are <= tolerances

    Args:
        actual: what we actually want
        expected: what is expected
        percentiles: of the error distributions
        tolerances: maximum percentile error
    Returns:
        nothing, throws error if things are borked.
    """
    diff = np.abs(actual - expected)
    diff_rel = diff
    percentile_values = np.percentile(diff_rel,percentiles)
    for p,val,tol in zip(percentiles,percentile_values,tolerances):
        assert (val <= tol) , "q{:.0f} was too high (at {:.4g}, max: {:.4g})".\
            format(p,val,tol)
    return percentile_values,diff_rel

def test_single_file(base_dir,gaussian_stdev,tolerances,file_id,
                     renormalize=True):
    deconv_name = base_dir + "woodside_2006_{:s}.csv".format(file_id)
    raw_name = \
            base_dir + "woodside_2006_{:s}_raw_probability.csv".format(file_id)
    deconv_ext,deconv_prob = read_ext_and_probability(deconv_name)
    ext,raw_prob = read_ext_and_probability(raw_name)
    if (renormalize):
        f,raw_prob, ext= \
            InverseBoltzmannUtil.enforce_normalization_sum_1(raw_prob,ext)
        _,deconv_prob, deconv_ext, gaussian_stdev = \
                InverseBoltzmannUtil._normalize(f,
                                                deconv_prob,
                                                deconv_ext,
                                                gaussian_stdev)
    else:
        # f is the normalization factor; it is just one if we dont renormalize
        f = 1
    # interpolate the deconvoled probability into the raw grid
    interp_ext, interp_raw_prob =  \
        InverseBoltzmannUtil.get_interpolated_probability(ext,raw_prob)
    interp_deconvolved_ext,interp_deconvolved_prob = \
        InverseBoltzmannUtil.\
        get_interpolated_probability(deconv_ext,deconv_prob,
                                     interp_ext=interp_ext,
                                     bounds_error=False,
                                     fill_value="extrapolate")
    # # start testing things
    # test that the deconvolution matches
    common_deconvolve_kwargs = dict(gaussian_stdev=gaussian_stdev,
                                    n_iters=300,
                                    return_full=False,
                                    delta_tol=1e-9,
                                    r_0=1)
    p_final_not_interp = \
        InverseBoltzmann.gaussian_deconvolve(P_q=raw_prob,
                                             extension_bins=ext,
                                             **common_deconvolve_kwargs)
    p_final = InverseBoltzmann.gaussian_deconvolve(P_q=interp_raw_prob,
                                                   extension_bins=interp_ext,
                                                   **common_deconvolve_kwargs)
    p_final_filtered = spatially_filtered_probability(interp_ext,p_final,
                                                      x_filter=1)
    # # check that the probability returns what we want 
    pct,diff_rel = assert_probabilities_close(actual=p_final_filtered,
                                              expected=interp_deconvolved_prob,
                                              percentiles=[50,95,99],
                                              tolerances =tolerances)
    # # check that the interpolating function does exactly what we just did
    interp_ext_not_by_stdev,_,p_interp_final= \
        InverseBoltzmannUtil.\
        interpolate_and_deconvolve_gaussian_psf(extension_bins=ext,
                                                P_q=raw_prob,
                                                **common_deconvolve_kwargs)
    allclose_dict = dict(rtol=1e-9,atol=1e-20)
    assert np.allclose(p_interp_final,p_final,**allclose_dict) , \
        "Didn't properly interpolate"
    # # check that if we don't intrpolate, we get the same interp
    should_be_ext,should_be_raw_prob,should_be_p_final_not_interp= \
        InverseBoltzmannUtil.\
        interpolate_and_deconvolve_gaussian_psf(extension_bins=ext,
                                                P_q=raw_prob,
                                                interp_kwargs=dict(upscale=1),
                                                **common_deconvolve_kwargs)
    rtol = 1e-3
    np.testing.assert_allclose(should_be_ext,ext,rtol=rtol)
    np.testing.assert_allclose(should_be_raw_prob,raw_prob,rtol=rtol,atol=1e-3)
    np.testing.assert_allclose(should_be_p_final_not_interp,p_final_not_interp,
                               rtol=rtol,atol=1e-3)
    # POST: if we say upscale <= 1 , then we dont interpolate
    # # check that the interpolation works about the same if we upscale 
    # # by the standard deviation
    interpolation_factor = InverseBoltzmannUtil.\
            upscale_factor_by_stdev(extension_bins=ext,
                                    gaussian_stdev=gaussian_stdev)
    kwargs_deconvolve_up = \
            dict(interp_kwargs=dict(upscale=interpolation_factor),
                 **common_deconvolve_kwargs)
    ext_up,_,p_interp_up = InverseBoltzmannUtil.\
            interpolate_and_deconvolve_gaussian_psf(extension_bins=ext,
                                                    P_q=raw_prob,
                                                    **kwargs_deconvolve_up)
    p_interp_up_grid = interp1d(x=ext_up,y=p_interp_up)(interp_ext)
    # we have to use a somehwhat higher tolerance for this...
    np.testing.assert_allclose(p_interp_up_grid,p_final,atol=0.05,rtol=1e-2)
    # interpolate back into the 'normal' grid. 
    # # check that the file IO for the command line version works OK. 
    """
    in order to generate samples, we need to get the cdf (see:
en.wikipedia.org/wiki/Inverse_transform_sampling)
stackoverflow.com/questions/21100716/fast-arbitrary-distribution-random-sampling
    """
    bins_ext=ext.size
    # we need to shift the interpolated extension bins by ~ one half a bin,
    # so that they will represent the midpoint of the bin
    shift = ((max(interp_ext)-min(interp_ext))/bins_ext) * 0.5
    cummulative_interp_prob = cumtrapz(x=interp_ext,y=interp_raw_prob,
                                       initial=0)
    # get an interpolating inverse; goes from probabilities to x values
    interpolated_inverse = interp1d(x=cummulative_interp_prob,
                                    y=interp_ext+shift)
    # generae a bunch of uniform random numbers (probabilities)
    n = int(1e6)
    uniform = np.random.random(size=n)
    ext_random = interpolated_inverse(uniform)
    f_deconv = InverseBoltzmannUtil.extension_deconvolution
    interp_ext_2,interp_prob_2,deconv_probability_2 = \
        f_deconv(gaussian_stdev=gaussian_stdev,
                 extension=ext_random,
                 bins=bins_ext)
    interp_deconv_2 = interp1d(x=interp_ext_2,y=deconv_probability_2,
                               fill_value=0,bounds_error=False)(deconv_ext)
    pct,diff_rel = assert_probabilities_close(actual=interp_deconv_2,
                                              expected=deconv_prob,
                                              percentiles=[50,95,99],
                                              tolerances =[0.015,0.098,0.12])
    out_file = "./out.csv"
    # at first, *dont* use smart interpolation 
    run_kw = dict(interpolate_kwargs=dict(upscale=10),smart_interpolation=False)
    InverseBoltzmannUtil.run_and_save_data(gaussian_stdev,ext_random,bins_ext,
                                           out_file=out_file,
                                           run_kwargs=run_kw)
    # read it back in 
    X = np.loadtxt(out_file,skiprows=0,delimiter=",").T
    X_expected = np.array((interp_ext_2,interp_prob_2,deconv_probability_2))
    assert np.allclose(X,X_expected,**allclose_dict) , "Didn't properly save"
    # POST: properly saved; all command line stuff works just fine. 
    # for kicks, save out the extension points. Note that we divide by 
    # 1e9, to get the extension in meters
    np.savetxt("./extension_vs_time.csv",X=ext_random*1e-9)
    # now that we know things save correctly, run with the 'smart' interpolation
    # and make sure things make sense...
    n_bins_original = 30
    interp_ext_help,interp_prob_help,interp_deconv_help = \
        InverseBoltzmannUtil.run(gaussian_stdev,extension=ext_random,
                                 bins=n_bins_original,smart_interpolation=True)
    # interpolate the (nominally correct) probability back to this probability 
    _,expected_prob_help,expected_deconv_help = \
            InverseBoltzmannUtil.interpolate_output(interp_ext_help,
                                                    interp_ext_2,
                                                    interp_prob_2,
                                                    deconv_probability_2)
    np.testing.assert_allclose(interp_prob_help,expected_prob_help,
                               rtol=0.05,atol=0.01)
    # deconvolved probabilities have a much higher error tolerance...
    np.testing.assert_allclose(expected_deconv_help,
                               interp_deconv_help,rtol=0.25,atol=0.3,
                               verbose=True)
    # POST: smart interpolation works (might get a diferent answer...)
    # make sure that if we save and interpolate back to the original 
    # grid, we get something reasonable...
    save_kw = dict(output_interpolated=False)
    InverseBoltzmannUtil.run_and_save_data(gaussian_stdev,ext_random,
                                           bins=n_bins_original,
                                           out_file=out_file,
                                           run_kwargs=run_kw,
                                           save_kwargs=save_kw)
    arr = np.loadtxt(out_file,delimiter=",")
    assert arr.shape == (n_bins_original,3) , "Didn't save out properly" 
    # POST: shape is correct. get the expected landscape at these x values 
    non_interp_x = arr[:,0]
    _,expected_prob_non_interp,expected_deconv_non_interp = \
            InverseBoltzmannUtil.interpolate_output(non_interp_x,
                                                    interp_ext_2,
                                                    interp_prob_2,
                                                    deconv_probability_2,
                                                    fill_value=0,
                                                    bounds_error=False)
    np.testing.assert_allclose(expected_prob_non_interp,arr[:,1],
                               rtol=0.05,atol=0.01)
    # deconvolved probabilities have a much higher error tolerance...
    np.testing.assert_allclose(expected_deconv_non_interp,arr[:,2],
                               rtol=0.25,atol=0.3)


def run(base_dir="./Data/"):
    """
    Runs all the tests
cu    """
    # # use Woodside, M. T. et al. Science, 2006. FIgure 3 for all the tests
    # test figure 3a
    np.random.seed(42)
    tolerances = [3e-3,4.2e-2,0.087]
    kw = dict(base_dir=base_dir,tolerances=tolerances)
    # we must normalize 3b, or it won't run
    test_single_file(gaussian_stdev=2.25,file_id="3b",**kw)
    for norm in [True,False]:
        test_single_file(gaussian_stdev=1.75,file_id="3c",renormalize=norm,**kw)
        test_single_file(gaussian_stdev=2.34,file_id="3a",renormalize=norm,**kw)
        test_single_file(gaussian_stdev=3,file_id="3d",renormalize=norm,**kw)

if __name__ == "__main__":
    run()

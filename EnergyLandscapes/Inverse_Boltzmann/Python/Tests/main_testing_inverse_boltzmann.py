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
    InverseBoltzmann


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

def test_probabilities_close(actual,expected,percentiles,tolerances):

    diff = np.abs(actual - expected)
    diff_rel = diff
    percentile_values = np.percentile(diff_rel,percentiles)
    for p,val,tol in zip(percentiles,percentile_values,tolerances):
        assert (val <= tol) , "q{:.0f} was too high (at {:.4g}, max: {:.4g})".\
            format(p,val,tol)
    return percentile_values,diff_rel

def test_single_file(base_dir,gaussian_stdev,tolerances,file_id):
    deconv_name = base_dir + "woodside_2006_{:s}.csv".format(file_id)
    raw_name = \
            base_dir + "woodside_2006_{:s}_raw_probability.csv".format(file_id)
    deconv_ext,deconv_prob = read_ext_and_probability(deconv_name)
    ext,raw_prob = read_ext_and_probability(raw_name)
    # interpolate the deconvoled probability into the raw grid
    interp_ext, interp_raw_prob =  \
        InverseBoltzmann.get_interpolated_probability(ext,raw_prob)
    interp_deconvolved_ext,interp_deconvolved_prob = \
        InverseBoltzmann.get_interpolated_probability(deconv_ext,deconv_prob,
                                                      interp_ext=interp_ext,
                                                      bounds_error=False,
                                                      fill_value="extrapolate")
    deconvolve_kwargs = dict(gaussian_stdev=gaussian_stdev,
                             extension_bins=interp_ext,
                             n_iters=300,
                             return_full=False,
                             delta_tol=1e-9,
                             r_0=1,
                             P_q=interp_raw_prob)
    p_final = InverseBoltzmann.gaussian_deconvolve(**deconvolve_kwargs)
    p_final_filtered = spatially_filtered_probability(interp_ext,p_final,
                                                      x_filter=1)
    pct,diff_rel = test_probabilities_close(actual=p_final_filtered,
                                            expected=interp_deconvolved_prob,
                                            percentiles=[50,95,99],
                                            tolerances =tolerances)
    plt.hist(diff_rel)
    plt.xscale('log')
    plt.show()
    # plot everything
    plt.plot(ext,raw_prob,'ro')
    plt.plot(interp_ext,interp_raw_prob,'b')
    plt.plot(interp_ext,interp_deconvolved_prob,'b')
    plt.plot(interp_ext,p_final_filtered)
    plt.show()
    

def run(base_dir="./Data/"):
    """
    Runs all the tests
    """
    # # use Woodside, M. T. et al. Science, 2006. FIgure 3 for all the tests
    # test figure 3a
    tolerances = [3e-3,0.05,0.87]
    kw = dict(base_dir=base_dir,tolerances=tolerances)
    test_single_file(gaussian_stdev=2.34,file_id="3a",**kw)
    test_single_file(gaussian_stdev=1.75,file_id="3c",**kw)
    test_single_file(gaussian_stdev=3,file_id="3d",**kw)

if __name__ == "__main__":
    run()

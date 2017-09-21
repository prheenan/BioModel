# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../../../../../../")
from Util import Test
from Util.Test import _f_assert,HummerData,load_simulated_data
from FitUtil.EnergyLandscapes.InverseWeierstrass.Python.Code import \
    InverseWeierstrass,WeierstrassUtil

def _digitize_idx(x,bins):
    # Hummer, 2010, near S2: j is defined centered arounda q
    diff = np.diff(bins)
    assert np.testing.assert_allclose(diff[0],diff,atol=0,rtol=1e-6,
                                      err_msg="Bin sizes must be the same")
    # POST: arrays match
    return np.digitize(x=x,bins=(bins-diff/2))

def _mask_iwt_object(data_x,data_y,bins):
    bin_idx = _digitize_idx(x=data_y)

def weighted_histogram():
    pass


def _get_bins_and_digitized(x_m_abs,obj,n):
    """
    Returns: tuple of <bins,digitized_extension> for object
    """
    bins = np.linspace(min(x_m_abs),max(x_m_abs),endpoint=True,num=n)
    digitized_ext = obj._GetDigitizedGen(Bins=bins,ToDigitize=obj.Extension)
    return bins,digitized_ext

def _assert_digitization_correct(x_m_abs,n,obj):
    """
    checks that the digitization procedure works fine

    Args:
        x_m_abs: the 'absolute' x value in meters expected
        n: the number of bins to use for digitixation
        obj: the obhect to digitize 

    Returns:
        nothing, throws an error if things go wrong
    """
    bins,digitized_ext = _get_bins_and_digitized(x_m_abs,obj,n)
    combined_digitized_data = sorted(np.concatenate(digitized_ext))
    np.testing.assert_allclose(combined_digitized_data,sorted(obj.Extension)), \
        "error, digitization lost data"
    # POST: exactly the correct data points were digitized. check that 
    # they wount up in the right bin
    for i,arr in enumerate(digitized_ext):
        # data with data between bin[i] and [i+1] is binned into i.
        # endpoints: data in the last bin (N-1) should be > bin N-2
        # endpoints: data in the first bin (0) should be < bin 1
        if (i < n-1):
            assert (arr <= bins[i]).all(), "upper bound broken"
        else:
            # bin n-1 should be entirely greater than n-2
            assert (arr > bins[i-1]).all(), "upper bound broken"            
        if (i > 0):
            assert (arr >= bins[i-1]).all(), "lower bound broken"            
        else:
            # bin 0 should be entirely less than 0 
            assert (arr < bins[i+1]).all(), "lower bound broken"
    # POST: all bins work 


def run():
    fwd,rev = load_simulated_data(n=2)

if __name__ == "__main__":
    run()

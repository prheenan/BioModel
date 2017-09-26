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
    InverseWeierstrass,WeierstrassUtil,WeightedHistogram


def assert_all_digitization_correct(objs):
    for o in objs:
        _assert_digitization_correct(o)

def run():
    fwd,rev = load_simulated_data(n=2)
    assert_all_digitization_correct(fwd)
    assert_all_digitization_correct(rev)

if __name__ == "__main__":
    run()

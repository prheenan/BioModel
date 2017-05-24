# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../../../../../")
from FitUtil.WormLikeChain.Python.TestExamples import util




def run():
    """
    Runs some unit testing on the WLC timing
    """
    test_obj = util.dsDNA_example_data()
    params_fit = dict([ [k,v] for k,v in test_obj.params_values.items()
                        if k != L0])
    util.get_fitting_parameters_with_noise(params_fit=params_fit)

if __name__ == "__main__":
    run()

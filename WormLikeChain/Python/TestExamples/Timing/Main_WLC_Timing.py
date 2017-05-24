# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../../../../../")
from FitUtil.WormLikeChain.Python.TestExamples import util

import cProfile


def run():
    """
    Runs some unit testing on the WLC timing
    """
    test_obj = util.GetBouichatData(StepInNm=0.3)
    params_fit = dict([ [k,v] for k,v in test_obj.param_values.items()
                        if k != 'L0'])
    L0 = test_obj.param_values['L0']
    ranges = [ [L0/2,L0*2]]
    ext_pred,force_grid = util.get_ext_and_force(test_obj)
    pr = cProfile.Profile()
    pr.enable()
    util.get_fitting_parameters_with_noise(ext_pred=ext_pred,
                                           force_grid=force_grid,
                                           params_fit=params_fit,
                                           ranges=ranges,
                                           noise_amplitude_N=10e-12)
    pr.disable()
    pr.print_stats(sort="time")

if __name__ == "__main__":
    run()

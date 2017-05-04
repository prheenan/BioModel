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

"""
This file is based on 

Dudko, O. K., et al.
Extracting Kinetics from Single-Molecule Force Spectroscopy: 
Nanopore Unzipping of DNA Hairpins. 
Biophys J 92, 4188-4195 (2007)

All equations referenced are from this paper

"""

def escape_rate(loading_rate,rupture_force,delta_G_ddagger,x_ddagger,k0,
                nu,beta):
    """
    Equation 1, the escape rate k(F) for irreversible rupture under a 
    constant external force F. Note that as long as the units are consistent,
    things should work out...

    Args:
         loading_rate: K*v in the dudko model, or more generally the local force
         derivative with respect to time. Units of <Force>/<Time>
         
         rupture_force: Units of <Force>, the forces we want the rates at

         delta_G_ddagger: the activation free energy of the single-well energy
         landscape, in <Force>*<Distance> = <Energy> units

         x_ddagger: the distance from the well to the landscape barrier, units
         of <Distance>

         k0: the 'intrinsic rate' (rate of escape at zero force), units of
         1/<Time>

         nu: scaling parameter (see discussion near Eq. 1). Relates to the shape
         of the well. Unitless.

         beta: 1/(k_b * T), where k_b is boltzmann's constant and T is the 
         temperature. Units of 1/(<Force>*<Distance>) = 1/<Energy> 
    Returns:
         array of escape rates, given the inputs. 
    """
    a = (1-nu*rupture_force*x_ddagger/delta_G_ddagger)
    b = 1-a**(1/nu)
    return k0 * (a ** (1/nu -1)) * np.exp(beta*delta_G_ddagger * b)

def dudko_model(loading_rate,rupture_force,delta_G_ddagger,x_ddagger,k0,nu,beta):
    """
    Equation 2: the probability of a given rupture force, given a fixed pulling
    speed (or loading rate).

    Args:
       see: escape_rate
    Returns:
       array of (unnormalized) probabilities at each given rupture force.
    """
    k_F = escape_rate(loading_rate,rupture_force,delta_G_ddagger,x_ddagger,
                      k0,nu,beta)
    c = beta*x_ddagger*loading_rate                      
    d = (1-nu * rupture_force * x_ddagger/delta_G_ddagger)**(1-1/nu)
    return (1/loading_rate) * k_F * np.exp(k0/c) * np.exp( (-k_F/c) * d)
            
def mean_rupture_force(loading_rate,delta_G_ddagger,x_ddagger,k0,nu,beta):
    """
    Equation 3: the expected value of the rupture force, given a fixed pulling
    speed (or loading rate).

    Args:
       see: escape_rate
    Returns:
       array of expected rupture forces as 
    """
    c0 = delta_G_ddagger*beta
    gamma = 0.577
    f = (1/c0) * np.log(k0*np.exp(c0+gamma)/(beta*x_ddagger*loading_rate))
    return delta_G_ddagger/(nu*x_ddagger) * (1- f**nu)
    
def stdev_rupture_force(loading_rate,delta_G_ddagger,x_ddagger,k0,nu,beta):
    """
    Equation 4: the standard deviation of the rupture force, 
    given a fixed pulling speed (or loading rate).

    Args:
       see escape_rate
    Returns:
       array of standard deviations
    """
    c0 = delta_G_ddagger*beta
    gamma_t = 1.064
    f = np.log(k0*np.exp(c0+gamma_t)/(beta*x_ddagger*loading_rate))
    variance = (np.pi**2/(6*(beta*x_ddagger)**2)) * (1/c0 * f)**(2*nu-2)
    return np.sqrt(variance)
            
def normalized_model(loading_rate,rupture_forces,**kwargs):
    """
    Just a convenience wrapper to normalize the model (Equaiton 2)

    Args:
       see escape_rate
    Returns:
       array of normalized probabilities, given the inputs parameters.
    """
    model = dudko_model(loading_rate,rupture_forces,**kwargs)
    model = model/sum(model)                            
    return model

def free_energy_landscape(x,delta_G_ddagger,x_ddagger,nu):
    """
    The 
    """
    tol = 1e-3
    x_rel = x/x_ddagger
    # XXX should do bell...
    if abs(nu-2/3) <= tol:
        # linear-cubic
        to_ret = 3/2 * delta_G_ddagger * x_rel - 2 * delta_G_ddagger * x_rel**3
    elif abs(nu-1/2) <= tol:
        # cusp
        to_ret = delta_G_ddagger * x_rel**2
    else:
        assert False , "Value of nu does not have known landscape"
    return to_ret

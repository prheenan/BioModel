# force floating point division. Can still use integer with //
from __future__ import division
# other good compatibility recquirements for python3
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys,copy

def next_q(q_0,D_q,beta,delta_t,dV_dq):
    """
    Returns the next molecular extension, as in appendix of Hummer, 2010

    Args:
        q_0: the initial position, units of m
        D_q: the diffusion coefficent (of the bead), units of m^2/s
        beta: 1/(k*T), room temperature is 1/(4.1e-21 J)
        delta_t: the time step, units of 1/s
        dV_dq: the force in the current state as a function of the molecular
               extension
    Returns:
        the next q
    """
    g_n = np.random.normal(loc=0,scale=1)
    dV_dq_i = dV_dq(q_0)
    return q_0 - D_q * delta_t * beta * dV_dq_i + (2*D_q*delta_t)**(1/2) * g_n

def p_jump_n(k_i,q_n,q_n_plus_one,delta_t):
    """
    The probability to jump from a gien state to the other state, see next_q

    Args:
        k_i: the transition rate, 1/s
        q_<n/n_plus_one>: see next_q 
        delta_t: see next_q
    Returns:
        probability between 0 and 1
    """
    return 1-np.exp(-(k_i(q_n) + k_i(q_n_plus_one)) * delta_t/2)

def single_step(q_n,D_q,beta,delta_t,dV_dq,k_i):
    """
    Runs a single step; gets q_n and if the molecule transitions

    Args:
        see next_q,p_jump_q
    Returns:
        tuple of <q_(n+1), did jump happen>
    """
    q_n_plus_one = next_q(q_0=q_n,D_q=D_q,beta=beta,delta_t=delta_t,dV_dq=dV_dq)
    p_jump_tmp = p_jump_n(k_i=k_i,q_n=q_n,q_n_plus_one=q_n_plus_one,
                          delta_t=delta_t)
    jump_bool = np.random.rand() < p_jump_tmp
    return q_n_plus_one,jump_bool

def k_i_f(k_0_i,beta,k_L,q_n,x_i,x_cap):
    """
    the transition rate (1/s) out of state i as a function of q

    Args:
        k_0_i: the zero-forcer transition rate, 1/s
        beta: see next_q
        k_L: the linker stiffness, N/m
        q_n: see next_q
        x_i: the state location for state i, meters
        x_cap: the location of the barrier, meters
    Returns:
        transition rate, 1/s
    """
    # see: near equation 16
    return k_0_i * np.exp(-beta/2 * k_L * ((x_cap-q_n)**2 - (x_i-q_n)**2))

def dV_dq_i(k_L,x_i,q_n,k,z_n,F_q_i):
    """
    Returns the force on a molecule with extension q in state i

    Args:
       k: stiffness of the probe, N/m
       z_n: the current probe location, m
       F_q_i: thje force on the molecule at position q, in state i
       otherS: see k_i_f, or next_q
    Returns:
       force, units of N
    """
    # see: near equation 16 (we just take the derivative)
    return -k_L * (x_i-q_n) + k*(q_n-z_n) + F_q_i

def F_q_i(k_L,x_i,q_n):
    """
    returns the force of a molecule in state i at extension q_n

    Args:
        see dV_dq_i
    Returns:
        force in N
    """
    # see: near equation 16
    return k_L * (x_i - q_n)

class simulation_state:
    def __init__(self,state,q_n,F_n,k_n,dV_n,z=None,i=None):
        self.q_n = q_n
        self.F_n = F_n
        self.state = state
        self.k_n = k_n
        self.dV_n = dV_n
        self.i = i
        self.z = z
        self.t = None
    @property
    def force(self):
        return self.F_n
    @property
    def extension(self):
        return self.q_n


def single_attempt(states,state,k,z,**kw):
    dV_tmp = lambda q: state.dV_n(q,z=z)
    q_next,swap = single_step(q_n=state.q_n,dV_dq=dV_tmp,k_i=state.k_n,**kw)
    state = 1-state.state if swap else state.state
    k_n,dV_n = states[state]
    force = k * (z - q_next)
    return simulation_state(state=state,q_n=q_next,F_n=force,k_n=k_n,dV_n=dV_n,
                            z=z)

def simulate(n_steps_equil,n_steps_experiment,x1,x2,x_cap_minus_x1,
             k_L,k,k_0_1,k_0_2,beta,z_0,z_f,s_0,delta_t,D_q):
    # get the force as a function of q
    barrier_x = [x1,x2]
    k_arr = [k_0_1,k_0_2]
    x_cap = x_cap_minus_x1 + x1
    F1,F2 = [lambda q: F_q_i(k_L,x,q) for x in barrier_x]
    # get the potential gradient (dV/dQ) as a function of q and z
    dV1,dV2 = [lambda q,z: dV_dq_i(k_L=k_L,x_i=x,q_n=q,k=k,z_n=z,F_q_i=F(q))
               for (x,F) in zip(barrier_x,[F1,F2])]
    k1,k2 = [lambda q: k_i_f(k_0_i=k_tmp,beta=beta,k_L=k_L,q_n=q,x_i=x,
                             x_cap=x_cap) for (x,k_tmp) in zip(barrier_x,k_arr)]
    states = [ [k1,dV1],
               [k2,dV2]]
    k_n,dV_n = states[s_0]
    q_n = z_0
    q_equil = [q_n]
    F_equil = [0]
    state_current = simulation_state(state=s_0,q_n=z_0,k_n=k_n,dV_n=dV_n,F_n=0)
    state_equil = [state_current]
    kw = dict(k=k,D_q=D_q,beta=beta,delta_t=delta_t)
    for i in range(n_steps_equil):
        state_current = single_attempt(states,state_current,z=z_0,**kw)
        state_equil.append(state_current)
    # POST: everything is equilibrated; go ahead and run the actual test
    state_current = state_equil[-1]
    state_exp = [state_current] 
    for i in range(n_steps_experiment):
        z_tmp = z_f(i)
        state_current.i = i
        state_current.t = i * delta_t
        state_current = single_attempt(states,state_current,z=z_tmp,**kw)
        state_exp.append(state_current)
    force = [s.force for s in state_equil + state_exp]
    ext = [s.extension for s in state_equil + state_exp]
    z = [s.z for s in state_equil + state_exp]
    plt.plot(ext,force)
    plt.show()
            

def run():
    """
    For ...
        all paraamters except k_0_1 and k_0_2...
    ... see appendix of Hummer, 2010, "Free energy profiles"

    k_0_1 = np.exp(-39)

    while
      delta_g = 193kJ/mol = 3.21e-19 J =78.2 kT, and (near equation 16)

    k_0_1/k_0_2 = exp(-beta DeltaG)
    
    so k_0_2 = k_0_1 np.exp(78.2) = exp(39.2)

    everything is in SI units
    """
    z_0 = 270e-9
    z_f = 470e-9
    v = 500e-9
    n =10000
    time_total = (z_f-z_0)/v
    delta_t = time_total/n
    params = dict(x1=170e-9,
                  x2=192e-9,
                  x_cap_minus_x1=11.9e-9,
                  k_L=0.29e-3,
                  k=0.1e-3,
                  k_0_1=np.exp(-39),
                  k_0_2=np.exp(39.2),
                  beta=1/4.1e-21,
                  z_0=z_0,
                  z_f=lambda i: (time_total * i/n) * v + z_0,
                  s_0=0,
                  delta_t=delta_t,
                  D_q=(250 * 1e-18)/1e-3)
    simulate(n_steps_equil=1000,n_steps_experiment=n,**params)

if __name__ == "__main__":
    run()
        
    
    

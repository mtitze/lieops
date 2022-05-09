import heyoka as hy
import numpy as np

from .common import realHamiltonEqs

def prepare(hamiltonian, length, n_steps, start=0):
    '''
    Prepare heyoka PDE taylor_adaptive solver.
    
    TODO: arbitrary dim check/update & further options of heyoka.
    
    Returns
    -------
    dict
        A dictionary containing the result of the solver, as well as
        the final points.
    '''
    
    q, p = hy.make_vars(*([f"q{k}" for k in range(hamiltonian.dim)] + 
                          [f"p{k}" for k in range(hamiltonian.dim)]))
    svals = np.linspace(start, length, n_steps)
    hameqs = realHamiltonEqs(hamiltonian)([q], [p]) # hameqs represents the Hamilton-equations for the real variables q and p.
    
    return {'hamilton_eqs': [(q, hameqs[0]), (p, hameqs[1])], 'svals': svals}
    
    
def run(heyp, q0, p0):
    '''
    Routine intended to be used in one-turn maps
    '''
    hameqs = heyp['hamilton_eqs']
    svals = heyp['svals']
    
    ta = hy.taylor_adaptive(hameqs, q0 + p0)
    status, min_h, max_h, n_steps, soltaylor = ta.propagate_grid(svals)
    
    # Assemble output information
    parameters = {}
    parameters['status'] = status
    parameters['min_h'] = min_h
    parameters['max_h'] = max_h
    parameters['n_steps'] = n_steps
    parameters['solution'] = soltaylor
    return {'parameters': parameters, 'zf': soltaylor[-1,:]}


def solve(hamiltonian, q0, p0, length, n_steps, **kwargs):
    '''
    Routine intended to be used as 'stand-alone' solver.
    '''
    heyp = prepare(hamiltonian=hamiltonian, length=length, n_steps=n_steps, **kwargs)
    return run(heyp, q0=q0, p0=p0)
    
    
import heyoka as hy
import numpy as np

from .common import realHamiltonEqs

def createHamiltonEqs(hamiltonian, **kwargs):
    '''
    
    TODO: arbitrary dim check/update & further options of heyoka.
    
    Returns
    -------
    dict
        A dictionary containing the real Hamilton equations and the integration steps.
    '''
    dim = hamiltonian.dim
    qp = hy.make_vars(*([f"coord_q{k}" for k in range(dim)] + 
                        [f"coord_p{k}" for k in range(dim)]))
    kwargs['real'] = kwargs.get('real', True) # for the solver it is necessary to use a real-valued Hamiltonian
    hameqs, rham = realHamiltonEqs(hamiltonian, **kwargs)
    hameqs_hy = hameqs(*qp) # hameqs represents the Hamilton-equations for the real variables q and p.
    return qp, hameqs_hy, rham

def prepare_propagate_grid(hamiltonian, length, n_steps, start=0, **kwargs):
    '''
    Prepare heyoka PDE taylor_adaptive solver using grid propagation.
    
    TODO: arbitrary dim check/update & further options of heyoka.
    
    Returns
    -------
    dict
        A dictionary containing the real Hamilton equations and the integration steps.
    '''
    svals = np.linspace(start, length, n_steps)
    qp, hameqs, rham = createHamiltonEqs(hamiltonian, **kwargs)
    return {'hamilton_eqs': [e for e in zip(*[qp, hameqs])], 'svals': svals, 'realHamiltonian': rham}
    
    
def run(heyp, q0, p0, **kwargs):
    '''
    Routine intended to be used in one-turn maps
    
    Returns
    -------
    dict
        A dictionary containing the result of the solver, as well as
        the final points.
    '''
    hameqs = heyp['hamilton_eqs']
    svals = heyp['svals']
    ta = hy.taylor_adaptive(hameqs, q0 + p0, **kwargs)
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
    assert type(q0) == list and type(p0) == list
    heyp = prepare_propagate_grid(hamiltonian=hamiltonian, length=length, n_steps=n_steps, **kwargs)
    return run(heyp, q0=q0, p0=p0)
    
    
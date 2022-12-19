import numpy as np
from njet import derive
import pytest

from lieops.solver.splitting import yoshida
from lieops.core.dragt import dragtfinn
from lieops.core.lie import create_coords, lexp

y1 = yoshida()
yoshida_scheme = y1.build(1)

@pytest.mark.parametrize("q0, p0, offset1, offset2", [(0, 0, 0.027, -0.018), (0.02, 0.007, 0.027, -0.018)])
def test_dragtfinn_2d(q0, p0, offset1, offset2, order=5, power=30, 
                      tol1=1e-8, tol2=1e-10, tol3=1e-8, **kwargs):
    '''
    Test if the Dragt-Finn factorization gives the same numerical values as if passing
    values through the flow function of a given 2D-Hamiltonian directly.
    
    Parameters
    ----------
    q0, p0: floats, optional
        The points at which we want to consider the flow of the Hamiltonian.
        These points will thus define the local symplectic approximation of the Dragt-Finn factorization.
        
    offset1, offset2: floats, optional
        Positions at which we want to compare the numerical values, relative to q0 and p0.
        
    order: int, optional
        The order of the n-jets which will be passed through the flow function, therefore controlling
        the order of the Taylor-series of the symplectic map.
        
    power: int, optional
        Input for the internal Dragt-Finn flow calculation.
    '''
 
    # Define the start coordinates and the Hamiltonian, which will be a rotation and some
    # 3rd order perturbation here:
    xi0 = (q0 + p0*1j)/float(np.sqrt(2))
    eta0 = xi0.conjugate()
    q, p = create_coords(1, real=True)
    mu = 1.236
    th = 0.5*mu*(q**2 + p**2) - q**3
    
    op = lexp(-th)
    _ = kwargs.setdefault('n_slices', 2)
    #_ = kwargs.setdefault('t', -1)
    op.calcFlow(method='channell', scheme=yoshida_scheme, **kwargs)
    reference = op(xi0 + offset1, eta0 + offset2)
    
    xietaf, dtest = op.tpsa(xi0, eta0, order=order)
    reference2 = [xe(offset1, offset2) for xe in xietaf]
    
    # Confirm that the determined Taylor expansion gives the same results at the requested
    # offset point than the flow function:
    assert all([abs(reference[k] - reference2[k]) < tol1 for k in range(2)])
    
    # Compute the f_k's which will provide a Dragt-Finn factorization
    fk = dragtfinn(*xietaf, offset=[offset1, offset2], tol=tol2, power=power)#**op._flow_parameters) # TODO
    
    # Check if the approximation is sufficiently close to the original values:
    run = [offset1, offset2]
    for f in fk:
        run = lexp(f)(*run, power=30)#, pullback=True) # TODO flow parameters here...

    assert all([abs(run[k] - reference[k]) < tol3 for k in range(2)])

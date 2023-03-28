import numpy as np
import pytest

from lieops.solver.splitting import yoshida
from lieops.core.dragt import dragtfinn
from lieops.core.lie import create_coords, lexp

y1 = yoshida()
yoshida_scheme = y1.build(1) # test_dragtfinn_1d will get an improved result if using a fourth-order symplectic integrator, but can also work (with tol3 adjusted) for the second-order integrator.

@pytest.mark.parametrize("q0, p0, offset1, offset2, tol1, tol2, tol3, order", 
                         [(0.002, 0.007, 0.0, 0.0, 1e-10, 1e-12, 3e-14, 5),
                          (0.002, 0.007, 0.0027, -0.0018, 1e-10, 1e-4, 1e-9, 5)])
def test_dragtfinn_1d(q0, p0, offset1, offset2, tol1, tol2, tol3, order, **kwargs):
    '''
    Test if the Dragt-Finn factorization gives the same numerical values as if passing
    values through the flow function of a given 1D-Hamiltonian directly.
    
    Parameters
    ----------
    q0, p0: floats, optional
        Positions at which we want to compare the numerical values, relative to q0 and p0.
        
    offset1, offset2: floats, optional
        The points at which we want to consider the flow of the Hamiltonian.
        These points will thus define the local symplectic approximation of the Dragt-Finn factorization.
        
    order: int, optional
        The order of the n-jets which will be passed through the flow function, therefore controlling
        the order of the Taylor-series of the symplectic map.
        
    **kwargs
        Optional keyword arguments passed to calcFlow.
    '''
 
    # Define the start coordinates and the Hamiltonian, which will be a rotation and some
    # 3rd order perturbation here:
    xi0 = (q0 + p0*1j)/float(np.sqrt(2))
    eta0 = xi0.conjugate()
    q, p = create_coords(1, real=True, max_power=kwargs.get('max_power', 10))
    mu = 1.236
    th = 0.5*mu*(q**2 + p**2) - q**3
    
    op = lexp(th)
    _ = kwargs.setdefault('n_slices', 6)
    op.calcFlow(method='channell', scheme=yoshida_scheme, **kwargs) # it is imperative to use a symplectic integrator at this point, otherwise dragtfinn will cause errors in its checks.
    reference = op(xi0, eta0)
    
    _ = op.tpsa(offset1, offset2, order=order)
    taylor_map = op.taylor_map(max_power=kwargs.get('max_power', 10))
    reference2 = [xe(xi0 - offset1, eta0 - offset2) for xe in taylor_map]
    
    # Confirm that the determined Taylor expansion gives the same results at the requested
    # offset point than the flow function:
    assert all([abs(reference[k] - reference2[k]) < tol1 for k in range(2)])
    
    # Compute the f_k's which will provide a Dragt-Finn factorization
    fk = dragtfinn(*taylor_map, offset=[offset1, offset2], tol_checks=tol2, order=order, power=20, warn=False) # instead of 'power=20', op._flow_parameters would work as well, but may take longer
    
    # Check if the approximation is sufficiently close to the original values:
    run = [xi0 - offset1, eta0 - offset2]
    for f in fk:
        run = lexp(f)(*run, **op.get_flow_parameters())
        
    assert all([abs(run[k] - reference[k]) < tol3 for k in range(2)])
    
    
xi10, xi20, eta10, eta20 = 0.01, -0.011, 0.04*1j -0.02, 0.31
xieta0 = [xi10, xi20, eta10, eta20]
    
offset0 = [0, 0, 0, 0]
offset1 = [0.1, 0.05, -0.8 + 0.02*1j, 0.4 + 0.09*1j]

xi1, xi2, eta1, eta2 = create_coords(2, max_power=10)
ham0 = - 8.81*xi1
ham1 = 0.32*xi1*eta1 + (1.21 - 0.934*1j)*eta2**2 + (1j*0.21 + 0.5234)*xi2*eta1
ham2 = 0.32*xi1*eta1 + (1.21 - 0.934*1j)*eta2**2 + (1j*0.21 + 0.5234)*xi2*eta1 - 8.81*xi1
ham3 = 0.32*xi1*eta1 + (1.21 - 0.934*1j)*eta2**3 + (1j*0.21 + 0.5234)*xi2*eta1**2 + 0.42*xi2*eta2
ham4 = 0.32*xi1*eta1 + (3.21 - 0.934*1j)*eta2**3 + (1j*0.21 + 0.5234)*xi2*eta1**2 + 0.42*xi2*eta2 - 8.81*xi1

@pytest.mark.parametrize("hamiltonian, xieta0, offset, tol1, tol_right, tol_left, tol_checks", 
                         [(ham0, xieta0, offset0, 1e-14, 1e-14, 1e-14, 1e-14),
                          (ham0, xieta0, offset1, 1e-14, 1e-14, 1e-14, 1e-14),
                          (ham1, xieta0, offset0, 1e-14, 1e-14, 1e-14, 1e-14),
                          (ham1, xieta0, offset1, 1e-14, 1e-14, 1e-14, 1e-14),
                          (ham2, xieta0, offset0, 1e-14, 1e-14, 1e-14, 1e-14),
                          (ham2, xieta0, offset1, 1e-14, 1e-14, 1e-14, 1e-14),
                          (ham3, xieta0, offset0, 1e-14, 1e-14, 1e-14, 1e-14),
                          (ham3, xieta0, offset1, 1e-14, 5e-14, 6e-12, 1e-13),
                          (ham4, xieta0, offset0, 2e-13, 1e-11, 1e-10, 1e-11),
                          (ham4, xieta0, offset1, 2e-13, 3e-8, 1e-6, 1e-11)])
def test_dragtfinn_2d(hamiltonian, xieta0, offset, tol1, tol_right, tol_left, tol_checks, order=7,**kwargs):
    '''
    Test the Dragt/Finn factorization for a 2D-Hamiltonian (similar to test_dragtfinn_1d).
    
    Parameters
    ----------
    hamiltonian: poly
        A Lie-polynomial, representing the Hamiltonian which will provide the flow to be examined.
        
    xieta: list
        A list of complex numbers at which point we should evaluate the flow.
        
    offset: list
        A list of complex numbers at which point the Taylor-expansion of the flow function should be determined.
        
    order: int, optional
        The order of the Taylor expansion.
        
    **kwargs
        Optional further parameters for flow calculations ('power', 'n_slices').
    '''
    dim2 = len(xieta0)
    hf = hamiltonian.calcFlow(method='channell', n_slices=kwargs.get('n_slices', 10)) # use a symplectic integrator at this step
    ref1 = hf(*xieta0)
    _ = hf.tpsa(*offset, order=order)
    taylor_map = hf.taylor_map()
    ref2 = [xe(*[xieta0[k] - offset[k] for k in range(dim2)]) for xe in taylor_map]

    assert all([abs(ref1[k] - ref2[k]) < tol1 for k in range(dim2)]) # consistency check that the Taylor map and the Hamiltonian flow agree
    
    df_right = dragtfinn(*taylor_map, power=kwargs.get('power', 40), offset=offset, pos2='right', warn=False, tol_checks=tol_checks) 
    df_left = dragtfinn(*taylor_map, power=kwargs.get('power', 40), offset=offset, pos2='left', warn=False, tol_checks=tol_checks)
        
    point_right = [xieta0[k] - offset[k] for k in range(dim2)]
    for op in df_right:
        point_right = lexp(op)(*point_right, power=kwargs.get('power', 40)) # power has to be sufficiently high in our examples; attention: this may depend on the Hamiltonian

    point_left = [xieta0[k] - offset[k] for k in range(dim2)]
    for op in df_left:
        point_left = lexp(op)(*point_left, power=kwargs.get('power', 40))

    assert all([abs(ref1[k] - point_right[k]) < tol_right for k in range(dim2)])
    assert all([abs(ref1[k] - point_right[k]) < tol_left for k in range(dim2)])
    
from njet import derive

def complexHamiltonEqs(hamiltonian):
    r'''
    Compute the Hamilton-equations for a given ops.poly class.
    
    Parameters
    ----------
    hamiltonian: ops.poly
        A polynomial representing the current Hamiltonian.
        
    Returns
    -------
    callable
        A function representing the right-hand side in the equation
        \dot \xi = -1j*\nabla H(xi, xi.conjugate())
    '''
    dhamiltonian = (hamiltonian*-1j).derive(order=1) 
    # The above factor -1j stems from the fact that the equations of motion are
    # given with respect to the complex xi and eta variables.
    def eqs(*z):
        zinp = list(z) + [e.conjugate() for e in z]
        dH = dhamiltonian.grad(*zinp)
        dxi = [dH.get((k,), 0) for k in range(hamiltonian.dim, 2*hamiltonian.dim)]
        return dxi
    return eqs

def getRealHamiltonFunction(hamiltonian, real=True, tol=0, **kwargs):
    '''
    Create a Hamilton function H(q, p) -> real, for a given Hamiltonian.
    
    Parameters
    ----------
    hamiltonian: ops.poly
        A poly object, representing a Hamiltonian in its default complex (xi, eta)-coordinates.
    
    real: boolean, optional
        If true, it will be assumed that the real form of the given
        Hamiltonian has no imaginary parts (for example, if the given Hamiltonian emerged from
        some real-valued functions).
        
    tol: float, optional
        If > 0, then drop Hamiltonian coefficients below this threshold.
        
    **kwargs
        Optional keyword arguments passed to hamiltonian.realBasis routine.
        
    Returns
    -------
    callable
        A function taking values in 2*hamiltonian.dim input parameters and returns a complex (or real) value.
        It will represent the Hamiltonian with respect to its real (q, p)-coordinates.
    '''
    dim = hamiltonian.dim
    rbh = hamiltonian.realBasis(**kwargs)
    if real:
        rbh = {k: v.real for k, v in rbh.items()}
    if tol > 0:
        rbh = {k: v for k, v in rbh.items() if abs(v) >= tol}
    def ham(*qp):
        result = 0
        for k, v in rbh.items():
            power = 1
            for l in range(dim):
                power *= qp[l]**k[l]*qp[l + dim]**k[l + dim]
            result += power*v
        return result
    return ham

def realHamiltonEqs(hamiltonian, **kwargs):
    r'''
    Obtain the real-valued Hamilton-equations for a given ops.poly class.
    
    Parameters
    ----------
    hamiltonian: ops.poly
        A ops.poly object, representing the polynomial expansion of a Hamiltonian in its (default)
        complex (xi, eta)-coordinates.
    
    **kwargs
        Optional keyword arguments passed to getRealHamiltonFunction routine.
        
    Returns
    -------
    callable
        A function taking values in real (q, p)-variables, representing the right-hand side of the
        Hamilton-equations \dot z = J \nabla H(z).
    '''
    realHam = getRealHamiltonFunction(hamiltonian, **kwargs)
    dim = hamiltonian.dim
    dhamiltonian = derive(realHam, order=1, n_args=2*dim)    
    def eqs(*qp):
        dH = dhamiltonian.grad(*qp)
        dqp = [dH.get((k + dim,), 0) for k in range(dim)] + [-dH.get((k,), 0) for k in range(dim)]
        return dqp
    return eqs
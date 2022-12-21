from lieops.core import create_coords, lexp

def test_2d_hamiltonian(n_slices=100, tol=1e-2, tol2=3e-1):
    '''
    Test the outcome of applying a Lie operator of a 2D Hamiltonian.
    '''
    xi0, eta0 = 0.888 + 0.625*1j, 0.125 - 124*1j
    q, p = create_coords(1, real=True)
    
    ham = 0.423*q*p + 0.24*q - 1.002*p**2 + (0.032 - 0.64*1j)*q**2
    op = lexp(ham)
    
    reference1 = op(xi0, eta0, power=10)
    reference2 = op(xi0, eta0, power=10, t=-1)
    reference3 = op(xi0, eta0, method='2flow', t=1)
    reference4 = op(xi0, eta0, method='2flow', t=-1)
    
    ch1 = op(xi0, eta0, method='channell', n_slices=n_slices, t=1)
    ch2 = op(xi0, eta0, method='channell', n_slices=n_slices, t=-1)
    assert all([abs(reference1[k] - ch1[k]) < tol for k in range(2)])
    assert all([abs(reference2[k] - ch2[k]) < tol for k in range(2)])
    assert all([abs(reference3[k] - ch1[k]) < tol2 for k in range(2)])
    assert all([abs(reference4[k] - ch2[k]) < tol2 for k in range(2)])

def test_4d_monomial(n_slices=1000, tol=5e-9):
    '''
    Test the outcome of applying a Lie operator of a single 4D monomial to coordinate functions.
    '''
    xi0, xi1, eta0, eta1 = 0.223 + 0.2*1j, 0.637 - 0.53*1j, 0.32 + 0.63*1j, 0.553 - 0.155*1j
    xieta0 = [xi0, xi1, eta0, eta1]
    q1, q2, p1, p2 = create_coords(2, real=True)
    mon = 0.423*q1*q2**2*p1*p2
    op = lexp(mon)
    
    reference1 = op(*xieta0, power=8)
    reference2 = op(*xieta0, power=8, t=-1)
    
    ch1 = op(*xieta0, method='channell', n_slices=n_slices, t=1)
    ch2 = op(*xieta0, method='channell', n_slices=n_slices, t=-1)
    
    assert all([abs(reference1[k] - ch1[k]) < tol for k in range(4)])
    assert all([abs(reference2[k] - ch2[k]) < tol for k in range(4)])
    
def test_4d_hamiltonian(n_slices=100, tol=2e-3):
    '''
    Test the outcome of applying a Lie operator of a Hamiltonian consisting
    of three terms.
    '''
    xi0, xi1, eta0, eta1 = 0.223 + 0.2*1j, 0.637 - 0.53*1j, 0.32 + 0.63*1j, 0.553 - 0.155*1j
    xieta0 = [xi0, xi1, eta0, eta1]
    q1, q2, p1, p2 = create_coords(2, real=True)
    
    mon = 0.423*q1*q2**2*p1*p2
    ham2 = mon + 0.553*q1*q2 - 1j*p2
    op = lexp(ham2)
    
    reference1 = op(*xieta0, power=10)
    reference2 = op(*xieta0, power=10, t=-1)
    
    ch1 = op(*xieta0, method='channell', n_slices=n_slices, t=1)
    ch2 = op(*xieta0, method='channell', n_slices=n_slices, t=-1)
        
    assert all([abs(reference1[k] - ch1[k]) < tol for k in range(4)])
    assert all([abs(reference2[k] - ch2[k]) < tol for k in range(4)])
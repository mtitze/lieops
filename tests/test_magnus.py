import numpy as np
from numpy.polynomial.polynomial import polymul
import pytest

from njet.jet import factorials

from lieops.ops.magnus import forests, fast_hard_edge_chain
from lieops.ops.lie import combine, create_coords, lexp, poly


def number_of_graphs(order):
    '''
    The number of trees of a given order.
    
    Notation in Iserles & Norsett: "On the solution of linear differential equations in Lie groups."
    '''
    facts = factorials(2*order)
    return [int(facts[2*k]/facts[k]/facts[k + 1]) for k in range(order)]

def np_block_polymul(block1, block2, **kwargs):
    n, m = block1.shape
    vals = []
    for k in range(m):
        vals.append(polymul(block1[:, k], block2[:, k]))
    return vals

def compare_polymul_results(result, reference, index, tol=1e-15):
    '''
    Compare results from numpy.polymul vs. the block_polymul routine.
    '''
    for k in range(len(reference)):
        e_ref = reference[k]
        # 'index' may reach the block size, meaning that in 'result', higher-order terms
        # are discarded. So we have to cut e_ref up to 'index'.
        # On the other hand, len(e_ref) may also be smaller than 'index',
        # if the current element contains mostly zeros (but other elements not). So we have
        # to cut 'result' up to len(e_ref).
        max_index = min([len(e_ref), index + 1])
        diff = e_ref[:max_index] - result[:max_index, k]
        adiff = max([abs(e) for e in diff])
        assert adiff < tol, f'Difference: {adiff} > {tol} (tol).'

def nc(*args):
    '''
    Returns the nested commutator expression (sometimes called Chibrikov basis)
    x1@(x2@(x3@(x4@ .... )))
    '''
    out = args[-1]
    for x in args[len(args) - 2::-1]:
        out = x@out
    return out

def bch(x, y):
    '''
    Compute the explicit commutators of the Baker-Campbell-Hausdorff series for
    orders up and including 6.
    '''
    result = {}
    result[1] = x + y
    result[2] = 0.5*(x@y)
    result[3] = 1/12*(nc(x, x, y) + nc(y, y, x))
    result[4] = -1/24*nc(y, x, x, y)
    #result[5] = 1/180*nc(y, y, x, x, y) - 1/360*(x@y)@nc(y, x, y) - 1/180*nc(y, x, x, x, y) - 1/120*(x@y)@nc(x, x, y) + 1/720*nc(y, y, y, x, y) - 1/720*nc(x, x, x, x, y)
    result[5] = -1/720*(nc(y, y, y, y, x) + nc(x, x, x, x, y)) + \
                1/360*(nc(x, y, y, y, x) + nc(y, x, x, x, y)) + \
                1/120*(nc(y, x, y, x, y) + nc(x, y, x, y, x))
    # From Wiki (fails at symmetry test!)
    #result[6] = 1/240*nc(x, y, x, y, x, y) + \
    #            1/720*(nc(x, y, x, x, x, y) - nc(x, x, y, y, x, y)) + \
    #            1/1440*(nc(x, y, y, y, x, y) - nc(x, x, y, x, x, y))
    # Instead, we will use the results from the BCH routine of H. Hofstaetter (https://github.com/HaraldHofstaetter/BCH)
    result[6] = -1/1440*nc(y, x, x, x, y, x) + 1/720*nc(y, y, x, x, y, x) - 1/240*nc(y, x, y, x, y, x) + 1/1440*nc(y, y, y, x, y, x) - 1/720*nc(y, x, y, y, y, x)
    result[7] = -1/30240 * nc(x, x, x, x, x, y, x) + \
                 1/10080 * nc(y, x, x, x, x, y, x) - \
                 1/10080 * nc(x, y, x, x, x, y, x) - \
                 1/3360  * nc(y, y, x, x, x, y, x) - \
                 1/5040 *  nc(x, x, y, x, x, y, x) + \
                 1/1260 *  nc(y, x, y, x, x, y, x) + \
                 1/7560 *  nc(x, y, y, x, x, y, x) - \
                 1/7560 *  nc(y, y, y, x, x, y, x) + \
                 1/10080 * nc(x, x, y, y, x, y, x) - \
                 1/1008 *  nc(x, y, x, y, x, y, x) + \
                 1/3360 *  nc(y, y, x, y, x, y, x) + \
                 1/1680 *  nc(y, x, y, y, x, y, x) - \
                 1/3360 *  nc(x, y, y, y, x, y, x) - \
                 1/10080 * nc(y, y, y, y, x, y, x) - \
                 1/5040 *  nc(x, y, x, y, y, y, x) + \
                 1/2520 *  nc(y, y, x, y, y, y, x) - \
                 1/10080 * nc(x, y, y, y, y, y, x) + \
                 1/30240 * nc(y, y, y, y, y, y, x)
    return result

def bch_symmetry_c(x, y, order, tol=1e-10):
    '''
    Let Z(X, Y) be the result of the Baker-Campbell-Hausdorff series:
      exp(Z(X, Y)) = exp(X) exp(Y)
    Then it must hold [1]:
      Z(X, Y) = -Z(-Y, -X)  
    This routine will test this equation for the routine 'combine', up to a specific order.
    
    Reference(s):
    [1] B. Mielnik and J. Plebanski: Combinatorial approach to Baker-Campbell-Hausdorff exponents.
        Annales de l’I. H. P., section A, tome 12, no 3 (1970), p. 215-254
    '''
    z1, ham1, _ = combine(x, y, power=order, time=False)
    z2, ham2, _ = combine(-y, -x, power=order, time=False)
    assert z1.keys() == z2.keys()
    
    for key in z1.keys():
        for kk in z1[key].keys():
            A = z1[key][kk]
            B = 0
            if kk in z2[key].keys():
                B = z2[key][kk]
            C = A + B
            assert abs(C) < tol
    return z1

def bch_vs_reference(x, y, result, tol=1e-11):
    '''
    Check combine(x, y) vs. the prediction from the clasical Baker-Campbell-Hausdorff equation
    for orders up and including 7.
    '''
    reference = bch(x, y)
    for r in result.keys():
        if r == 0:
            continue
        if r > 7:
            break
        diff = reference[r] - result[r - 1]
        hh = np.abs(list(diff.values()))
        if len(hh) > 0:
            hh = max(hh)
        else:
            hh = 0
        assert hh < tol

#########
# Tests #
#########

def test_number_of_graphs(order=10):
    '''
    Test whether the number of trees from the forest routine agrees with the prediction.
    '''
    prediction = number_of_graphs(order)
    a, _ = forests(order)
    
    for k in range(order):
        assert prediction[k] == len(a[k])
        
@pytest.mark.parametrize("m", [7, 12, 101])
def test_integrate1(m):
    '''
    Test integration routine in fast_hard_edge_chain.
    '''
    values = np.random.rand(m) + np.random.rand(m)*1j
    lengths = np.random.rand(m)
    c1 = fast_hard_edge_chain(values=values, lengths=lengths, blocksize=2)
    c1i = c1.integrate()
    assert c1i._integral == sum(values*lengths)
    
@pytest.mark.parametrize("n, m", [(1, 7), (2, 12), (6, 101), (8, 1036)])
def test_integrate2(n, m):
    '''
    Test integration routine in fast_hard_edge_chain.
    '''
    assert n >= 1 and m >= 1
    
    values = np.random.rand(m) + np.random.rand(m)*1j
    lengths = np.random.rand(m)
    c1 = fast_hard_edge_chain(values=values, lengths=lengths, blocksize=n + 2) # a block size of n + 2 will permit us to perform n + 1 integrations without data loss
    c1i = c1.integrate(n)
    c1ii = c1i.integrate()

    assert c1i._imax == n
    assert c1ii._imax == n + 1
    
    facts = factorials(n + 1)
    block = c1i._block
    integral = 0
    for k in range(n + 1): # c1i._imax == n, so that we iterate up and including n
        row = block[k, :]
        integral += sum(row*lengths**(k + 1))/facts[k + 1]
    assert integral == c1ii._integral
    
@pytest.mark.parametrize("n, m, n_blocks", [(10, 200, 6), (13, 25, 3)])
def test_block_polymul1(n, m, n_blocks):
    '''
    Test block_polymul in fast_hard_edge_chain vs. numpy polymul results.
    '''
    
    # build the blocks
    blocks1, blocks2 = [], []
    for k in range(n_blocks):
        blocks1.append(np.random.rand(n, m))
        blocks2.append(np.random.rand(n, m))
    
    # compare the results
    for k in range(n_blocks):
        result, index = fast_hard_edge_chain.block_polymul(blocks1[k], blocks2[k])
        result_ref = np_block_polymul(blocks1[k], blocks2[k])
        
        compare_polymul_results(result, result_ref, index)
        
        
@pytest.mark.parametrize("myvals1, mylengths1, myvals2, mylengths2, blocksize", 
                         [([0, 1, 0, 2.2, 0, 7, 3, 1], [1, 1, 1.6, 0.3, 8.8, 12.2, 2, 0.2],
                           [1, 1, 0.2, -3.2, 0.53, 9.236, -2.22, 0.655], [0.4, 1.1, 5, 2, 6, 0.1, 0.66, 0.2], 5)])
def test_block_polymul2(myvals1, mylengths1, myvals2, mylengths2, blocksize: int):

    assert len(myvals1) == len(mylengths1) and len(myvals2) == len(mylengths2) and len(myvals1) == len(myvals2)
    assert blocksize > 1
    
    c1 = fast_hard_edge_chain(values=myvals1, lengths=mylengths1, blocksize=blocksize)
    c1 = c1.integrate() # just to add some spice to this test
    b1 = np.copy(c1._block)
    
    c2 = fast_hard_edge_chain(values=myvals2, lengths=mylengths2, blocksize=blocksize)
    b2 = np.copy(c2._block)

    b12_ref = np_block_polymul(b1, b2)
    c12 = c1*c2
    b12 = c12._block
    compare_polymul_results(b12, b12_ref, c12._imax)
    
@pytest.mark.parametrize("m, n_chains, blocksize", [(101, 12, 8)])
def test_block_polymul3(m, n_chains, blocksize: int, tol=1e-8):
    # n.b. tol may be rather large here, because in integrate cumsum routine can lead to round-off errors etc.
    assert blocksize > 5
    chains1, chains2 = [], []
    for k in range(n_chains):
        values1 = np.random.rand(m)
        lengths1 = np.random.rand(m)
        values2 = np.random.rand(m)
        lengths2 = np.random.rand(m)

        c1 = fast_hard_edge_chain(values=values1, lengths=lengths1, blocksize=blocksize)
        c1 = c1.integrate(2)
        
        c2 = fast_hard_edge_chain(values=values2, lengths=lengths2, blocksize=blocksize)
        c2 = c2.integrate(5)
        
        chains1.append(c1)
        chains2.append(c2)
    
    for k in range(n_chains):
        c1, c2 = chains1[k], chains2[k]
        b1, b2 = np.copy(c1._block), np.copy(c2._block)
        b12_ref = np_block_polymul(b1, b2)
        c12 = c1*c2
        b12 = c12._block
        compare_polymul_results(b12, b12_ref, c12._imax, tol=tol)
        
        
def test_bch_symmetry_c(request, tol=1e-10):
    '''
    Test the symmetry of the BCH formula (see bch_symmetry_c routine) 
    and vs. the reference BCH.
    '''
    q, p = create_coords(1, real=True)
    x = p
    y = p*q**2
    z = p*q**4
    
    r1 = bch_symmetry_c(order=8, x=x, y=y, tol=tol)
    r2 = bch_symmetry_c(order=7, x=x, y=z, tol=tol)
    
    bch_vs_reference(x, y, r1)
    bch_vs_reference(x, z, r2)
    
def test_rotation_addition(mu=-0.43, tol=1e-15, power=10):
    '''
    A basic test concerning the addition of angles when combining three Lie-operators.
    '''
    o1 = lexp(poly(a=[1], b=[1], value=mu), power=power)
    mu2 = mu/3
    o2 = lexp(poly(a=[1], b=[1], value=mu2), power=power)
    o222 = o2@o2@o2
    assert (o222.argument[(1, 1)] - mu) < tol
    
def test_associativity_vs_bch(max_power=10, tol=5e-10):
    '''
    One can prove that the BCH formula must be (locally) associative. However,
    due to the lack of sufficient higher-orders, here we check only the
    agreement between the difference of a#(b#c) and (a#b)#c of both the
    results from the 'combine' routine and the one from the BCH up to order 7.
    '''
    xi, eta = create_coords(1, max_power=max_power)
    X, P = create_coords(1, real=True, max_power=max_power)
    
    mu0 = 0.206
    mu1 = 0.372
    mu0 = mu0*2*np.pi
    mu1 = mu1*2*np.pi

    w0 = 1.03
    Hs0 = w0/3*X**3
    w1 = 0.69
    Hs1 = w1/3*X**3

    H0 = -mu0*xi*eta
    H1 = -mu1*xi*eta

    a, b, c, d = H0, Hs1, H1*1j, Hs0
    
    power = 6
    
    ab, _, _ = combine(a, b, power=power, time=False)
    abc_1, _, _ = combine(sum(ab.values()), c, power=power, time=False)
    bc, _, _ = combine(b, c, power=power, time=False)
    abc_2, _, _ = combine(a, sum(bc.values()), power=power, time=False)
    diff = sum(abc_1.values()) - sum(abc_2.values())
    
    ref_ab = sum(bch(a, b).values())
    ref_abc_1 = sum(bch(ref_ab, c).values())
    ref_bc = sum(bch(b, c).values())
    ref_abc_2 = sum(bch(a, ref_bc).values())
    
    assert abc_1.keys() == abc_2.keys()
    ref_diff = ref_abc_1 - ref_abc_2
    assert diff.keys() == ref_diff.keys()
    assert max([abs(w) for w in diff - ref_diff]) < tol
    
    
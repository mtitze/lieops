from njet.functions import exp
from njet import jet

def check_zero_first_entry(x):
    '''
    Helper function to determine whether the given object is zero: If the object is a jet,
    then this function checks if the first entry is zero.
    '''
    if isinstance(x, jet):
        return x[0] == 0
    else:
        return x == 0

def productExceptSelf(a):
    '''
    Efficient code to compute the products of elements in an array, except for the value at the index itself.
    
    Example:
    
    Input: arr[]  = {10, 3, 5, 6, 2}
    Output: prod[]  = {180, 600, 360, 300, 900}
    3 * 5 * 6 * 2 product of other array 
    elements except 10 is 180
    10 * 5 * 6 * 2 product of other array 
    elements except 3 is 600
    10 * 3 * 6 * 2 product of other array 
    elements except 5 is 360
    10 * 3 * 5 * 2 product of other array 
    elements except 6 is 300
    10 * 3 * 6 * 5 product of other array 
    elements except 2 is 900
    
    See the Python3 output here: https://www.geeksforgeeks.org/a-product-array-puzzle/
    
    Parameters
    ----------
    a: list 
        The elements whose products should be computed in the above sense.
        
    Returns
    -------
    list
        A list of products in the above sense.
    '''
    n = len(a)
    
    prod = 1
    flag = 0
    for i in range(n):
        # Counting the number of elements which have value 0
        if check_zero_first_entry(a[i]):
            flag += 1
        else:
            prod *= a[i]
            
    arr = [0 for i in range(n)]
    for i in range(n):
        if flag > 1:
            # If the number of elements in the array with value 0 is more than 1, then each
            # value in the new array will be equal to 0
            arr[i] = 0
        elif flag == 0:
            # If no element has value 0, then we will insert the product/a[i] in the new array
            arr[i] = (prod/a[i])
        elif flag == 1 and not check_zero_first_entry(a[i]):
            # If one element of the array has value 0, then all the elements except that index
            # value will be equal to 0
            arr[i] = 0
        else:
            # flag == 1 and a[i] == 0
            arr[i] = prod
            
    return arr


def get_monomial_flow(hamiltonian):
    '''
    Compute the flow of a monomial.
    
    Parameters
    ----------
    hamiltonian: poly
        A poly object representing a single monomial.
        
    Returns
    -------
    callable
        A function taking dim2 := hamiltonian.dim*2 entries, and returning dim2
        entries, representing the flow of the given hamiltonian.

    References
    ----------
    [1] I. Gjaja: "Monomial factorization of symplectic maps", 
        Part. Accel. 1994, Vol. 43(3), pp. 133 -- 144.

    [2] P. J. Channell: "A brief introduction to symplectic 
        integrators and recent results", Report LA-UR-94-0063,
        Los Alamos National Lab. 1993.

    [3] P. J. Channell, personal correspondence.
    '''
    
    assert len(hamiltonian.keys()) == 1, 'No monomial provided.'
    
    # Let H be a time-independent Hamiltonian, given in terms of 
    # q and p coordinates. Then its respective Lie operator, describing the solution
    # of the Hamilton-equations to H, has the form exp(-t:H:).
    #
    # In our setting, H is expressed, by default, in terms of complex xi/eta variables.
    # It holds for two functions F and G: poisson_factor*{F, G}_(xi/eta) = {F, G}_{p, q},
    # so that with f := poisson_factor we have slightly different Hamilton-equations:
    #
    # d(xi)/dt = {xi, H}_{p, q} = f*{xi, H}_{xi/eta} = {xi, f*H}_{xi/eta}
    # d(eta)/dt = {eta, H}_{p, q} = f*{eta, H}_{xi/eta} = {eta, f*H}_{xi/eta}
    #
    # Therefore we shall multiply H with f in order to apply the formulae.
    
    monomial = hamiltonian*hamiltonian._poisson_factor
    powers = list(monomial.keys())[0] # The tuple representing the powers of the Hamiltonian.
    value = monomial[powers] # The coefficient in front of the Hamiltonian.
    
    dim = monomial.dim
    dim2 = dim*2
    
    def monomial_flow(*z):
        A = productExceptSelf([z[k]**powers[k]*z[k + dim]**powers[k + dim] for k in range(dim)])
        
        out_q = []
        out_p = []
        for l in range(dim):
            # We deal with the four cases in Channell's code [3], here in slightly different order. See also in my notes.
            nl = powers[l]
            ml = powers[l + dim]
            ql = z[l]
            pl = z[l + dim]
                
            if ml != nl and ml != 0 and nl != 0: # Case 1
                kappa = A[l]*ql**(nl - 1)*pl**(ml - 1)*(ml - nl)*value + 1
                out_q.append(kappa**(ml/(ml - nl))*ql)
                out_p.append(kappa**(nl/(nl - ml))*pl)
            elif ml == nl and ml != 0: # Case 2; clearly nl != 0 as well
                exponent = A[l]*(ql*pl)**(ml - 1)*ml*value
                out_q.append(exp(exponent)*ql)
                out_p.append(exp(-exponent)*pl)
            elif nl == 0: # Case 3
                out_p.append(pl)
                if ml != 0:
                    out_q.append(A[l]*pl**(ml - 1)*ml*value + ql)
                else:
                    out_q.append(ql)
            else:
                # Case 4; here nl != 0 and ml == 0. This is because:
                # I) Clearly nl != 0 because of Case 3 above.
                # II) Furthermore it must hold ml == 0. Otherwise we would have ml != 0 and so, because (ml != nl and ml != 0 and nl != 0) has been treated in case 1,
                #     we would enter this case if ml == nl would hold. So ml == nl and ml != 0 and nl != 0. But this was already case 2. So necessarily ml == 0 here.
                out_q.append(ql)
                out_p.append(A[l]*ql**(nl - 1)*nl*-value + pl)
        
        return out_q + out_p
    
    return monomial_flow


def flow(monomials, **kwargs):
    '''
    Compute the flow of a given series of monomials.
    
    Parameters
    ----------
    monomials: list
        A list of poly objects, each representing a monomial on 2n-dimensional phase space.
        Such a list may be generated by a suitable Yoshida scheme (found in lieops.solver.splitting).
        
    Returns
    -------
    fc: callable
        A function taking 2n variables and returning 2n variables
    '''
    assert len(monomials) > 0
    assert all([monomials[0].dim == m.dim for m in monomials[1:]])
    dim = monomials[0].dim
        
    # initialize & compute the individual flow functions
    flows = [get_monomial_flow(-h) for h in monomials] # the minus sign is required due to the fact that exp(:h:) has a flow belonging to -h.
    
    # define the flow functions of the composition
    def fc(*z):
        for fl in flows:
            z = fl(*z)
        return z
    
    return fc

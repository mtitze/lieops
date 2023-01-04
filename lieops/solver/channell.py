from njet.functions import exp
from .splitting import iterative_commuting_split

def productExceptSelf(arr):
    '''
    Efficient code to compute the products of elements in an array, except for the value at the index itself.
    
    Example
    -------
    Input: productExceptSelf([10, 3, 5, 6, 2])
    Output: [180, 600, 360, 300, 900]
    
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
    
    See the Python3 output here: 
    https://www.geeksforgeeks.org/a-product-array-puzzle/
    
    Parameters
    ----------
    a: list 
        The elements whose products should be computed in the above sense.
        
    Returns
    -------
    list
        A list of products in the above sense.
    '''
    n = len(arr)
    # Base case
    if n == 1:
        return [1]

    # Allocate memory for the product array; initialize the product array as 1
    prod = [1 for i in range(n)]
    
    # In this loop, temp variable contains the product of
    # elements on the left side, excluding arr[i]
    temp = 1
    for i in range(n):
        prod[i] = temp
        temp *= arr[i]

    # In this loop, temp variable contains the product of
    # elements on the right side, excluding arr[i]
    temp = 1
    for i in range(n - 1, -1, -1):
        prod[i] *= temp
        temp *= arr[i]

    return prod


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
    value = -monomial[powers] # The coefficient in front of the Hamiltonian. Note the minus sign here, which comes from Channell's solution (alternatively we could modify the next equations with "-value").
    dim = monomial.dim
        
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


class channell:
    
    def __init__(self, hamiltonian, n_slices: int=1, **kwargs):
        '''
        Class to manage and compute the Flow of a given Hamiltonian, based
        on Channell's method [1].
        
        Parameters
        ----------
        hamiltonian: poly
            A poly object, representing a Hamiltonian on 2n-dimensional phase space.

        split_method: callable, optional
            A function to determine the splitting of the given Hamiltonian into monomials.

        n_slices: int, optional
            Set an optional number of slices to improve results.
            
        References
        ----------
        [1] P. J. Channell: "A brief introduction to symplectic 
            integrators and recent results", Report LA-UR-94-0063,
            Los Alamos National Lab. 1993.
        '''
        
        assert len(hamiltonian.keys()) > 0
        hamiltonian = hamiltonian/n_slices
        if len(hamiltonian.keys()) > 1:
            _ = kwargs.setdefault('scheme', [0.5, 1, 0.5])
            split_method = kwargs.get('split_method', iterative_commuting_split)
            if 'split_method' not in kwargs.keys():
                # need to ensure that the default splitting routine "iterative_commuting_split" returns individual monomials, so that we can apply Channell's algorithm.
                kwargs['combine'] = False
            _ = kwargs.pop('split_method', None)
            monomials = split_method(hamiltonian, **kwargs)
            self.split_method = split_method
            self.split_method_inp = kwargs.copy()
        else:
            monomials = [hamiltonian]
        monomials = monomials*n_slices
        assert len(monomials) > 0
        self.monomials = monomials
        self.hamiltonian = hamiltonian
        self.n_slices = n_slices
        
    def calcFlows(self, reverse=False, **kwargs):
        '''
        Compute the flow of the current Hamiltonian.
        
        Parameters
        ----------
        reverse: boolean, optional
            If true, propagate the resulting monomials in reverse order.
            
        Returns
        -------
        callable
            A function taking 2n variables and returning 2n variables
        '''
        # initialize & compute the individual flow functions
        flows = [get_monomial_flow(h) for h in self.monomials]
        if reverse:
            flows = flows[::-1]
        self.flows = flows

    def __call__(self, *z, **kwargs):
        for fl in self.flows:
            z = fl(*z, **kwargs)
        return z
            
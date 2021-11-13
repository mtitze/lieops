from njet.jet import factorials
from njet.ad import standardize_function
from njet import derive
from .linalg import first_order_normal_form, matrix_from_dict

def first_order_nf_expansion(H, order, z=[], warn: bool=True, n_args: int=0, tol: float=1e-14, **kwargs):
    '''
    Return the Taylor-expansion of a Hamiltonian H in terms of first-order complex normal form coordinates
    around an optional point of interest. For the notation see my thesis.
    
    Parameters
    ----------
    H: callable
        A real-valued function of 2*n parameters (Hamiltonian).
        
    order: int
        The maximal order of expansion.
    
    z: subscriptable, optional
        A point of interest around which we want to expand.
        
    n_args: int, optional
        If H takes a single subscriptable as argument, define the number of arguments with this parameter.
        
    warn: boolean, optional
        Turn on some basic checks:
        a) Warn if the expansion of the Hamiltonian around z contains first-order terms larger than a specific value. 
        b) Verify that the 2nd order terms in the expansion of the Hamiltonian agree with those from the linear theory.
        Default: True.
        
    tol: float, optional
        An optional tolerance for checks. Default: 1e-14.
        
    **kwargs
        Arguments passed to linalg.first_order_normal_form
        
    Returns
    -------
    dict
        A dictionary of the Taylor coefficients of the Hamiltonian around z, where the first n
        entries denote powers of xi, while the last n entries denote powers of eta.
    '''
    Hst, dim = standardize_function(H, n_args=n_args)
    
    # Step 1 (optional): Shift H near z (N.B. shifts are symplectic, as they drop out from derivatives.)
    if len(z) > 0:
        H = lambda x: Hst([x[k] + z[k] for k in range(len(z))])
    else:
        H = Hst
    
    # Step 2: Obtain the Hesse-matrix of H.
    # N.B. we need to work with the Hesse-matrix here (and *not* with the Taylor-coefficients), because we want to get
    # a (linear) map K so that H o K is in CNF (complex normal form). This is guaranteed if the Hesse-matrix 
    # of H o K is in CNF form -- and this is true if the Hesse-matrix of H is transformed to CNF.
    # Note that the Taylor-coefficients of H in 2nd-order are 1/2*Hesse_matrix. This means that at (++) (see below),
    # no factor of two is required.
    dH = derive(H, order=2, n_args=dim)
    z0 = dim*[0]
    Hesse_dict = dH.hess(z0)
    Hesse_matrix = matrix_from_dict(Hesse_dict, symmetry=1, **kwargs)
    
    # Optional: Raise a warning in case the shifted Hamiltonian still has first-order terms.
    if warn:
        gradient = dH.grad()
        if any([abs(gradient[k]) > tol for k in gradient.keys()]) > 0:
            print (f'Warning: H has non-zero gradient around the requested point\n{z}\nfor given tolerance {tol}:')
            print ([gradient[k] for k in sorted(gradient.keys())])

    # Step 3: Compute the linear map to first-order complex normal form.
    nfdict = first_order_normal_form(Hesse_matrix, **kwargs)
    K = nfdict['K'] # K.transpose()*Hesse_matrix*K is in cnf
    
    # Step 4: Obtain the expansion of the Hamiltonian up to the requested order.
    Kmap = lambda zz: [sum([K[j, k]*zz[k] for k in range(len(zz))]) for j in range(len(zz))] # TODO: implement column matrix class 
    HK = lambda zz: H(Kmap(zz))
    dHK = derive(HK, order=order, n_args=dim)
    results = dHK(z0, mult=False) # mult=False ensures that we obtain the Taylor-coefficients
    
    if warn:
        # Check if the 2nd order Taylor coefficients of the derived shifted Hamiltonian agree in complex
        # normal form with the values predicted by linear theory.
        HK_hesse_dict = dHK.hess(Df=results)
        HK_hesse_dict = {k: v for k, v in HK_hesse_dict.items() if abs(v) > tol}
        for k in HK_hesse_dict.keys():
            diff = abs(HK_hesse_dict[k] - nfdict['cnf'][k[0], k[1]]) # (++)
            if diff > tol:
                print(f'CNF entry {k} does not agree with Hamiltonian expansion: diff {diff} > {tol} (tol).')
        
    return results


class liepoly:
    '''
    Class to model the Lie operator :p:, where p is a polynomial given in terms of
    complex (xi, eta)-coordinates.
    '''
    def __init__(self, **kwargs):
        # self.dim denotes the total number of available xi (or eta)-factors.
        if 'values' in kwargs.keys():
            self.values = kwargs['values']
            if len(self.values) == 0:
                self.dim = kwargs.get('dim', 0)
            else:
                self.dim = kwargs.get('dim', len(next(iter(self.values)))//2)
        else:
            self.set_monomial(**kwargs)
        self.max_power = kwargs.get('max_power', 0)
        
    def set_monomial(self, a=[], b=[], value=1, **kwargs):
        dim = max([len(a), len(b)])
        if len(a) < dim:
            a += [0]*(dim - len(a))
        if len(b) < dim:
            b += [0]*(dim - len(b))
        self.dim = dim
        self.values = {tuple(a + b): value}
        
    def __call__(self, z):
        # evaluate the polynomial at a specific position z.
        result = 0
        for k, v in self.values.items():
            prod = v
            for j in range(self.dim):
                if z[j] == 0: # in Python 0**0 = 1, but here these values are zero.
                    prod = 0
                    break
                prod *= z[j]**k[j]
            result += prod
        return result
        
    def __add__(self, other):
        add_values = {k: v for k, v in self.values.items()}
        if self.__class__.__name__ != other.__class__.__name__:
            zero_tpl = tuple([0]*self.dim*2)
            new_value = add_values.get(zero_tpl, 0) + other
            if new_value != 0:
                add_values[zero_tpl] = new_value
            else:
                _ = add_values.pop(zero_tpl)
            max_power = self.max_power
        else:
            assert other.dim == self.dim
            for k, v in other.values.items():
                new_v = add_values.get(k, 0) + v
                if new_v != 0:
                    add_values[k] = new_v
                else:
                    _ = add_values.pop(k)

            max_power = max([self.max_power, other.max_power])
        return self.__class__(values=add_values, dim=self.dim, max_power=max_power)
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return self.__class__(values={k: -v for k, v in self.values.items()}, 
                              dim=self.dim, max_power=self.max_power)
    
    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):      
        assert other.dim == self.dim      
        mult = {}
        for t1, v1 in self.values.items():
            for t2, v2 in other.values.items():
                a, b = t1[:self.dim], t1[self.dim:]
                c, d = t2[:self.dim], t2[self.dim:]
                for k in range(self.dim):
                    det = a[k]*d[k] - b[k]*c[k]
                    if det == 0:
                        continue
                    new_power = tuple([a[j] + c[j] if j != k else a[j] + c[j] - 1 for j in range(self.dim)] + \
                                [b[j] + d[j] if j != k else b[j] + d[j] - 1 for j in range(self.dim)])
                    new_value = mult.get(new_power, 0) - 1j*det*v1*v2
                    if new_value != 0:
                        mult[new_power] = new_value

        # remove expressions larger than a given power, if max_power > 0:
        max_power = max([self.max_power, other.max_power])
        if max_power > 0:
            mult = {k: v for k, v in mult.items() if sum(k) <= max_power}
        return self.__class__(values=mult, dim=self.dim, max_power=max_power)
        
    def __rmul__(self, other):
        return self.__class__(values={k: v*other for k, v in self.values.items()}, 
                              dim=self.dim, max_power=self.max_power)
        
    def power(self, power: int, y):
        '''
        Compute nested Poisson-bracket.
        E.g. let x = self. Then {x, {x, {x, {x, {x, {x, y}}}}}} =: x**6(y)
        Special case: x**0(y) := y
        
        Parameters
        ----------
        power: int
            Number of repeated brackets
        y: self.__class___
            Class which we want to evaluate on
            
        Returns
        -------
        list
            List [x**k(y) for k in range(n)], if n is the power requested.
        '''
        assert power >= 0
        assert self.__class__.__name__ == y.__class__.__name__
        result = self.__class__(values={k: v for k, v in y.values.items()}, 
                                dim=y.dim, max_power=y.max_power)
        # N.B.: We can not set values = self.values, otherwise result.values will get changed if self.values are changing.
        all_results = []
        for k in range(power):
            result = self*result
            all_results.append(result)
        return all_results
    
    def __str__(self):
        return self.values.__str__()

    def _repr_html_(self):
        return f'<samp>{self.__str__()}</samp>'
    
    
def exp_ad(x, y, power: int):
    '''
    Compute the exponential lie operator exp(:x:)y up to a given power.
    
    Parameters
    ----------
    x: liepolynom
        The polynomial defining the Lie operator :x:
    y: liepolynom
        The polynomial of which we want to apply the exponential lie operator on.
    power: int
        Integer defining the maximal power up to which we want to compute the expression.
        
    Returns
    ------- 
    list
        List containing the terms 1/k!*(:x:**k)y in the exponential expansion up to the given power.
    '''
    facts = factorials(power)
    one = liepoly(a=[0]*x.dim, b=[0]*x.dim, value=1) 
    all_results = []
    powers = x.power(power, y)
    for k in range(power): # powers[0] corresponds to {x, y}, i.e. order 1
        powers[k] = 1/facts[k + 1]*powers[k]
    return [one] + powers
            
    
def exp_ad_par(e, t):
    '''
    If exp(:x:)y is given, this function computes exp(t:x:)y for any complex number t.
    
    Parameters
    ----------
    e: list
        The output of exp_ad routine
        
    t: float
        A parameter. Can also be of type complex.
        
    Returns
    -------
    list
        The summands of exp(t:x:)y.
    '''
    return [t**k*e[k] for k in range(len(e))]


def lie_pullback(x, power: int, components=[]):
    '''
    Let f: R^n -> R be a differentiable function and :x: a polynomial Lie map. 
    Then this routine will compute the Taylor polynomials in the components of M: R^n -> R^n,
    where M is the map satisfying
    exp(:x:) f = f o M
    Furthermore, the routine will provide the Taylor polynomials of the inverse M**(-1).
    
    Note that the degree to which powers are discarded can be set by
    x.max_power.
    
    Parameters
    ----------
    x: liepoly
        The polynomial representing the map x
        
    power: int
        The degree up to which exp(:x:) should be evaluated.
        
    components: list
        List of integers denoting the components to be computed. If nothing specified, all components are calculated.
        
    Returns
    -------
    M: list
        List of components of the map M described above.
        
    Minv: list
        List of components of the map M**(-1) described above.
    '''
    
    dim2 = 2*x.dim
    if len(components) == 0:
        components = range(dim2)

    # We have to compute the maps exp(:x:)z_k for k in components.
    # Each one of these maps correspond to the k-th component of M.
    # N.B. exp(:x:)eta required if x has complex entries (TODO: perhaps find a trick to avoid the calculation...)
    M1, M2 = [], []
    Minv1, Minv2 = [], []
    for k in components:
        lp = liepoly(values={tuple([0 if j != k else 1 for j in range(dim2)])  :1} , dim=x.dim)
        
        exp_lp = exp_ad(x, lp, power=power)
        exp_lp_inv = exp_ad_par(exp_lp, -1)
        
        if k <= x.dim:
            M1.append(sum(exp_lp))
            Minv1.append(sum(exp_lp_inv))
        else:
            M2.append(sum(exp_lp))
            Minv2.append(sum(exp_lp_inv))
    return M1 + M2, Minv1 + Minv2


def Omega(mu, a, b):
    '''
    Compute the scalar product of mu and a - b.
    
    Parameters
    ----------
    mu: subscriptable
    a: subscriptable
    b: subscriptable
    
    Returns
    -------
    float
        The scalar product (mu, a - b).
    '''
    return sum([mu[k]*(a[k] - b[k]) for k in range(len(mu))])


def homological_eq(mu, Z):
    '''
    Let e[k], k = 1, ..., len(mu) be actions, H0 := sum_k mu[k]*e[k] and Z a
    polynomial of degree n. Then this routine will solve 
    the homological equation 
    {H0, chi} + Z = Q with
    {H0, Q} = 0.

    Attention: No check whether Z is actually homogeneous or real, but if one of
    these properties hold, then also chi and Q will admit this property.
    
    Parameters
    ----------
    mu: list
        list of floats (tunes).
        
    Z: liepoly
        Polynomial of degree n.
        
    Returns
    -------
    chi: liepoly
        Polynomial of degree n with the above property.
        
    Q: liepoly
        Polynomial of degree n with the above property.
    '''
    chi, Q = liepoly(values={}, dim=Z.dim), liepoly(values={}, dim=Z.dim)
    for powers, value in Z.values.items():
        om = Omega(mu, powers[:Z.dim], powers[Z.dim:])
        if om != 0:
            chi.values[powers] = 1j/om*value
        else:
            Q.values[powers] = value
    return chi, Q


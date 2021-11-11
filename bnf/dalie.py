from njet.jet import factorials
from njet.ad import standardize_function
from njet import derive


def first_order_nf_expansion(H, order, z=[], warn: bool=True, n_args: int=0, tol: float=1e-14, **kwargs):
    '''
    Return the expansion of a Hamiltonian H in terms of first-order complex normal form coordinates
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
    H, dim = standardize_function(H, n_args=n_args)
    
    # Step 1: Shift H near z (N.B. shifts are symplectic, as they drop out from derivatives.)
    if len(z) > 0:
        H_shift = lambda x: H([x[k] + z[k] for k in range(len(z))])
    else:
        H_shift = H
    
    # Step 2: Derive H twice to get its second-order Taylor coefficients.
    dH_shift = derive(H_shift, order=2, n_args=dim)
    z0 = dim*[0]
    H2_shift = dH_shift.hess(z0, mult=False) # obtain the Taylor coefficients of the 2nd order.
    
    # Optional: Raise a warning in case the shifted Hamiltonian still has first-order terms.
    if warn:
        gradient = dH_shift.grad()
        if any([abs(gradient[k]) > tol for k in gradient.keys()]) > 0:
            print (f'Warning: H has non-zero gradient around the requested point\n{z}\nfor given tolerance {tol}:')
            print (gradient)

    # Step 3: Compute the linear map to first-order complex normal form of the shifted Hamiltonian
    nfdict_shift = linalg.first_order_normal_form(H2_shift, **kwargs)
    K_shift = nfdict_shift['K']  # K.transpose()*H_shift*K is in cnf

    # Step 4: Obtain the expansion of the shifted Hamiltonian up to the requested order
    trn_shift = lambda zz: [sum([K_shift[j, k]*zz[k] for k in range(len(zz))]) for j in range(len(zz))] # TODO: implement column matrix class 
    Hcnf_shift = lambda zz: H_shift(trn_shift(zz))
    dHcnf_shift = derive(Hcnf_shift, order=order, n_args=dim)
    
    results = dHcnf_shift.eval(trn_shift(z0))
    
    if warn:
        # check if the entries in the Hessian of the derived shifted Hamiltonian agree with the
        # ones in linear theory.
        check_H2_shift = dHcnf_shift.hess(mult=False)
        check_H2_shift = {k: v for k, v in check_H2_shift.items() if abs(v) > tol}
        for k in check_H2_shift.keys():
            diff = abs(check_H2_shift[k]/2 - nfdict_shift['cnf'][k[0], k[1]])
            if diff > tol:
                print(f'Cnf entry {k} does not agree with Hamiltonian expansion: diff {diff} > {tol} (tol).')
        
    return results


class liepoly:
    '''
    Class to model the Lie operator :p:, where p is a polynomial given in terms of
    complex (xi, eta)-coordinates.
    '''
    def __init__(self, **kwargs):
        if 'values' in kwargs.keys():
            self.values = kwargs['values']
            if len(self.values) == 0:
                self.dim = kwargs.get('dim', 0)
            else:
                self.dim = kwargs.get('dim', len(list(self.values.keys())[0])//2)
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
        assert type(y) == self.__class__
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


def lie_pullback(x, power: int):
    '''
    Let f: R^n -> R be a differentiable function. Then this routine
    will compute the Taylor polynomials in the components of M: R^n -> R^n,
    where M is the map satisfying
    exp(:x:) f = f o M
    Furthermore, the routine will provide the Taylor polynomials of the inverse M**(-1).
    
    Note that the degree to which powers are discarded must be set by
    x.max_power.
    
    Parameters
    ----------
    x: liepoly
        The polynomial representing the map x
        
    power: int
        The degree up to which exp(:x:) should be evaluated.
        
    Returns
    -------
    M: list
        List of components of the map M described above.
        
    Minv: list
        List of components of the map M**(-1) described above.
    '''
    
    # We have to compute the maps exp(:x:)z_k for k = 1, ..., x.dim.
    # Each one of these maps correspond to the k-th component of M.
    M1, M2 = [], []
    Minv1, Minv2 = [], []
    for k in range(x.dim):
        zeros = [0]*x.dim
        ek = [0]*x.dim
        ek[0] = 1
        xi = liepoly(a=ek , b=zeros, value=1) # liepoly is initialized with a + b, so no danger of change in ek, zeros across instances
        eta = liepoly(a=zeros, b=ek, value=1) # exp(:x:)eta required if x has complex entries (TODO: perhaps find a trick to avoid the calculation...)
        exp_xi = exp_ad(x, xi, power=power)
        exp_xi_inv = exp_ad_par(exp_xi, -1)
        exp_eta = exp_ad(x, eta, power=power)
        exp_eta_inv = exp_ad_par(exp_eta, -1)
        M1.append(sum(exp_xi))
        M2.append(sum(exp_eta))
        Minv1.append(sum(exp_xi_inv))
        Minv2.append(sum(exp_eta_inv))
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


import numpy as np
import mpmath as mp

from njet.ad import standardize_function
from njet.jet import check_zero
from njet import derive

from .lie import liepoly, create_coords
from .lie import lexp as _lexp
from lieops.linalg.nf import normal_form
from lieops.linalg.matrix import matrix_from_dict


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


def homological_eq(mu, Z, **kwargs):
    '''
    Let e[k], k = 1, ..., len(mu) be actions, H0 := sum_k mu[k]*e[k] and Z a
    polynomial of degree n. Then this routine will solve 
    the homological equation 
    {H0, chi} + Z = Q with
    {H0, Q} = 0.

    Attention: No check whether Z is actually homogeneous or real, but if one of
    these properties hold, then also chi and Q will admit such properties.
    
    Parameters
    ----------
    mu: list
        list of floats (tunes).
        
    Z: liepoly
        Polynomial of degree n.
        
    **kwargs
        Arguments passed to liepoly initialization.
        
    Returns
    -------
    chi: liepoly
        Polynomial of degree n with the above property.
        
    Q: liepoly
        Polynomial of degree n with the above property.
    '''
    chi, Q = liepoly(values={}, dim=Z.dim, **kwargs), liepoly(values={}, dim=Z.dim, **kwargs)
    for powers, value in Z.items():
        om = Omega(mu, powers[:Z.dim], powers[Z.dim:])
        if om != 0:
            chi[powers] = 1j/om*value
        else:
            Q[powers] = value
    return chi, Q


def first_order_nf_expansion(H, power: int=2, z=[], warn: bool=True, n_args: int=0, tol: float=1e-14, 
                             code='numpy', **kwargs):
    '''
    Return the Taylor-expansion of a Hamiltonian H in terms of first-order complex normal form coordinates
    around an optional point of interest. For the notation see my thesis.
    
    Parameters
    ----------
    H: callable
        A real-valued function of 2*n parameters (Hamiltonian).
        
    power: int, optional
        The maximal polynomial power of expansion. Must be >= 2.
    
    z: subscriptable, optional
        A point of interest around which we want to expand. If nothing specified,
        then the expansion will take place around zero.
        
    n_args: int, optional
        If H takes a single subscriptable as argument, define the number of arguments with this parameter.
        
    warn: boolean, optional
        Turn on some basic checks:
        a) Warn if the expansion of the Hamiltonian around z contains gradient terms larger than a specific value. 
        b) Verify that the 2nd order terms in the expansion of the Hamiltonian agree with those from the linear theory.
        
    tol: float, optional
        An optional tolerance for checks.
        
    **kwargs
        Arguments passed to linalg.normal_form routine.
        
    Returns
    -------
    dict
        A dictionary of the Taylor coefficients of the Hamiltonian around z, where the first n
        entries denote powers of xi, while the last n entries denote powers of eta.
        
    dict
        The output of linalg.normal_form routine, providing the linear map information at the requested point.
    '''
    assert power >= 2
    Hst, dim = standardize_function(H, n_args=n_args)
    assert dim%2 == 0, 'Dimension must be even; try passing n_args argument.'
    
    # Step 1 (optional): Construct H locally around z (N.B. shifts are symplectic, as they drop out from derivatives.)
    # This step is required, because later on (at point (+)) we want to extract the Taylor coefficients, and
    # this works numerically only if we consider a function around zero.
    if len(z) > 0:
        H = lambda x: Hst([x[k] + z[k] for k in range(len(z))])
    else:
        z = dim*[0]
        H = Hst
    
    # Step 2: Obtain the Hesse-matrix of H.
    # N.B. we need to work with the Hesse-matrix here (and *not* with the Taylor-coefficients), because we want to get
    # a (linear) map K so that the Hesse-matrix of H o K is in CNF (complex normal form). This is guaranteed
    # if the Hesse-matrix of H is transformed to CNF.
    # Note that the Taylor-coefficients of H in 2nd-order are 1/2*Hesse_matrix. This means that at (++) (see below),
    # no factor of two is required.
    dH = derive(H, order=2, n_args=dim)
    z0 = dim*[0]
    Hesse_dict = dH.hess(z0)
    Hesse_matrix = matrix_from_dict(Hesse_dict, symmetry=1, code=code)
    
    # Optional: Raise a warning in case the shifted Hamiltonian still has first-order terms.
    if warn:
        gradient = dH.grad()
        if any([abs(gradient[k]) > tol for k in gradient.keys()]) > 0:
            print (f'Warning: H has a non-zero gradient around the requested point\n{z}\nfor given tolerance {tol}:')
            print ([gradient[k] for k in sorted(gradient.keys())])

    # Step 3: Compute the linear map to first-order complex normal form near z.
    nfdict = normal_form(Hesse_matrix, **kwargs)
    Kinv = nfdict['Kinv'] # Kinv.transpose()@Hesse_matrix@Kinv is in cnf; K(q, p) = (xi, eta)
    
    # Step 4: Obtain the expansion of the Hamiltonian up to the requested power.
    Kmap = lambda zz: [sum([zz[k]*Kinv[j, k] for k in range(len(zz))]) for j in range(len(zz))] # TODO: implement column matrix class. Attention: Kinv[j, k] must stand on right-hand side, otherwise zz[k] may be inserted into a NumPy array!
    HK = lambda zz: H(Kmap(zz))
    dHK = derive(HK, order=power, n_args=dim)
    results = dHK(z0, mult_drv=False) # mult_drv=False ensures that we obtain the Taylor-coefficients of the new Hamiltonian. (+)
    
    if warn:
        # Check if the 2nd order Taylor coefficients of the derived shifted Hamiltonian agree in complex
        # normal form with the values predicted by linear theory.
        HK_hesse_dict = dHK.hess(Df=results)
        HK_hesse_dict = {k: v for k, v in HK_hesse_dict.items() if abs(v) > tol}
        for k in HK_hesse_dict.keys():
            diff = abs(HK_hesse_dict[k] - nfdict['cnf'][k[0], k[1]]) # (++)
            if diff > tol:
                raise RuntimeError(f'CNF entry {k} does not agree with Hamiltonian expansion: diff {diff} > {tol} (tol).')
        
    return results, nfdict


def bnf(H, order: int=1, tol=1e-14, **kwargs):
    '''
    Compute the Birkhoff normal form of a given Hamiltonian up to a specific order.
    
    Attention: Constants and any gradients of H at z will be ignored. If there is 
    a non-zero gradient, a warning is issued by default.
    
    Parameters
    ----------
    H: callable or dict
        Defines the Hamiltonian to be normalized. If H is of type dict, then it must be of the
        form (e.g. for phase space dimension 4): {(i, j, k, l): value}, where the tuple (i, j, k, l)
        denotes the exponents in xi1, xi2, eta1, eta2.
                
    order: int
        The order up to which we build the normal form. Here order = k means that we compute
        k homogeneous Lie-polynomials, where the smallest one will have power k + 2 and the 
        succeeding ones will have increased powers by 1.
        
    max_power: int, optional
        An optional maximal power to be taken into consideration when applying ad-operations between Lie-polynomials.
        
    power: int, optional
        A maximal power by which we want to expand any exponential ad-series while evaluating Lie operators.
        
    tol: float, optional
        Tolerance below which we consider a value as zero. This will be used when examining the second-order
        coefficients of the given Hamiltonian.
        
    **kwargs
        Keyword arguments are passed to .first_order_nf_expansion routine.
        
    Returns
    -------
    dict
        A dictionary with the following keys:
        nfdict: The output of hte first_order_nf_expansion routine.
        H:      Dictionary denoting the used Hamiltonian.
        H0:     Dictionary denoting the second-order coefficients of H.
        mu:     List of the tunes used (coefficients of H0).
        chi:    List of liepoly objects, denoting the Lie-polynomials which map to normal form.
        Hk:     List of liepoly objects, corresponding to the transformed Hamiltonians.
        Zk:     List of liepoly objects, notation see Lem. 1.4.5. in Ref. [1]. 
        Qk:     List of liepoly objects, notation see Lem. 1.4.5. in Ref. [1].
        
    Reference(s):
    [1]: M. Titze: "Space Charge Modeling at the Integer Resonance for the CERN PS and SPS", PhD Thesis (2019). 
    '''
    power = order + 2 # the maximal power of the homogeneous polynomials chi mapping to normal form.
    max_power = kwargs.get('max_power', order + 2) # the maximal power to be taken into consideration when applying ad-operations between Lie-polynomials. Todo: check default & relation to 'power'
    lo_power = kwargs.get('power', order + 2) # The maximal power by which we want to expand exponential series when evaluating Lie operators. Todo: check default.
    
    if type(H) != dict:
        # obtain an expansion of H in terms of complex first-order normal form coordinates
        taylor_coeffs, nfdict = first_order_nf_expansion(H, power=power, **kwargs)
    else:
        taylor_coeffs = H
        nfdict = {}
        
    # get the dimension (by looking at one key in the dict)
    dim2 = len(next(iter(taylor_coeffs)))
    dim = dim2//2
            
    # define mu and H0. For H0 we skip any (small) off-diagonal elements as they must be zero by construction.
    H0 = {}
    mu = []
    for j in range(dim): # add the second-order coefficients (tunes)
        tpl = tuple([0 if k != j and k != j + dim else 1 for k in range(dim2)])
        muj = taylor_coeffs[tpl]
        assert muj.imag < tol
        muj = muj.real
        H0[tpl] = muj
        mu.append(muj)
    H0 = liepoly(values=H0, dim=dim, max_power=max_power)
    
    # For H, we take the values of H0 and add only higher-order terms (so we skip any gradients (and constants). 
    # Note that the skipping of gradients leads to an artificial normal form which may not have anything relation
    # to the original problem. By default, the user will be informed if there is a non-zero gradient 
    # in 'first_order_nf_expansion' routine.
    H = H0.update({k: v for k, v in taylor_coeffs.items() if sum(k) > 2})
    
    # Induction start (k = 2); get P_3 and R_4. Z_2 is set to zero.
    Zk = liepoly(dim=dim, max_power=max_power) # Z_2
    Pk = H.homogeneous_part(3) # P_3
    Hk = H.copy() # H_2 = H
        
    chi_all, Hk_all = [], [H]
    Zk_all, Qk_all = [], []
    for k in range(3, power + 1):
        chi, Q = homological_eq(mu=mu, Z=Pk, max_power=max_power) 
        if len(chi) == 0:
            # in this case the canonical transformation will be the identity and so the algorithm stops.
            break
        Hk = lexp(-chi, power=lo_power)(Hk)
        # Hk = lexp(-chi, power=k + 1)(Hk) # faster but likely inaccurate; need tests
        Pk = Hk.homogeneous_part(k + 1)
        Zk += Q 
        
        chi_all.append(chi)
        Hk_all.append(Hk)
        Zk_all.append(Zk)
        Qk_all.append(Q)

    # assemble output
    out = {}
    out['nfdict'] = nfdict
    out['H'] = H
    out['H0'] = H0
    out['mu'] = mu    
    out['chi'] = chi_all
    out['Hk'] = Hk_all
    out['Zk'] = Zk_all
    out['Qk'] = Qk_all
        
    return out


# We now extend the lexp class with the normal form analysis functionality
class lexp(_lexp):
    
    def __init__(self, *args, **kwargs):
        self.code = kwargs.get('code', 'numpy')
        # The following two internal routines are optional transformations before and after self.__call__ is executed.
        # they are used to conveniently switch the representation of a Lie operator between
        # certain coordinate systems.
        self.transform('default') # to set self._inp and self._out to be used in self.evaluate.
        _lexp.__init__(self, *args, **kwargs)
            
    def set_argument(self, H, **kwargs):
        if not H.__class__.__name__ == 'liepoly':
            assert 'order' in kwargs.keys(), "Lie operator initialized with general callable requires 'order' argument to be set." 
            self.order = kwargs['order']
            # obtain an expansion of H in terms of complex first-order normal form coordinates
            taylor_coeffs, self.nfdict = first_order_nf_expansion(H, code=self.code, **kwargs)
            _lexp.set_argument(self, x=liepoly(values=taylor_coeffs, **kwargs)) # max_power may be set here.
        else: # original behavior
            _lexp.set_argument(self, x=H, **kwargs)
            
    def bnf(self, order, output=True, **kwargs):
        '''
        Compute the Birkhoff normal form of the current Lie exponential operator.
        
        Parameters
        ----------
        order: int
            Order up to which the normal form should be computed.
            
        **kwargs
            Optional arguments passed to 'bnf' routine.
        '''
        return bnf(self.argument._values, order=order, power=self.power, 
                  max_power=self.argument.max_power, **kwargs)
            
    def transform(self, label='', inp=True, out=True, **kwargs):
        '''
        Transform the input and output of self.__call__ into a different coordinate system.
        
        Pre-defined coordinate systems require self.nfdict to be set, given by the
        output of the linalg.normal_form routine.
        
        Parameters
        ----------
        label: str
            The name of the transformation to be used. Currently supported:
            
            1) cnf, default, complex_normal_form
            2) ops, ordinary_phase_space
            3) rnf, real_normal_form, floquet
            4) aa, angle_action
            
            if left blank, then a custom transformation is assumed.
            
        inp: boolean
            Whether the transformation should be applied to the input before self.__call__ (default: True).
            
        out: boolean
            Whether the transformation should be applied to the output after self.__call__ (default: True).
        '''
        if label in ['cnf', 'default', 'complex_normal_form']:
            _inp = lambda z: z
            _out = lambda z: z

        elif label in ['ops', 'ordinary_phase_space']:
            if 'nfdict' in kwargs.keys():
                nfdict = kwargs['nfdict']
            else:
                nfdict = self.nfdict
            _inp = lambda z: nfdict['K']@z # z = (p, q) => U*S*z = (xi, eta) ; K = U*S (see notation in linalg.normal_form)
            _out = lambda z: nfdict['Kinv']@z
        
        elif label in ['rnf', 'real_normal_form', 'floquet']:
            if 'nfdict' in kwargs.keys():
                nfdict = kwargs['nfdict']
            else:
                nfdict = self.nfdict
            _inp = lambda z: nfdict['U']@z # z = (u, v) => U*z = (xi, eta) (see notation in linalg.normal_form)
            _out = lambda z: nfdict['Uinv']@z
        
        elif label in ['aa', 'angle_action']:
            # assume that the first dim parameters are angles, and the last dim parameters are actions.
            dim = kwargs.get('dim', self.n_args//2)
            code = kwargs.get('code', self.code)
            if code == 'mpmath':
                sqrt = mp.sqrt
                exp = mp.exp
                log = mp.log
            else:
                sqrt = np.sqrt
                exp = np.exp
                log = np.log
                
            def _inp(z):
                angle, action = z[:dim], z[dim:]
                xi, eta = [], []
                for k in range(dim):
                    xi.append(sqrt(action[k])*exp(-1j*angle[k]))
                    eta.append(sqrt(action[k])*exp(1j*angle[k]))
                return xi + eta
            
            def _out(z):
                xi, eta = z[:dim], z[dim:]
                angle, action = [], []
                for k in range(dim):          
                    action.append(xi[k]*eta[k])
                    if not check_zero(xi[k]):
                        angle.append(-1j*log(eta[k]/xi[k])/2)
                    elif not check_zero(eta[k]): 
                        angle.append(1j*log(xi[k]/eta[k])/2) 
                    else:
                        angle.append(0) # default value for 0-actions
                return angle + action
            
        else:
            # User-defined transformation
            inp, out = False, False
            assert 'T' in kwargs.keys() or 'Tinv' in kwargs.keys(), f"Custom transformation requires 'T' or 'Tinv' parameters to be set."
            if 'T' in kwargs.keys():
                assert hasattr(T, '__call__'), 'T not callable.'
                inp = True
                _inp = T
            if 'Tinv' in kwargs.keys():
                assert hasattr(Tinv, '__call__'), 'Tinv not callable.'
                out = True
                _out = Tinv
            
        if inp:
            self._inp = _inp
        else:
            self._inp = lambda z: z
            
        if out:
            self._out = _out
        else:
            self._out = lambda z: z
                
    def evaluate(self, z, **kwargs):
        z = self._inp(z)
        return self._out(_lexp.evaluate(self, z=z, **kwargs))

        
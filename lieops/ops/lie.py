import numpy as np

from njet import derive
from lieops.linalg.nf import first_order_nf_expansion, _create_umat_xieta

from .genfunc import genexp
from .magnus import fast_hard_edge_chain, hard_edge, hard_edge_chain, norsett_iserles
from .poly import poly as _poly

from lieops.solver.common import getRealHamiltonFunction
from lieops.solver import heyoka
from lieops.solver.bruteforce import flowFunc, calcFlow, calcOrbits

class poly(_poly):
    
    def flow(self, t=-1, *args, **kwargs):
        '''
        Let f: R^n -> R be a differentiable function and :x: the current polynomial Lie map. 
        Then this routine will compute the components of M: R^n -> R^n,
        where M is the map satisfying
        exp(:x:) f = f o M

        Note that the degree to which powers are discarded is given by self.max_power.

        Parameters
        ----------
        t: float, optional
            The flow parameter t so that we have the following interpretation:
            self.flow(t) = lexp(t*:self:)
            Note that the flow will have t=-1 by default.
        
        *args
            Arguments passed to lieoperator class.

        **kwargs
            Additional arguments are passed to lieoperator class.

        Returns
        -------
        lieoperator
            Class of type lieoperator, modeling the flow of the current Lie polynomial.
        '''
        kwargs['max_power'] = kwargs.get('max_power', self.max_power)
        return lexp(self, t=t, *args, **kwargs)
    
    def bnf(self, order: int=1, power=10, **kwargs):
        '''
        Compute the Birkhoff normal form of the current Lie exponential operator.
        
        Example
        ------- 
        nf = self.bnf()
        echi1 = nf['chi'][0].flow(t=1) # exp(:chi1:)
        echi1i = nf['chi'][0].flow(t=-1) # exp(-:chi1:)
        
        Then the map 
          z -> exp(:chi1:)(self(exp(-:chi1:)(z))) 
        will be in NF.
        
        Parameters
        ----------
        order: int, optional
            Order up to which the normal form should be computed.
            
        **kwargs
            Optional arguments passed to 'bnf' routine.
        '''
        kwargs['max_power'] = kwargs.get('max_power', self.max_power)
        return bnf(self, order=order, power=power, n_args=self.dim*2, **kwargs)
    
def create_coords(dim, real=False, **kwargs):
    '''
    Create a set of complex (xi, eta)-Lie polynomials for a given dimension.
    
    Parameters
    ----------
    dim: int
        The requested dimension.
        
    real: boolean, optional
        If true, create real-valued coordinates q and p instead. 
        
        Note that it holds:
        q = (xi + eta)/sqrt(2)
        p = (xi - eta)/sqrt(2)/1j
        
    **kwargs
        Optional arguments passed to poly class.
        
    Returns
    -------
    list
        List of length 2*dim with poly entries, corresponding to the xi_k and eta_k Lie polynomials. Hereby the first
        dim entries belong to the xi-values, while the last dim entries to the eta-values.
    '''
    resultx, resulty = [], []
    for k in range(dim):
        ek = [0 if i != k else 1 for i in range(dim)]
        if not real:
            xi_k = poly(a=ek, b=[0]*dim, dim=dim, **kwargs)
            eta_k = poly(a=[0]*dim, b=ek, dim=dim, **kwargs)
        else:
            sqrt2 = float(np.sqrt(2))
            xi_k = poly(values={tuple(ek + [0]*dim): 1/sqrt2,
                                   tuple([0]*dim + ek): 1/sqrt2},
                                   dim=dim, **kwargs)
            eta_k = poly(values={tuple(ek + [0]*dim): -1j/sqrt2,
                                    tuple([0]*dim + ek): 1j/sqrt2},
                                    dim=dim, **kwargs)
        resultx.append(xi_k)
        resulty.append(eta_k)
    return resultx + resulty

class lieoperator:
    '''
    Class to construct and work with an operator of the form g(:x:).
    
    Parameters
    ----------
    x: poly
        The function in the argument of the Lie operator.
    
    **kwargs
        Optional arguments may be passed to self.set_generator, self.calcOrbits and self.calcFlow.
    '''
    def __init__(self, x, **kwargs):
        self.init_kwargs = kwargs
        self.flow_parameter = kwargs.get('t', 1) # An optional parameter t so that the Lie operator has the form g(t:x:). The purpose of that parameter is that it may be slow to compute g(:x:) once, but, as soon as g(:x:) has been computed, g(t:x:) is faster (using the expansion of g). Furthermore, it may deal as integration step for routines in case of g = exp.
        self.set_argument(x, **kwargs)
        
        if 'generator' in kwargs.keys():
            self.set_generator(**kwargs)
        
        if 'components' in kwargs.keys():
            self.components = kwargs['components']
        else:
            self.components = create_coords(dim=self.argument.dim, **kwargs)
        
        if 't' in kwargs.keys():
            _ = self.calcFlow(**kwargs)
            
    def set_argument(self, x, **kwargs):
        assert isinstance(x, poly)
        self.argument = x
        self.n_args = 2*self.argument.dim
        
    def set_generator(self, generator, **kwargs):
        '''
        Define the generating series for the function g.
        
        Parameters
        ----------
        generator: subscriptable or callable
            If subscriptable, generator[k] =: a_k defines the generating series for the function
            g so that the Lie operator corresponds to g(:x:).
            
            g(z) = sum_k a_k*z**k.
            
            If g is a callable object, then the a_k's are determined from the Taylor coefficients of
            g. Hereby g must depend on only one parameter and it has to be supported by the njet
            module. Furthermore, the additional (integer) parameter 'power' is required to define the 
            maximal order up to which the Taylor coefficients will be determined.
        '''
        if hasattr(generator, '__iter__'):
            # assume that g is in the form of a series, e.g. given by a generator function.
            self.generator = generator
        elif hasattr(generator, '__call__'):
            if not 'power' in kwargs.keys():
                raise RuntimeError("Generation with callable object requires 'power' argument to be set.")
            # assume that g is a function of one variable which needs to be derived n-times at zero.
            assert generator.__code__.co_argcount == 1, 'Function needs to depend on a single variable.'
            dg = derive(generator, order=kwargs['power'])
            taylor_coeffs = dg(0, mult_drv= False)
            self.generator = [taylor_coeffs.get((k,), 0) for k in range(len(taylor_coeffs))]
        else:
            raise NotImplementedError('Input generator not recognized.')
        self.power = len(self.generator) - 1
        
    def calcOrbits(self, **kwargs):
        '''
        Compute the polynomials g(:x:)y for every requested y.
        '''
        if 'components' in kwargs.keys():
            self.components = kwargs['components']
        self.orbits = calcOrbits(self, components=self.components)
        return self.orbits
     
    def calcFlow(self, method='bruteforce', **kwargs):
        '''
        Compute the Lie operators [g(t:x:)]y for a given parameter t, for every y in self.components.
        
        Parameters
        ----------
        t: float (or e.g. numpy array), optional
            Parameter in the argument at which the Lie operator should be evaluated.
            
        **kwargs
            Optional arguments passed to self.calcOrbits, if the orbits were not yet computed.
            
        Returns
        -------
        list
            A list containing the flow of every component function of the Lie-operator.
        '''
        #if 'components' in kwargs.keys():
        #    self.components = kwargs['components']
        
        if 'orbits' in kwargs.keys():
            orbits = kwargs['orbits']
        elif not hasattr(self, 'orbits'):
            orbits = self.calcOrbits(**kwargs)
        else:
            orbits = self.orbits
            
        t = kwargs.get('t', self.flow_parameter)
        flow = calcFlow(orbits, t)
        self.orbits = orbits
        self.flow = flow
        self.flow_parameter = t
        return flow
    
    def flowFunc(self, t, *z):
        return flowFunc(self.orbits, t, *z)

    def evaluate(self, *z, **kwargs):
        '''
        Evaluate current flow of Lie operator at a specific point, using finite Taylor series.
        
        Parameters
        ----------
        z: subscriptable
            The vector z in the expression (g(t:x:)y)(z)
            
        Returns
        -------
        list
            The values (g(t:x:)y)(z) for y in self.components.
        '''
        if 't' in kwargs.keys():
            if kwargs['t'] != self.flow_parameter: # re-evaluate the flow at the requested flow parameter t.
                flow = self.calcFlow(**kwargs)
        else:
            assert hasattr(self, 'flow'), "Flow needs to be calculated first (check self.calcFlow)."
            flow = self.flow    
        return [flow[k](*z) for k in range(len(flow))]
        
    def act(self, *z, **kwargs):
        '''
        Let g(:x:) be the current Lie operator. Then act onto a single poly class or a list of poly classes.
        This function is basically intended as a shortcut for the successive call of self.calcOrbits and self.calcFlow.
        
        Parameters
        ----------
        z: poly classe(s)
            The polynomial(s) on which the current Lie operator should act on.
            
        Returns
        -------
        poly or list
            Depending on the input, either the poly g(:x:)y is returned, or a list g(:x:)y for the given
            poly elements y.
        '''
        _ = self.calcOrbits(components=z, **kwargs)
        result = self.calcFlow(**kwargs)
        if len(result) == 1:
            result = result[0]
        return result

    def __call__(self, *z, **kwargs):
        '''
        Compute the result of the current Lie operator g(:x:), applied to either 
        1) a specific point
        2) another Lie polynomial
        3) another Lie operator
        
        Parameters
        ----------
        z: subscriptable or poly or lieoperator
            
        **kwargs
            Optional arguments passed to self.calcFlow. Note that if an additional parameter t is passed, 
            then the respective results for g(t*:x:) are calculated.
            
        Returns
        -------
        list or poly or lieoperator
            1) If z is a list, then the values (g(:x:)y)(z) for the current poly elements y in self.components
            are returned (see self.evaluate).
            2) If z is a Lie polynomial, then the orbit of g(:x:)z will be computed and the flow returned as 
               poly class (see self.act).
            3) If z is a Lie operator f(:y:), then the Lie operator h(:z:) = g(:x:) f(:y:) is returned (see self.compose).
        '''
        if isinstance(z[0], poly):
            assert all([p.dim == z[0].dim for p in z]), 'Arguments have different dimensions.'
            return self.act(*z, **kwargs)
        elif isinstance(z[0], type(self)):
            if hasattr(self, 'bch'):
                return self.bch(*z, **kwargs) # Baker-Campbell-Hausdorff (using Magnus/combine routine)
            else:
                raise NotImplementedError(f"Composition of objects of type '{self.__class__.__name__}' not supported.")
        else:
            return self.evaluate(*z, **kwargs)
            
    def copy(self):
        '''
        Create a copy of the current Lie operator
        
        Returns
        -------
        lieoperator
            A copy of the current Lie operator.
        '''
        kwargs = {}
        kwargs.update(self.init_kwargs)
        kwargs['t'] = self.flow_parameter
        if hasattr(self, 'generator'):
            kwargs['generator'] = self.generator
        out = self.__class__(self.argument, **kwargs)
        if hasattr(self, 'orbits'):
            out.orbits = [o.copy() for o in self.orbits]
            out.components = [c.copy() for c in self.components]
        if hasattr(self, 'flow'):
            out.flow = [l.copy() for l in self.flow]
        return out

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
        
    Z: poly
        Polynomial of degree n.
        
    **kwargs
        Arguments passed to poly initialization.
        
    Returns
    -------
    chi: poly
        Polynomial of degree n with the above property.
        
    Q: poly
        Polynomial of degree n with the above property.
    '''
    chi, Q = poly(values={}, dim=Z.dim, **kwargs), poly(values={}, dim=Z.dim, **kwargs)
    for powers, value in Z.items():
        om = sum([(powers[k] - powers[Z.dim + k])*mu[k] for k in range(len(mu))])
        if om != 0:
            chi[powers] = 1j/om*value
        else:
            Q[powers] = value
    return chi, Q

def bnf(H, order: int=1, tol=1e-12, cmplx=True, **kwargs):
    '''
    Compute the Birkhoff normal form of a given Hamiltonian up to a specific order.
    
    Attention: Constants and any gradients of H at z will be ignored. If there is 
    a non-zero gradient, a warning is issued by default.
    
    Parameters
    ----------
    H: callable or dict
        Defines the Hamiltonian to be normalized. 
        I) If H is of type poly, then either H should
        already be in complex normal form, i.e. its second-order coefficients must
        be a sum of xi*eta-terms.
        II) If H is callable, then a transformation to complex normal form is performed beforehand.
                
    order: int
        The order up to which we build the normal form. Here order = k means that we compute
        k homogeneous Lie-polynomials, where the smallest one will have power k + 2 and the 
        succeeding ones will have increased powers by 1.
    
    cmplx: boolean, optional
        If false, assume that the coefficients of the second-order terms of the Hamiltonian are real.
        In this case a check will be done and an error will be raised if that is not the case.
        
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
        chi:    List of poly objects, denoting the Lie-polynomials which map to normal form.
        Hk:     List of poly objects, corresponding to the transformed Hamiltonians.
        Zk:     List of poly objects, notation see Lem. 1.4.5. in Ref. [1]. 
        Qk:     List of poly objects, notation see Lem. 1.4.5. in Ref. [1].
        
    Reference(s):
    [1]: M. Titze: "Space Charge Modeling at the Integer Resonance for the CERN PS and SPS", PhD Thesis (2019).
    [2]: B. Grebert. "Birkhoff normal form and Hamiltonian PDEs". (2006)
    '''
    power = order + 2 # the maximal power of the homogeneous polynomials chi mapping to normal form.
    max_power = kwargs.get('max_power', order + 2) # the maximal power to be taken into consideration when applying ad-operations between Lie-polynomials. TODO: check default & relation to 'power'
    lo_power = kwargs.get('power', order + 2) # The maximal power by which we want to expand exponential series when evaluating Lie operators. TODO: check default.
    
    #######################
    # STEP 1: Preparation
    #######################
    
    if hasattr(H, '__call__'):
        # assume that H is given as a function, depending on phase space coordinates.
        kwargs['power'] = power
        Hinp = H
        if isinstance(H, poly):
            # we have to transfom the call-routine of H: H depend on (xi, eta)-coordinates, but the nf routines later on assume (q, p)-coordinates.
            # In principle, one can change this transformation somewhere else, but this may cause the normal form routines
            # to either yield inconsistent output at a general point z -- or it may introduce additional complexity.
            Hinp = getRealHamiltonFunction(H, tol=tol, **kwargs)
        taylor_coeffs, nfdict = first_order_nf_expansion(Hinp, tol=tol, **kwargs)
        # N.B. while Hinp is given in terms of (q, p) variables, taylor_coeffs correspond to the Taylor-expansion
        # of the Hamiltonian around z=(q0, p0) with respect to the normal form (xi, eta) coordinates (see first_order_nf_expansion routine).
    else:
        # Attention: In this case we assume that H is already in complex normal form (CNF): Off-diagonal entries will be ignored (see code below).
        taylor_coeffs = H
        nfdict = {}
        
    # Note that if a vector z is given, the origin of the results correspond to z (since the normal form above
    # is constructed with respect to a Hamiltonian shifted by z.
        
    # get the dimension (by looking at one key in the dict)
    dim2 = len(next(iter(taylor_coeffs)))
    dim = dim2//2
            
    # define mu and H0. For H0 we skip any (small) off-diagonal elements as they must be zero by construction.
    H0 = {}
    mu = []
    for j in range(dim): # add the second-order coefficients (tunes)
        tpl = tuple([0 if k != j and k != j + dim else 1 for k in range(dim2)])
        muj = taylor_coeffs[tpl]
        # remove tpl from taylor_coeffs, to verify that later on, all Taylor coefficients have no remaining 2nd-order coeff (see below).
        taylor_coeffs.pop(tpl)
        if not cmplx:
            # by default we assume that the coefficients in front of the 2nd order terms are real.
            assert muj.imag < tol, f'Imaginary part of entry {j} above tolerance: {muj.imag} >= {tol}. Check input or try cmplx=True option.'
            muj = muj.real
        H0[tpl] = muj
        mu.append(muj)
    H0 = poly(values=H0, dim=dim, max_power=max_power)
    assert len({k: v for k, v in taylor_coeffs.items() if sum(k) == 2 and abs(v) >= tol}) == 0 # All other 2nd order Taylor coefficients must be zero.

    # For H, we take the values of H0 and add only higher-order terms (so we skip any gradients (and constants). 
    # Note that the skipping of gradients leads to an artificial normal form which may not have any relation
    # to the original problem. By default, the user will be informed if there is a non-zero gradient 
    # in 'first_order_nf_expansion' routine.
    H = H0.update({k: v for k, v in taylor_coeffs.items() if sum(k) > 2})
        
    ########################
    # STEP 2: NF-Algorithm
    ########################
               
    # Induction start (k = 2); get P_3 and R_4. Z_2 is set to zero.
    Zk = poly(dim=dim, max_power=max_power) # Z_2
    Pk = H.homogeneous_part(3) # P_3
    Hk = H.copy() # H_2 = H
        
    chi_all, Hk_all = [], [H]
    Zk_all, Qk_all = [], []
    lchi_all = []
    for k in range(3, power + 1):
        chi, Q = homological_eq(mu=mu, Z=Pk, max_power=max_power) 
        if len(chi) == 0:
            # in this case the canonical transformation will be the identity and so the algorithm stops.
            break
        lchi = lexp(-chi, power=lo_power)
        Hk = lchi(Hk)
        # Hk = lexp(-chi, power=k + 1)(Hk) # faster but likely inaccurate; need tests
        Pk = Hk.homogeneous_part(k + 1)
        Zk += Q
        
        lchi_all.append(lchi)
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
    out['lchi_inv'] = lchi_all
    out['chi'] = chi_all
    out['Hk'] = Hk_all
    out['Zk'] = Zk_all
    out['Qk'] = Qk_all
    out['order'] = order
    out['lo_power'] = lo_power
    out['max_power'] = max_power
        
    return out

    
class lexp(lieoperator):
    
    def __init__(self, x, power=10, *args, **kwargs):
        '''
        Class to describe Lie operators of the form
          exp(:x:),
        where :x: is a poly class.

        In contrast to a general Lie operator, we now have the additional possibility to combine several of these operators using the 'combine' routine.
        '''
        self._compose_power_default = 6 # the default power when composing two Lie-operators (used in self.compose)
        kwargs['generator'] = genexp(power)
        self.code = kwargs.get('code', 'numpy')
        lieoperator.__init__(self, x=x, *args, **kwargs)
        
    def set_argument(self, H, **kwargs):
        if isinstance(H, poly): # original behavior
            lieoperator.set_argument(self, x=H, **kwargs)           
        elif hasattr(H, '__call__'): # H a general function (Hamiltonian)
            assert 'order' in kwargs.keys(), "Lie operator initialized with general callable requires 'order' argument to be set." 
            self.order = kwargs['order']
            # obtain an expansion of H in terms of complex first-order normal form coordinates
            taylor_coeffs, self.nfdict = first_order_nf_expansion(H, code=self.code, **kwargs)
            lieoperator.set_argument(self, x=poly(values=taylor_coeffs, **kwargs)) # max_power may be set here.
        else:
            raise RuntimeError(f"Argument of type '{H.__class__.__name__}' not supported.")
        
    def bch(self, *z, **kwargs):
        '''
        Compute the composition of the current Lie operator exp(:x:) with other ones exp(:y_1:),
        ..., exp(:y_N:)
        to return the Lie operator exp(:z:) given as
           exp(:z:) = exp(:x:) exp(:y_1:) exp(:y_2:) ... exp(:y_N:).
           
        Parameters
        ----------
        z: lieoperators
            The Lie operators z = exp(:y:) to be composed with the current Lie operator from the right.
            
        power: int, optional
            The power in the integration variable, to control the degree of accuracy of the result.
            See also lie.combine routine. If nothing specified, self._compose_power_default will be used.
            
        **kwargs
            Additional parameters sent to lie.combine routine.
            
        Returns
        -------
        lieoperator
            The resulting Lie operator of the composition.
        '''
        assert isinstance(self, type(z[0]))
        kwargs['power'] = kwargs.get('power', self._compose_power_default)
        comb, _, _ = combine(self.argument, *[other.argument for other in z], **kwargs)
        if len(comb) > 0:
            outp = sum(comb.values())
        else: # return zero poly
            outp = poly()
        return self.__class__(outp, power=self.power)
    
    def __matmul__(self, other):
        if isinstance(self, type(other)):
            return self.bch(other)
        else:
            raise NotImplementedError(f"Operation with type {other.__class__.__name__} not supported.")
            
    def act(self, *z, method='taylor', **kwargs):
        if method == 'taylor':
            return lieoperator.act(self, *z, **kwargs)
        elif method == 'heyoka':
            # We shall use the fact that exp(:f:)p(x) = p(exp(:f:)x) holds.
            
            # Step 1: Transport the coordinate functions to their final values, using the Heyoka solver:
            self.flow_parameter = kwargs.get('t', self.flow_parameter)
            solver = heyoka(self.argument, t=self.flow_parameter, **kwargs)
            dim = z[0].dim
            vects = [[0 if k != j else 1 for k in range(dim*2)] for j in range(dim*2)]
            xietaf = solver(*vects)
            
            print (solver(1, 0))
            print (solver(0, 1))
            
#            print (vects)
#            print (xietaf)
            
            # Step 2: Act on each entry of p
            self.components = z # store the components, just for the record
            #for p in z:
                #print (p)
                #print (xietaf)
            result = [p(*xietaf) for p in z]
        
            if len(result) == 1:
                result = result[0]
            return result
    
    
def combine(*args, power: int, mode='default', **kwargs):
    r'''
    Compute the Lie polynomials of the Magnus expansion, up to a given order.
    
    Parameters
    ----------
    power: int
        The power in s (s: the variable of integration) up to which we consider the Magnus expansion.
        
    *args
        A series of poly objects p_j, j = 0, 1, ..., k which to be combined. They may represent 
        the exponential operators exp(:p_j:).
        
    mode: str, optional
        Modus how the magnus series should be evaluated. Supported modes are:
        1) 'default': Use routines optimized to work with numpy arrays (fast)
        2) 'general': Use routines which are intended to work with general objects.
        
    lengths: list, optional
        An optional list of lengths. If nothing specified, the lengths are assumed to be 1.
        
    **kwargs
        Optional keyworded arguments passed to poly instantiation and norsett_iserles routine.
        
    Returns
    -------
    dict
        The resulting Lie-polynomials z_j, j \in \{0, 1, ..., r\}, r := power, so that 
        z := z_0 + z_1 + ... z_r satisfies exp((L0 + ... + Lk):z:) = exp(L0:p_0:) exp(L1:p_1:) ... exp(Lk:p_k:),
        accurately up to the requested power r. Hereby it holds: lengths = [L0, L1, ..., Lk].
        Every z_j belongs to Norsett & Iserles approach to the Magnus series.
        
    hard_edge_chain
        The s-dependent Hamiltonian used to construct z.
        
    dict
        A dictionary containing the forest used in the calculation. This is the outcome of the norsett_iserles
        routine, which is called internally here.
    '''
    n_operators = len(args)

    # some consistency checks
    assert n_operators > 0
    assert type(power) == int and power >= 0
    dim = args[0].dim
    assert all([op.dim == dim for op in args]), 'The number of variables of the individual Lie-operators are different.'
    
    lengths = kwargs.get('lengths', [1]*n_operators)
    kwargs['max_power'] = kwargs.get('max_power', min([op.max_power for op in args]))
    # The given Lie-polynomials p_1, p_2, ... are representing the chain exp(:p_1:) exp(:p_2:) ... exp(:p_k:) of Lie operators.
    # This means that the last entry p_k is the operator which will be executed first:
    args = args[::-1] 
    lengths = lengths[::-1]
    
    # remove any possible zeros
    args1, lengths1 = [], []
    for k in range(n_operators):
        if args[k] != 0 and lengths[k] != 0:
            args1.append(args[k])
            lengths1.append(lengths[k])
    assert len(args1) > 0, 'No non-zero operators in the chain.'
    
    # Build the hard-edge Hamiltonian model.
    all_powers = set([k for op in args1 for k in op.keys()])
    if mode == 'general':
        # use hard-edge element objects which are intended to carry general objects.
        hamiltonian_values = {k: hard_edge_chain(values=[hard_edge([args1[m].get(k, 0)], lengths={1: lengths1[m]}) for m in range(n_operators)]) for k in all_powers}
    if mode == 'default':
        # use fast hard-edge element class which is optimized to work with numpy arrays.
        hamiltonian_values = {k: fast_hard_edge_chain(values=[args1[m].get(k, 0) for m in range(n_operators)], lengths=lengths1, blocksize=kwargs.get('blocksize', power + 2)) for k in all_powers}
    hamiltonian = poly(values=hamiltonian_values, **kwargs)
    
    # Now perform the integration up to the requested power.
    z_series, forest = norsett_iserles(order=power, hamiltonian=hamiltonian, **kwargs)
    out = {}
    for order, trees in z_series.items():
        lp_order = 0 # the poly object for the specific order, further polynoms will be added to this value
        for tpl in trees: # index corresponds to an enumeration of the trees for the specific order
            lp, factor = tpl
            # lp is a poly object. Its keys consist of hard_edge_hamiltonians. However we are only interested in their integrals. Therefore:
            lp_order += poly(values={k: v._integral*factor for k, v in lp.items()}, **kwargs)
        if lp_order != 0:
            out[order] = lp_order
    return out, hamiltonian, forest


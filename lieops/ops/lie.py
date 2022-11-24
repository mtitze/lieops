import numpy as np

from lieops.linalg.nf import first_order_nf_expansion

from .generators import genexp
from .magnus import fast_hard_edge_chain, hard_edge, hard_edge_chain, norsett_iserles
from .poly import _poly

from lieops.solver import heyoka, get_2flow, channell
from lieops.solver.splitting import recursive_monomial_split
from lieops.solver.bruteforce import calcFlow as BFcalcFlow

import lieops.ops.tools
import lieops.ops.birkhoff

class poly(_poly):
    
    def lexp(self, *args, **kwargs):
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
        return lexp(self, *args, **kwargs)
    
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
        return lieops.ops.birkhoff.bnf(self, order=order, power=power, n_args=self.dim*2, **kwargs)
    
    def calcFlow(self, *args, **kwargs):
        '''
        Compute the flow of the current Lie polynomial. Shortcut for self.lexp and then calcFlow on
        the respective object.
        
        Parameters
        ----------
        *args
            Optional arguments passed to self.lexp.
            
        **kwargs
            Optional keyworded arguments passed to self.lexp and lieoperator.calcFlow.
            
        Returns
        -------
        lo, lexp
            A lexp (lieoperator) object, containing the flow function of the current Lie polynomial
            in one of its fields.
        '''
        lo = self.lexp(*args, **kwargs)
        lo.calcFlow(**kwargs)
        return lo
    
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
        Optional arguments may be passed to self.set_generator and (possible) self.calcFlow.
    '''
    def __init__(self, argument, **kwargs):
        self._flow_parameters = {'t': 1, 'method': 'bruteforce'} # default parameters for flow calculations.
        
        self.init_kwargs = kwargs
        self.set_argument(argument, **kwargs)        
        self.set_components(**kwargs)
        if 'generator' in kwargs.keys():
            self.set_generator(**kwargs)
            
    def set_argument(self, argument, **kwargs):
        assert isinstance(argument, poly)
        self.argument = argument
        self.n_args = 2*self.argument.dim

    def set_components(self, **kwargs):
        if 'components' in kwargs.keys():
            self.components = kwargs['components']
        else:
            self.components = create_coords(dim=self.argument.dim, **kwargs)
        
    def set_generator(self, generator, **kwargs):
        '''
        Define the generating series for the function g.
        
        Parameters
        ----------
        generator: subscriptable or callable
            If subscriptable, generator[k] =: a_k defines the generating series for the function
            g so that the Lie operator corresponds to g(:x:).
            
            g(z) = sum_k a_k*z**k.
            
            If g is a callable object, then additional arguments are passed to this callable to
            create the a_k's.
        '''
        if hasattr(generator, '__iter__'):
            # assume that g is in the form of a series, e.g. given by a generator function.
            self.generator = generator
        elif hasattr(generator, '__call__'):
            self.generator = generator(**kwargs)
        else:
            raise NotImplementedError('Input generator not recognized.')
        self.power = len(self.generator) - 1
     
    def calcFlow(self, **kwargs):
        '''
        Compute the function(s) [g(t:x:)]y for a given parameter t, for every y in self.components.
        The result will be written to self.flow.
        
        Parameters
        ----------
        method: str, optional
            The method to be applied in calculating the flow.
            
        update: boolean, optional
            An internal switch to force the calculation of the current flow (default=True).
            
        **kwargs
            Optional arguments passed to flow subroutines.
        '''
        update = self._update_flow_parameters(**kwargs)
        if not update and hasattr(self, 'flow'):
            return
        _ = kwargs.pop('method', None)
        self._calcFlowFromParameters(**kwargs)
        
    def _calcFlowFromParameters(self, **kwargs):
        if self._flow_parameters['method'] == 'bruteforce':
            self.flow = BFcalcFlow(lo=self, **kwargs) # n.b. 't' may be a keyword of 'kwargs'
        else:
            raise NotImplementedError(f"method '{self._flow_parameters['method']}' not recognized.")
        
    def _update_flow_parameters(self, update=True, **kwargs):
        '''
        Update self._flow_parameters if necessary; return boolean if they have been updated 
        (and therefore self.flow may have to be re-calculated).
        
        This internal routine is indended to help in speeding up flow calculations, so that the flow
        is only calculated if parameters have been changed.
        '''
        if 't' in kwargs.keys():
            if self._flow_parameters['t'] != kwargs['t']:
                self._flow_parameters['t'] = kwargs['t']
                update = True
                
        if 'method' in kwargs.keys():
            if self._flow_parameters['method'] != kwargs['method']:
                self._flow_parameters['method'] = kwargs['method']
                update = True
                
        if 'components' in kwargs.keys():
            # next(iter(list[1:]), default) trick see https://stackoverflow.com/questions/2492087/how-to-get-the-nth-element-of-a-python-list-or-a-default-if-not-available
            if any([next(iter(self.components[k:]), None) != kwargs['components'][k] for k in range(len(kwargs['components']))]):
                self.components = kwargs['components']
                update = True
        return update

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
        self.calcFlow(**kwargs)
        if hasattr(self.flow, '__getitem__'):
            # We assume that self.flow is a set of functions for each coordinate
            return [self.flow[k](*z) for k in range(len(self.flow))]
        else:
            # We assume that self.flow returns a list
            return self.flow(*z)

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
            self.calcFlow(components=z, **kwargs)
            result = self.flow
            if len(z) == 1 and hasattr(result, '__getitem__'): # if the input was a single element, naturally return a single element as well (and not a list of length 1)
                result = result[0]
            return result
        
        elif isinstance(z[0], type(self)):
            if hasattr(self, 'bch'):
                return self.bch(*z, **kwargs) # Baker-Campbell-Hausdorff (using Magnus/combine routine)
            else:
                raise NotImplementedError(f"Operation on type '{z[0].__class__.__name__}' not supported.")
                
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
        kwargs['components'] = self.components
        if hasattr(self, 'generator'):
            kwargs['generator'] = self.generator
        out = self.__class__(self.argument, **kwargs)
        if hasattr(self, 'flow'):
            if hasattr(self.flow, '__getitem__'):
                out.flow = [l.copy() for l in self.flow]
            else:
                out.flow = self.flow
        return out

    
class lexp(lieoperator):
    
    def __init__(self, argument, *args, **kwargs):
        '''
        Class to describe Lie operators of the form
          exp(:x:),
        where :x: is a poly class.

        In contrast to a general Lie operator, we now have the additional possibility to combine several of these operators using the 'combine' routine.
        '''
        self._bch_power_default = 6 # the default power when composing two Lie-operators (used in self.bch)
        if 'power' in kwargs.keys():
            self.set_generator(kwargs['power'])
        lieoperator.__init__(self, argument=argument, *args, **kwargs)
        
    def set_argument(self, H, **kwargs):
        if isinstance(H, poly): # original behavior
            lieoperator.set_argument(self, argument=H, **kwargs)           
        elif hasattr(H, '__call__'): # H a general function (Hamiltonian)
            assert 'order' in kwargs.keys(), "Lie operator initialized with general callable requires 'order' argument to be set." 
            self.order = kwargs['order']
            # obtain an expansion of H in terms of complex first-order normal form coordinates
            taylor_coeffs, self.nfdict = first_order_nf_expansion(H, **kwargs)
            lieoperator.set_argument(self, argument=poly(values=taylor_coeffs, **kwargs)) # max_power may be set here.
        else:
            raise TypeError(f"Argument of type '{H.__class__.__name__}' not supported.")
            
    def set_generator(self, power):
        self.generator = genexp(power)
        self.power = len(self.generator) - 1
        
    def _update_flow_parameters(self, update=False, **kwargs):
        if 'power' in kwargs.keys():
            # if the user is giving the routine a 'power' argument, it will be assumed that the bruteforce method should be used with respect to the given power. Therefore:
            self.set_generator(kwargs['power'])
            self._flow_parameters['method'] = 'bruteforce'
            update = True
        return lieoperator._update_flow_parameters(self, update=update, **kwargs)
        
    def bch(self, *z, bch_sign=-1, **kwargs):
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
            See also lie.combine routine. If nothing specified, self._bch_power_default will be used.
            
        **kwargs
            Additional parameters sent to lie.combine routine.
            
        Returns
        -------
        lieoperator
            The resulting Lie operator of the composition.
        '''
        assert isinstance(self, type(z[0]))
        kwargs['power'] = kwargs.get('power', self._bch_power_default)
        comb, _, _ = combine(self.argument*bch_sign, 
                             *[other.argument*bch_sign for other in z], **kwargs)
        if len(comb) > 0:
            outp = sum(comb.values())*bch_sign
        else: # return zero poly
            outp = poly()
            
        outp_kwargs = {}
        if hasattr(self, 'power'):
            outp_kwargs = {'power': self.power}
        return self.__class__(outp, **outp_kwargs)
    
    def __matmul__(self, other):
        if isinstance(self, type(other)):
            return self.bch(other)
        else:
            raise NotImplementedError(f"Operation on type {other.__class__.__name__} not supported.")
            
    def __pow__(self, power):
        outp_kwargs = {}
        if hasattr(self, 'power'):
            outp_kwargs = {'power': self.power}
        return self.__class__(self.argument*power, **outp_kwargs)
            
    def _calcFlowFromParameters(self, **kwargs):
        if self._flow_parameters['method'] == '2flow':
            # if self.argument has order <= 2, one can compute the flow exactlyen changed at a later point.
            self._2flow_xieta = create_coords(self.argument.dim)
            self._2flow = get_2flow(self.argument*self._flow_parameters['t'], 
                                        tol=kwargs.get('tol', 1e-12))
            # apply self._2flow on the individual xi/eta-coordinates. They will be used
            # later on, for each of the given components, using the pull-back property of the flow
            self._2flow_xietaf = [self._2flow(xieta, **kwargs) for xieta in self._2flow_xieta]
            self.flow = [c(*self._2flow_xietaf) for c in kwargs.get('components', self.components)]
            
        elif self._flow_parameters['method'] == 'channell':
            kwargs.setdefault('scheme', [0.5, 1, 0.5])
            self._channell_scheme = kwargs['scheme']
            self.flow = channell(-self.argument*self._flow_parameters['t'], reverse=True, **kwargs)
            
        else:
            lieoperator._calcFlowFromParameters(self, **kwargs)
        
    
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
    n_operators = len(args1)
    
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


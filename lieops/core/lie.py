import numpy as np
import copy

from lieops.linalg.nf import first_order_nf_expansion

from .generators import genexp
from .combine import magnus
from .poly import _poly

from lieops.solver import heyoka, get_2flow, channell
from lieops.solver.splitting import recursive_monomial_split
from lieops.solver.bruteforce import calcFlow as BFcalcFlow

import lieops.core.tools
import lieops.core.birkhoff

from njet import derive

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
            Optional arguments passed to 'lieops.core.birkhoff.bnf' routine.
        '''
        kwargs['max_power'] = kwargs.get('max_power', self.max_power)
        return lieops.core.birkhoff.bnf(self, order=order, power=power, n_args=self.dim*2, **kwargs)
    
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
    
    def __matmul__(self, other):
        if isinstance(other, lieoperator):
            return -other(self)
        else:
            return _poly.__matmul__(self, other)        

    
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
        self._default_flow_parameters = {'t': 1} # default parameters for flow calculations.
        self._flow = {} # dictionary to store the output of the flow calculations, for each method (methods strings will be the dict keys)

        # we shall set the 'bruteforce' as initial default method to calculate the flow. This can be changed by the user in the 'calcFlow' routine, if he sets a different 'method'.
        self._flow_method = 'bruteforce' 
        self._flow_parameters = {'bruteforce': self._default_flow_parameters.copy()}
    
        self.set_argument(argument, **kwargs)
        if 'generator' in kwargs.keys():
            self.set_generator(**kwargs)
            
    def set_argument(self, argument, **kwargs):
        assert isinstance(argument, poly)
        self.argument = argument
        self.n_args = 2*self.argument.dim
        
    def get_components(self):
        '''
        Return the components which has been set in the current flow method.
        '''
        return self._flow_parameters[self._flow_method].get('components', None)

    def set_components(self, **kwargs):
        '''
        Set given components to the current flow method.
        '''
        if 'components' in kwargs.keys():
            self._flow_parameters[self._flow_method]['components'] = kwargs['components']
        else:
            _ = kwargs.setdefault('dim', self.argument.dim)
            _ = kwargs.setdefault('max_power', self.argument.max_power)
            self._flow_parameters[self._flow_method]['components'] = create_coords(**kwargs)
            
    def get_flow_parameters(self):
        '''
        Return the flow parameters for the current flow method.
        '''
        return self._flow_parameters.get(self._flow_method, self._default_flow_parameters.copy())
    
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

    def _update_flow_parameters(self, update=False, **kwargs):
        '''
        Update self._flow_parameters if necessary; return boolean if they have been updated 
        (and therefore self.flow may have to be re-calculated).
        
        This internal routine is indended to help in determining when to re-calculate
        the flow and thus speeding up flow calculations.
        '''
        current_parameters = self.get_flow_parameters()
        update = update or not kwargs.items() <= current_parameters.items()
        if 'components' in kwargs.keys():
            components = kwargs['components']
            if not 'components' in current_parameters.keys():
                current_parameters['components'] = components
                update = True
            else:
                current_components = current_parameters['components']
                # next(iter(list[1:]), default) trick see https://stackoverflow.com/questions/2492087/how-to-get-the-nth-element-of-a-python-list-or-a-default-if-not-available
                n = max([len(components), len(current_components)])
                if any([next(iter(current_components[k:]), None) != next(iter(components[k:]), None) for k in range(n)]):
                    current_parameters['components'] = components
                    update = True

        if update:
            current_parameters.update(kwargs)
            self._flow_parameters[self._flow_method] = current_parameters
            if self._flow_method in self._flow.keys():
                # also clean up any output for the chosen method (TODO: may include an option to keep the old input & output and return to it later)
                _ = self._flow.pop(self._flow_method, None)
        return update
     
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
        self._flow_method = kwargs.get('method', self._flow_method)
        update = self._update_flow_parameters(**kwargs)
        if update or not self._flow_method in self._flow.keys():
            self._calcFlowFromParameters(**kwargs)
        # set the current flow function to the requested method.
        self.flow = self._flow[self._flow_method]['flow'] 
        
    def _calcFlowFromParameters(self, **kwargs):
        if self._flow_method == 'bruteforce':
            flow_parameters = self.get_flow_parameters()
            if not 'components' in flow_parameters.keys():
                self.set_components(**kwargs)
            self._calcFlow_bruteforce(**kwargs)
        else:
            raise NotImplementedError(f"method '{self._flow_method}' not recognized.")
            
    def _calcFlow_bruteforce(self, **kwargs):
        # For a general Lie operator g(:f:), we apply g(:f:) to the given operand directly
        result = {}
        flow_poly = BFcalcFlow(lo=self, **kwargs) # n.b. 't' may be a keyword of 'kwargs'. In any case it also has been updated in self._flow_parameters
        result['flow'] = lambda *z: [flow_poly[k](*z) for k in range(len(flow_poly))]
        result['poly'] = flow_poly
        self._flow['bruteforce'] = result
        
    def _calcPolyFromFlow(self, **kwargs):
        '''
        If self.flow has been computed, compute (or return) the resulting polynomial g(:x:):y:, 
        for every y in self.components.
        '''
        result = self._flow.get(self._flow_method, {})
        if 'poly' in result.keys():
            return result['poly']
        else:
            raise RuntimeError(f"Polynomial approximation of method '{self._flow_method}' can not be found in output dict 'self._flow'.")
        
    def __call__(self, *z, outl1=False, **kwargs):
        '''
        Compute the result of the current Lie operator g(:x:), applied to either 
        1) a specific point
        2) another Lie polynomial
        
        Parameters
        ----------
        z: subscriptable or poly or lieoperator
            
        outl1: boolean, optional
            Determine the output format in case the output would be a single Lie polynomial.
            The default behavior is to return a single element. If a list of length 1 should be returned
            instead, change this parameter to True.
            
        **kwargs
            Optional arguments passed to self.calcFlow. Note that if an additional parameter t is passed, 
            then the respective results for g(t*:x:) are calculated.
            
        Returns
        -------
        list or poly or lieoperator
            1) If z is a list, then the values (g(:x:)y)(z) for the current poly elements y in self.components
            are returned (see self.evaluate).
            2) If z is a Lie polynomial, then the orbit of g(:x:)z will be computed and the flow returned as 
               poly class.
        '''
        if isinstance(z[0], poly):
            assert all([p.dim == z[0].dim for p in z]), 'Arguments have different dimensions.'
            self.calcFlow(components=z, **kwargs)
            result = self._calcPolyFromFlow(**kwargs)
            if len(result) == 1 and not outl1: # if the input was a single element, naturally return a single element as well (and not a list of length 1)
                result = result[0]
            return result
        else:
            self.calcFlow(**kwargs)
            return self.flow(*z)
        
    def __matmul__(self, other):
        if isinstance(other, poly):
            return self(other)
        else:
            raise NotImplementedError(f"Operation on type {other.__class__.__name__} not supported.")
            
    def copy(self):
        '''
        Create a copy of the current Lie operator
        
        Returns
        -------
        lieoperator
            A copy of the current Lie operator.
        '''
        out = self.__class__(self.argument)
        out.__dict__.update(copy.deepcopy(self.__dict__))
        return out
    
    def tpsa(self, *position, **kwargs):
        '''
        Pass n-jets through the flow function, given by the current Lie operator.
        
        Parameters
        ----------
        *position: float or array, optional
            An optional point of reference. By default the position will be the origin.
        
        order: int, optional
            The number of derivatives we want to take into account.
        '''
        assert hasattr(self, 'flow'), 'Flow has to be computed first.'
        if 'order' in kwargs.keys():
            order = kwargs['order']
        elif self.argument.maxdeg() <= 2:
            order = 1 # a 2nd order degree polynomial, applied to a coordinate function, will yield a first-order term, so order 1 is sufficient here.
        else:
            assert self.argument.max_power < np.inf, f'No order set. In this case max_power of {self.__class__.__name__}.argument can not be infinite.'
            order = self.argument.max_power + 1 # TODO: check if this is sufficient; see the comment in lieops.core.poly._poly concerning max_power
        n_args = self.argument.dim*2
        dflow = derive(self.flow, n_args=n_args, order=order)
        if len(position) == 0: 
            position = (0,)*n_args
        expansion = dflow(*position, mult_prm=True, mult_drv=False) # N.B. the plain jet output is stored in dflow._evaluation. From here one can use ".get_taylor_coefficients" with other parameters -- if desired -- or re-use the jets for further processing.
        xietaf = [poly(values=e, dim=self.argument.dim, max_power=self.argument.max_power) for e in expansion]
        return xietaf, dflow

    
class lexp(lieoperator):
    
    def __init__(self, argument, *args, **kwargs):
        '''
        Class to describe Lie operators of the form
          exp(:x:),
        where :x: is a poly class.

        In contrast to a general Lie operator, we now have the additional possibility to combine several of these operators using the lieops.core.combine.magnus routine.
        '''
        self._bch_order_default = 6 # the default order when composing two Lie-operators (used in self.bch)
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
            
    def set_generator(self, power, **kwargs):
        lieoperator.set_generator(self, generator=genexp(power), **kwargs)
                
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
            
        order: int, optional
            The order in the integration variable, to control the degree of accuracy of the result.
            See also lie.core.combine.magnus routine. If nothing specified, self._bch_order_default will be used.
            
        **kwargs
            Additional parameters sent to lie.core.combine.magnus routine.
            
        Returns
        -------
        lieoperator
            The resulting Lie operator of the composition.
        '''
        assert isinstance(self, type(z[0]))
        _ = kwargs.setdefault('order', self._bch_order_default)
        _ = kwargs.setdefault('disable_tqdm', True)
        comb, _, _ = magnus(self.argument*bch_sign, *[other.argument*bch_sign for other in z], **kwargs)
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
            return lieoperator.__matmul__(self, other)
            
    def __pow__(self, power):
        outp_kwargs = {}
        if hasattr(self, 'power'):
            outp_kwargs = {'power': self.power}
        return self.__class__(self.argument*power, **outp_kwargs)
    
    def _update_flow_parameters(self, update=False, **kwargs):
        if 'power' in kwargs.keys():
            # if the user is giving the routine a 'power' argument, it will be assumed that the bruteforce method is used as default with respect to the given power:
            self.set_generator(kwargs['power'])
            _ = kwargs.setdefault('method', 'bruteforce')
        return lieoperator._update_flow_parameters(self, update=update, **kwargs)
    
    def _calcFlowFromParameters(self, **kwargs):
        if self._flow_method == '2flow':
            self._calcFlow_2flow(**kwargs)
        elif self._flow_method == 'channell':
            self._calcFlow_channell(**kwargs)
        elif self._flow_method == 'njet':
            self._calcFlow_njet(**kwargs)
        else:
            lieoperator._calcFlowFromParameters(self, **kwargs)
            
    def _calcFlow_bruteforce(self, pullback=False, **kwargs):
        if pullback:
            # Since the lexp class represents the Lie operator exp(:f:), there is a second
            # possibility to compute the flow: Namely, we can apply the operator
            # to the components first, and then apply the given operand Q:
            # exp(:f:)Q(xi_1, ..., eta_n) = Q(exp(:f:)xi_1, ..., exp(:f:)eta_n)
            # The resulting set of Lie operators (for a given set of Q-polynomials) is stored
            # in self._bruteforce_result below.
            requested_components = kwargs.pop('components', self._flow_parameters['components'])
            xieta = create_coords(self.argument.dim, max_power=self.argument.max_power)
            xietaf = BFcalcFlow(lo=self, components=xieta, **kwargs) # n.b. 't' may be a keyword of 'kwargs'
            flow_poly = [c(*xietaf) for c in requested_components]
            flow = lambda *z: [flow_poly[k](*z) for k in range(len(flow_poly))]
            self._flow['bruteforce'] = {'xietaf': xietaf, 'poly': flow_poly, 'flow': flow}
        else:
            lieoperator._calcFlow_bruteforce(self, **kwargs)
            
    def _calcFlow_channell(self, **kwargs):
        '''
        Compute flow using the algorithm by P. Channell.
        
        This routine may need to use a slicing or splitting to improve its accuracy.
        '''
        kwargs.update(self._flow_parameters['channell']) # ensure that other parameters are properly passed to the channell flow class
        _ = kwargs.pop('t')
        flow = channell(self.argument*self._flow_parameters['channell']['t'], **kwargs)
        flow.calcFlows(**kwargs)
        self._flow['channell'] = {'flow': flow}
    
    def _calcFlow_2flow(self, **kwargs):
        '''
        Compute the flow in case self.argument is of order <= 2, by using
        an exact integration, see lieops.solver.get_2flow.
        
        Parameters
        ----------
        tol: float, optional
            A default tolerance used for the lieops.solver.flow2by2.get_2flow routine.
        '''
        flow = get_2flow(self.argument*self._flow_parameters['2flow']['t'], tol=kwargs.get('tol', 1e-12))
        # flow is a function expecting lieops.core.lie.poly objects. Therefore:
        xieta = create_coords(self.argument.dim, max_power=self.argument.max_power)
        xietaf = [flow(xe) for xe in xieta]
        flow = lambda *z: [xef(*z) for xef in xietaf]
        self._flow['2flow'] = {'xieta': xieta, 'xietaf': xietaf, 'flow': flow}
        
    def _calcFlow_njet(self, n_slices: int=1, **kwargs):
        '''
        Split and/or slice the current lie operator and compute its resulting flow, ready to
        take an njet.
        
        The flow maps are hereby calculated by any other method determined by the user. 
        
        Parameters
        ----------
        power: int, (optional)
            Control the number of nested commutators in case 'flow_method' == 'bruteforce'.
            Required if 'flow_method' is not specified (default).

        flow_method: str, optional
            The method to compute the flow of the individual slices, used in calculating the outcome
            of passing the jets through these slices.
            
        split_method: str, optional
            Customized splitting method of the hamiltonians. See lieops.solver.splitting for examples.
            
        n_slices: int, optional
            Slice the hamiltonian uniformly into n_slices parts before passing the njets.
            
        order: int, optional
            Control the order of the njets passed through the Lie operator. If nothing specified,
            a reasonable order will be attempted, depending on max_power of the current Lie operator argument.
            
        components: list, optional
            A list of poly objects which to apply the Lie operator onto. By default, self.components are taken.           
        **kwargs
            Optional keyworded arguments passed to the flow_method routine.
        '''
        kwargs.update(self._flow_parameters['njet']) # (++)
        
        # Step 1: Handle user input; determine method to calculate the flow of the individual parts.
        flow_method = kwargs.get('flow_method', 'bruteforce')
        assert flow_method != 'njet', "Argument 'flow_method' can not be 'njet' itself."
        if flow_method == 'bruteforce':
            try:
                _ = kwargs.setdefault('power', getattr(self, 'power'))
            except:
                raise RuntimeError("Default flow_method: bruteforce requires 'power' argument to be set.")

        # Step 2: Split hamiltonian into parts and compute their flows according to the requested parameters
        xieta = create_coords(dim=kwargs.get('dim', self.argument.dim), max_power=kwargs.get('max_power', self.argument.max_power))
        _ = kwargs.pop('method', None)
        _ = kwargs.pop('components', None) # We remove 'components' from kwargs here, so that at (+) we can use the default coordinate components for the flow, while using the other entries in kwargs for user-specified input of the njet_flow_method.
        if 'split_method' in kwargs.keys():
            # use a user-defined custom splitting method (see lieops.solver.splitting
            hamiltonians = kwargs['split_method'](self.argument/n_slices, **kwargs)
        else:
            hamiltonians = [self.argument/n_slices]
        # N.B. We do not multiply self.argument with self._flow_parameters['t'] for the hamiltonians here,
        # because at (++) we updated the flow parameters in the operators, thereby the t-parameter
        # will be recognized.
        _ = kwargs.pop('tpsa', False) # ensure that we do not use tpsa at (+), as we do that later.
        operators = [self.__class__(h) for h in hamiltonians]
        for op in operators:
            op.calcFlow(method=flow_method, components=xieta, **kwargs) # (+)
        result = {}
        result['unique_operators'] = operators
        operators = operators*n_slices
        def lo_concat(*z):
            for op in operators:
                z = op(*z)
            return z
        result.update({'flow': lo_concat, 'xieta': xieta, 'operators': operators, 'n_slices': n_slices, 'flow_method': flow_method})
        self._flow['njet'] = result
    
    def _calcPolyFromFlow(self, **kwargs):
        '''
        If self.flow has been computed, compute (or return) the resulting polynomial exp(:x:):y:, 
        for every y in self.components. This will be done by pulling back the polynomial :y:, using the
        values exp(:x:)z_j, where z_j are the xi/eta coordinates.
        '''
        method = self._flow_method
        flow_out = self._flow[method]
        if not 'poly' in flow_out.keys():
            parameters = self._flow_parameters[method]
            assert 'components' in parameters.keys()
            
            if 'xietaf' in flow_out.keys():
                xietaf = flow_out['xietaf']
            elif 'flow' in flow_out.keys():
                # compute the final xi/eta-coordinates from the bare flow function, using TPSA
                xietaf, dflow = self.tpsa(**kwargs)
                flow_out['xietaf'] = xietaf
                flow_out['dflow'] = dflow
                self._flow[method].update(flow_out)
            else:
                raise RuntimeError(f"No result(s) field present for method '{method}'.")
            # Compute the result by pullback: exp(:f:)Q(xi_1, ..., eta_n) = Q(exp(:f:) xi_1, ..., exp(:f:)eta_n) holds:
            flow_poly = [c(*xietaf) for c in parameters['components']]
            self._flow[method]['poly'] = flow_poly
            return flow_poly
        else:
            return lieoperator._calcPolyFromFlow(self, **kwargs)
    
    def __call__(self, *z, **kwargs):
        if isinstance(z[0], type(self)):
            return self.bch(*z, **kwargs) # Baker-Campbell-Hausdorff (using Magnus/combine routine)
        else:
            return lieoperator.__call__(self, *z, **kwargs)


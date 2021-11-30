from njet.jet import factorials, check_zero
from njet import derive
from .genfunc import genexp

# Note: The notation in this script (e.g. xi, eta), as well as parts of the background material, can 
# be found in my Thesis: Malte Titze -- Space Charge Modeling at the Integer Resonance for the CERN PS and SPS, on p.33 onwards.

class liepoly:
    '''
    Class to model the Lie operator :p:, where p is a polynomial given in terms of
    complex (xi, eta)-coordinates.
    
    self.max_power > 0 means that any calculations beyond this degree are discarded.
    For binary operations this means that the minimum of both fields are computed.
    '''
    def __init__(self, max_power=float('inf'), **kwargs):
        # self.dim denotes the number of xi (or eta)-factors.
        if 'values' in kwargs.keys():
            self.values = kwargs['values']
        elif 'a' in kwargs.keys() or 'b' in kwargs.keys():
            self.set_monomial(**kwargs)
        else:
            self.values = {}
            
        if len(self.values) == 0:
            self.dim = kwargs.get('dim', 0)
        else:
            self.dim = kwargs.get('dim', len(next(iter(self.values)))//2)
            
        self.set_max_power(max_power)
        
    def set_max_power(self, max_power):
        self.max_power = max_power
        self.values = {k: v for k, v in self.values.items() if sum(k) <= max_power}
        
    def set_monomial(self, a=[], b=[], value=1, **kwargs):
        dim = max([len(a), len(b)])
        if len(a) < dim:
            a += [0]*(dim - len(a))
        if len(b) < dim:
            b += [0]*(dim - len(b))
        self.values = {tuple(a + b): value}
        
    def maxdeg(self):
        '''
        Obtain the maximal degree of the current Lie polynomial. 
        '''
        if len(self.values) == 0:
            return 0
        else:
            return max([sum(k) for k, v in self.values.items()])
    
    def mindeg(self):
        '''
        Obtain the minimal degree of the current Lie polynomial. 
        '''
        if len(self.values) == 0:
            return 0
        else:
            return min([sum(k) for k, v in self.values.items()])
        
    def copy(self):
        return self.__class__(values={k: v for k, v in self.values.items()}, dim=self.dim, max_power=self.max_power)
    
    def extract(self, condition):
        '''
        Extract a Lie polynomial from the current Lie polynomial, based on a condition.
        
        Parameters
        ----------
        condition: callable
            A function which maps a given tuple (an index) to a boolean. For example 'condition = lambda x: sum(x) == k' would
            yield the homogeneous part of the current Lie polynomial (this is realized in 'self.homogeneous_part').
            
        Returns
        -------
        liepoly
            The extracted Lie polynomial.
        '''
        return self.__class__(values={key: value for key, value in self.values.items() if condition(key)}, dim=self.dim, max_power=self.max_power)
    
    def homogeneous_part(self, k: int):
        '''
        Extract the homogeneous part of order k from the current Lie polynomial.
        
        Parameters
        ----------
        k: int
            The requested order.
            
        Returns
        -------
        liepoly
            The extracted Lie polynomial.
        '''
        return self.extract(condition=lambda x: sum(x) == k)
        
    def __call__(self, z):
        '''
        Evaluate the polynomial at a specific position z.
        
        Parameters
        ----------
        z: subscriptable
            The point at which the polynomial should be evaluated. It is assumed that len(z) == self.dim
            (the components of z correspond to the xi-values).
        '''
        assert len(z) == 2*self.dim, f'Number of input parameters: {len(z)}, expected: {2*self.dim}'
        result = 0
        for k, v in self.values.items():
            prod = 1
            for j in range(self.dim):
                prod *= z[j]**k[j]*z[j + self.dim]**k[j + self.dim]
            result += v*prod
        return result
        
    def __add__(self, other):
        add_values = {k: v for k, v in self.values.items()}
        if self.__class__.__name__ != other.__class__.__name__:
            # Treat other object as constant.
            if other != 0:
                zero_tpl = (0,)*self.dim*2
                new_value = add_values.get(zero_tpl, 0) + other
                if not check_zero(new_value):
                    add_values[zero_tpl] = new_value
                else:
                    _ = add_values.pop(zero_tpl, None)
            max_power = self.max_power
        else:
            assert other.dim == self.dim
            for k, v in other.values.items():
                new_v = add_values.get(k, 0) + v
                if not check_zero(new_v):
                    add_values[k] = new_v
                else:
                    _ = add_values.pop(k, None)
            max_power = min([self.max_power, other.max_power])
        return self.__class__(values=add_values, dim=self.dim, max_power=max_power)
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return self.__class__(values={k: -v for k, v in self.values.items()}, 
                              dim=self.dim, max_power=self.max_power)
    
    def __sub__(self, other):
        return self + -other

    def __matmul__(self, other):
        return self.poisson(other)
        
    def poisson(self, other):
        '''
        Compute the Poisson-bracket {self, other}
        '''
        assert self.__class__.__name__ == other.__class__.__name__
        assert other.dim == self.dim
        max_power = min([self.max_power, other.max_power])
        poisson_values = {}
        for t1, v1 in self.values.items():
            power1 = sum(t1)
            for t2, v2 in other.values.items():
                power2 = sum(t2)
                if power1 + power2 - 2 > max_power:
                    continue
                a, b = t1[:self.dim], t1[self.dim:]
                c, d = t2[:self.dim], t2[self.dim:]
                for k in range(self.dim):
                    det = a[k]*d[k] - b[k]*c[k]
                    if det == 0:
                        continue
                    new_power = tuple([a[j] + c[j] if j != k else a[j] + c[j] - 1 for j in range(self.dim)] + \
                                [b[j] + d[j] if j != k else b[j] + d[j] - 1 for j in range(self.dim)])
                    new_value = poisson_values.get(new_power, 0) - 1j*det*v1*v2
                    if not check_zero(new_value):
                        poisson_values[new_power] = new_value
                    else:
                        _ = poisson_values.pop(new_power, None)
        return self.__class__(values=poisson_values, dim=self.dim, max_power=max_power)
    
    def __mul__(self, other):
        if self.__class__.__name__ == other.__class__.__name__:
            assert self.dim == other.dim
            dim2 = 2*self.dim
            max_power = min([self.max_power, other.max_power])
            mult_values = {}
            for t1, v1 in self.values.items():
                power1 = sum(t1)
                for t2, v2 in other.values.items():
                    power2 = sum(t2)
                    if power1 + power2 > max_power:
                        continue
                    prod_tpl = tuple([t1[k] + t2[k] for k in range(dim2)])
                    prod_val = mult_values.get(prod_tpl, 0) + v1*v2 # it is assumed that v1 and v2 are both not zero, hence prod_val != 0.
                    if not check_zero(prod_val):
                        mult_values[prod_tpl] = prod_val
                    else:
                        _ = mult_values.pop(prod_tpl, None)
            return self.__class__(values=mult_values, dim=self.dim, max_power=max_power)            
        else:
            return self.__class__(values={k: v*other for k, v in self.values.items() if not check_zero(other)}, 
                                          dim=self.dim, max_power=self.max_power)
        
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        # implement '/' operator
        if self.__class__.__name__ != other.__class__.__name__:
            # Attention: If other is a NumPy array, there is no check if one of the entries is zero.
            return self.__class__(values={k: v/other for k, v in self.values.items() if not check_zero(other)}, 
                                          dim=self.dim, max_power=self.max_power)
        else:
            raise NotImplementedError('Division by Lie polynomial not supported.')
        
    
    def __pow__(self, other):
        assert type(other) == int 
        assert other >= 0
        if other == 0:
            return self.__class__(values={(0,)*self.dim*2: 1}, 
                                  dim=self.dim, max_power=self.max_power) # N.B. 0**0 := 1

        remainder = other%2
        half = self**(other//2)
        if remainder == 1:
            return self*half*half
        else:
            return half*half
        
    def __len__(self):
        return len(self.values)
    
    def __eq__(self, other):
        return self.values == other.values
        
    def ad(self, y, power: int=1):
        '''
        Compute repeated Poisson-brackets.
        E.g. let x = self. Then {x, {x, {x, {x, {x, {x, y}}}}}} =: x**6(y)
        Special case: x**0(y) := y
        
        Let z be a homogeneous Lie polynomial and deg(z) the degree of z. Then it holds
        deg(x**m(y)) = deg(y) + m*(deg(x) - 2).
        This also holds for the maximal degrees and minimal degrees in case that x, and y
        are inhomogeneous.
        
        Therefore, if x and y both have non-zero max_power fields, 'self.power' will not evaluate
        terms x**m with
        m >= min([(max_power - mindeg(y))//(mindeg(x) - 2), power]).
        
        Parameters
        ----------
        y: liepoly
            Lie polynomial which we want to evaluate on

        power: int, optional
            Number of repeated brackets (default: 1).


        Returns
        -------
        list
            List [x**k(y) for k in range(n + 1)], if n is the power requested.
        '''
        assert self.__class__.__name__ == y.__class__.__name__
        assert power >= 0
        
        # Adjust requested power if max_power makes this necessary, see comment above.
        max_power = min([self.max_power, y.max_power])
        mindeg_x = self.mindeg()
        if mindeg_x > 2 and max_power < float('inf'):
            mindeg_y = y.mindeg()
            power = min([(max_power - mindeg_y)//(mindeg_x - 2), power]) # N.B. // works as floor division
            
        result = self.__class__(values={k: v for k, v in y.values.items()}, 
                                dim=y.dim, max_power=max_power)
        all_results = [result]
        # N.B.: We can not set values = self.values, otherwise result.values will get changed if self.values is changing.
        for k in range(power):
            result = self@result
            if len(result) == 0:
                break
            all_results.append(result)
        return all_results
    
    def __str__(self):
        return self.values.__str__()

    def _repr_html_(self):
        return f'<samp>{self.__str__()}</samp>'
    
    def derive(self, **kwargs):
        '''
        Derive the current Lie polynomial.
        
        Parameters
        ----------
        order: int
            The order by which we are going to derive the polynomial.
            
        Returns
        -------
        derive
            A class of type njet.ad.derive with n_args=2*self.dim parameters.
            Note that a function evaluation should be consistent with the fact that 
            the last self.dim entries are the complex conjugate values of the 
            first self.dim entries.
        '''
        return derive(self, n_args=2*self.dim, **kwargs)
    
    def flow(self, power: int, t=1, **kwargs):
        '''
        Let f: R^n -> R be a differentiable function and :x: the current polynomial Lie map. 
        Then this routine will compute the components of M: R^n -> R^n,
        where M is the map satisfying
        exp(:x:) f = f o M

        Note that the degree to which powers are discarded is given by self.max_power.

        Parameters
        ----------
        power: int
            The maximal power up to which exp(:x:) should be evaluated.
        
        t: float, optional
            An additional parameter to model exp(t*:x:) (default: 1). Note that
            this parameter can also be changed in the lieoperator class later.
            
        **kwargs
            Additional arguments are passed to the lieoperator class.

        Returns
        -------
        lieoperator
            Class of type lieoperator, modeling the flow of the current Lie polynomial.
        '''
        return lieoperator(self, generator=genexp(power), t=t, **kwargs)
        
    def compose(self, f, **kwargs):
        '''
        Let :x: represent the current Lie polynomial. Depending on the input,
        this routine will either return the map f(x) or the Lie polynomial :f(x):.
        
        Parameters
        ----------
        f: callable
            A function depending on a single parameter. It needs to be supported by the njet module.
            
        **kwargs
            Additional parameters passed to lie.compose routine.
            
        Returns
        -------
        callable or liepoly
            The output depends on the optional argument 'power'.
            
            If no argument 'power' has been passed, then it will
            be taken from the current value self.max_power.
            
            If power < float('inf'), then the Lie polynomial :f(x): is returned,
            where f has been expanded up to the specified power. If power == float('inf'),
            then the function f(x) is returned.
        '''
        if not 'power' in kwargs.keys():
            kwargs['power'] = self.max_power
        return compose([self], f, **kwargs)
            
    
def create_coords(dim, **kwargs):
    '''
    Create a set of complex (xi, eta)-Lie polynomials for a given dimension.
    
    Parameters
    ----------
    dim: int
        The requested dimension.
        
    **kwargs
        Optional arguments passed to liepoly class.
        
    Returns
    -------
    list
        List of length 2*dim with liepoly entries, corresponding to the xi_k and eta_k Lie polynomials. Hereby the first
        dim entries belong to the xi-values, while the last dim entries to the eta-values.
    '''
    resultx, resulty = [], []
    for k in range(dim):
        ek = [0 if i != k else 1 for i in range(dim)]
        xi_k = liepoly(a=ek, b=[0]*dim, dim=dim, **kwargs)
        eta_k = liepoly(a=[0]*dim, b=ek, dim=dim, **kwargs)
        resultx.append(xi_k)
        resulty.append(eta_k)
    return resultx + resulty


def compose(lps, f, power=float('inf')):
    r'''
    Let z = [z1, ..., zk] be Lie polynomials and f an analytical function, taking k values.
    Depending on the input, this routine will either return the Lie polynomial :f(z1, ..., zk): or
    the map f(z1, ..., zk).
    
    Parameters
    ----------
    lps: liepoly or list of liepoly objects
        The Lie polynomial(s) to be composed.
        
    f: callable
        A function on which we want to apply the list of liepoly objects.
        It needs to be supported by the njet module.
        
    power: int, optional
        The maximal power of the resulting Lie polynomial (default: float('inf')).
        If a value is provided, the routine will return a class of type liepoly, representing
        a Lie polynomial. If nothing is provided, the routine will return the function
        f(z1, ..., zk)
        
    Returns
    -------
    callable or liepoly
        As described above, depending on the 'power' input parameter, either the map f(z1, ..., zk) or
        the Lie polynomial :f(z1, ..., zk): is returned.
    '''
    if lps.__class__.__name__ == 'liepoly':
        lps = [lps]
    n_args_f = len(lps)
    dim_poly = lps[0].dim
    
    assert n_args_f == f.__code__.co_argcount, 'Input function depends on a different number of arguments.'
    assert all([lp.dim == dim_poly for lp in lps]), 'Input polynomials not all having the same dimensions.'

    composition = lambda z: f(*[lps[k](z) for k in range(n_args_f)])   
    if power == float('inf'):
        return composition
    else:
        dcomp = derive(composition, order=power, n_args=2*dim_poly)
        taylor_coeffs = dcomp([0]*2*dim_poly, mult=False)
        return liepoly(values=taylor_coeffs, dim=dim_poly, max_power=power)
    

def exp_ad(x, y, power, **kwargs):
    '''
    Compute the exponential Lie operator exp(:x:)y up to a given power.

    Parameters
    ----------
    x: liepoly
        The Lie polynomial in the exponent of exp.
    
    y: liepoly or list of liepoly objects
        The Lie polynomial(s) on which we want to apply the exponential Lie operator.
        
    power: int
        Integer defining the maximal power up to which we want to compute the result.
        
    **kwargs
        Additional arguments passed to the lieoperator class.

    Returns
    ------- 
    liepoly or list of liepoly objects
        Class of type liepoly, representing the result of the exponential Lie operator exp(:x:)y,
        up to the requested power.
    '''
    if y.__class__.__name__ == 'liepoly': # if the input is a Lie polynomial, also return a Lie polynomial
        return lieoperator(x, generator=genexp(power), components=[y], **kwargs).flow[0]
    else:
        return lieoperator(x, generator=genexp(power), components=y, **kwargs).flow


class lieoperator:
    '''
    Class to construct and work with a Lie operator of the form g(:x:).
    '''
    def __init__(self, x, **kwargs):
        self.exponent = x
        self.n_args = 2*self.exponent.dim
        if 'generator' in kwargs.keys():
            self.set_generator(**kwargs)
        if 'components' in kwargs.keys() or 't' in kwargs.keys():
            self.calcOrbits(**kwargs)
            self.calcFlow(**kwargs)
        
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
            taylor_coeffs = dg([0], mult=False)
            self.generator = [taylor_coeffs.get((k,), 0) for k in range(len(taylor_coeffs))]
        else:
            raise NotImplementedError('Input function not recognized.')
        self.power = len(self.generator) - 1
            
    def action(self, y, **kwargs):
        '''
        Apply the Lie operator g(:x:) to a given lie polynomial y to return the elements
        in the series of g(:x:)y.
        '''
        assert hasattr(self, 'generator'), 'No generator set (check self.set_generator).'
        ad_action = self.exponent.ad(y, power=self.power)
        assert len(ad_action) > 0
        # N.B. if self.generator[j] = 0, then k_action[j]*self.generator[j] = {}. It will remain in the list below (because 
        # it always holds len(k_action) > 0 by construction).
        # This is important when calculating the flow later on. In order to check for this consistency, we have added 
        # the above assert statement.
        return [ad_action[j]*self.generator[j] for j in range(len(ad_action))]
        
    def calcOrbits(self, **kwargs):
        '''
        Compute the summands in the series of the Lie operator g(:x:)y,
        for every requested y.
        
        Parameters
        ----------
        components: list, optional
            List of liepoly objects on which the Lie operator g(:x:) should be applied.
            If nothing specified, then the canonical coordinates are used.
            
        **kwargs
            Optional arguments passed to lie.create_coords.
        '''
        if 'components' in kwargs.keys():
            self.components = kwargs['components']
        else:
            self.components = create_coords(dim=self.exponent.dim, **kwargs) # run over all canonical coordinates.
        self.orbits = [self.action(y) for y in self.components]
        
    def calcFlow(self, t=1, **kwargs):
        '''
        Compute the Lie operators [g(t:x:)]y for every y in self.components.
        
        Parameters
        ----------
        t: float (or e.g. numpy.complex128 array)
            Parameter in the exponent at which the Lie operator should be evaluated.
        '''
        if not hasattr(self, 'orbits'):
            raise RuntimeError("No orbits found. Flow calculation requires at least one Lie polynomial to be transported (check self.calcOrbits).")
        # N.B. We multiply with the parameter t on the right-hand side, because if t is e.g. a numpy array, then
        # numpy would put the liepoly classes into its array, something we do not want. Instead, we want to
        # put the numpy arrays into our liepoly class.
        self.flow = [sum([self.orbits[k][j]*t**j for j in range(len(self.orbits[k]))]) for k in range(len(self.orbits))]
        self.flow_parameter = t
    
    def __call__(self, z, **kwargs):
        '''
        Compute the result of the current Lie operator g(:x:), applied to either a Lie polynomial
        or evaluated at a specific point.
        
        Parameters
        ----------
        z: subscriptable or liepoly
            The Lie polynomial or point of interest. 
            If z is a Lie polynomial, then the orbit of g(:x:)z will be computed and the flow returned as liepoly class.
            
            If z is a list, then the values (g(:x:)y)(z) for the given y in self.components
            are returned.
            
            Note that if an additional parameter t is passed, then the result of applying g(t*:x:) is calculated.
            
        **kwargs
            Optional arguments passed to self.calcFlow.
            
        Returns
        -------
        list or liepoly
            A list of length len(self.flow), representing the individual components of the Lie operator
            (g(:x:)y)(z), for every y in self.components, or the Lie polynomial g(:x:)z, depending on the input.
        '''
        if z.__class__.__name__ == 'liepoly':
            self.calcOrbits(components=[z], **kwargs)
            self.calcFlow(**kwargs)
            return self.flow[0]
        else:       
            assert hasattr(self, 'flow'), "Flow needs to be calculated first (check self.calcFlow)."
            if 't' in kwargs.keys(): # re-evaluate the flow at the requested flow parameter t.
                self.calcFlow(**kwargs)
            return [self.flow[k](z) for k in range(len(self.flow))]
    
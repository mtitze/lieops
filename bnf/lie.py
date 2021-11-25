from njet.jet import factorials, check_zero
from njet import derive

class liepoly:
    '''
    Class to model the Lie operator :p:, where p is a polynomial given in terms of
    complex (xi, eta)-coordinates.
    
    self.max_power > 0 means that any calculations beyond this degree are discarded.
    For binary operations this means that the minimum of both fields are computed.
    
    A liepoly class should be initialized with non-zero values.
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
        # It is assumed that the values of the Lie polynomial are non-zero.
        return max([sum(k) for k, v in self.values.items()])
    
    def mindeg(self):
        '''
        Obtain the minimal degree of the current Lie polynomial. 
        '''
        # It is assumed that the values of the Lie polynomial are non-zero.
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
        assert len(z) == 2*self.dim
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
            return self.__class__(values={k: v*other for k, v in self.values.items()}, 
                                          dim=self.dim, max_power=self.max_power)
        
    def __rmul__(self, other):
        return self.__mul__(other)
    
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
        
    def ad(self, y, power: int=1):
        '''
        Compute repeated Poisson-brackets.
        E.g. let x = self. Then {x, {x, {x, {x, {x, {x, y}}}}}} =: x**6(y)
        Special case: x**0(y) := y
        
        Let z be a homogeneous Lie polynomials and deg(z) the degree of z. Then it holds
        deg(x**m(y)) = deg(y) + m*(deg(x) - 2).
        This also holds for the maximal degrees and minimal degrees in case that x, and y
        are inhomogeneous.
        
        Therefore, if x and y both have non-zero max_power fields, 'self.power' will not evaluate
        terms x**m with
        m >= min([(max_power - mindeg(y))//(mindeg(x) - 2), power]).
        
        Parameters
        ----------
        y: self.__class___
            Class which we want to evaluate on

        power: int, optional
            Number of repeated brackets (default: 1).


        Returns
        -------
        list
            List [x**k(y) for k in range(n)], if n is the power requested.
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
        # N.B.: We can not set values = self.values, otherwise result.values will get changed if self.values is changing.
        
        all_results = []
        for k in range(power):
            result = self@result
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
        derive: object
            A class of type njet.ad.derive with n_args=2*self.dim parameters.
            Note that a function evaluation should be consistent with the fact that 
            the last self.dim entries are the complex conjugate values of the 
            first self.dim entries.
        '''
        return derive(self, n_args=2*self.dim, **kwargs)
    
    def pullback(self, power: int, t=1, **kwargs):
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
            An additional parameter to compute exp(t*:x:) instead.
            
        **kwargs
            Additional arguments are passed to the liemap class.

        Returns
        -------
        liemap
            class of type liemap containing the map belonging to the current Lie polynomial.
        '''
        return liemap(self, power=power, t=t, **kwargs)
        
    def compose(self, lps):
        '''
        Let :x: represent the current Lie polynomial and [:y1:, :y2:, ...] a list
        of Lie operators of length x.dim.
        Then this routine will compute the Lie polynomial :x(yk):, where x(yk) is
        the polynomial in which yk has been applied to the elementary element zk of x.
        '''
        dim2 = self.dim*2
        assert len(lps) == dim2

        # A list of elementary Lie polynomials
        z = [liepoly(values={tuple([0 if j != k else 1 for j in range(dim2)]):1} , dim=self.dim, max_power=self.max_power) for k in range(dim2)]
        yz = [lps[k]*z[k] for k in range(dim2)]
        
        for tpl, value in self.values.items():
            power1 = sum(tpl)
            for k in range(dim2):
                if tpl[k] == 0:
                    continue
                    
                for tpl2, value2 in yz[k].values.items():
                    if sum(tpl2) + power1 > self.max_power:
                        continue
                        
                    tuple([tpl2[l] + tpl[l] for l in range(dim2)])
            # now construct new lie polynomial based on the powers
            
    
def create_xieta(dim, **kwargs):
    '''
    Create a set of (xi, eta)-Lie polynomials for a given dimension.
    
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
    
    
def exp_ad(x, y, power: int):
    '''
    Compute the exponential Lie operator exp(:x:)y up to a given power.
    
    Parameters
    ----------
    x: liepoly
        The polynomial defining the Lie operator :x:
    y: liepoly
        The polynomial of which we want to apply the exponential Lie operator on.
    power: int
        Integer defining the maximal power up to which we want to compute the expression.
        
    Returns
    ------- 
    list
        List containing the terms 1/k!*(:x:**k)y in the exponential expansion up to the given power.
    '''
    facts = factorials(power)
    powers = x.ad(y, power)
    for k in range(len(powers)): # powers[0] corresponds to {x, y}, i.e. order 1
        powers[k] = 1/facts[k + 1]*powers[k]
    return [y] + powers
            
    
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
    # N.B. We multiply with the parameter t on the right-hand side, because if t is e.g. a numpy array, then
    # numpy would put the liepoly classes into its array, something we do not want. Instead, we want to
    # put the numpy arrays into our liepoly class.
    return [e[k]*t**k for k in range(len(e))]

    
class liemap:
    '''
    Class to conveniently manage a collection of Lie polynomials which correspond to a map
    M: R^n -> R^m.
    '''
    def __init__(self, lp, **kwargs):
        self.exponent = lp
        self.n_args = 2*self.exponent.dim
        if 'power' in kwargs.keys():
            self.build(**kwargs)
        
    def build(self, power, **kwargs):
        '''
        Compute the summands in the series of the exponential Lie operator exp(:x:)z_k,
        for every requested k.
        
        Parameters
        ----------
        power: int
            The maximal power up to which exp(:x:) should be evaluated.
        
        components: list, optional
            List of integers denoting the components to be computed (i.e. the k-indices described above). 
            If nothing specified, all components are calculated.
        '''
        # N.B. exp(:x:)eta required if x has complex entries (TODO: perhaps find a trick to avoid the calculation...)
        self.power = power
        components = kwargs.get('components', range(self.n_args))
        self.n_values = len(components)
        xieta = create_xieta(dim=self.exponent.dim, **kwargs)
        self.series = []
        for k in components:
            self.series.append(exp_ad(self.exponent, xieta[k], power=power))
        self.exponent_parameter = kwargs.get('t', 1)
        self.eval(**kwargs)
        
    def eval(self, t=1, **kwargs):
        '''
        Compute the exponential Lie operators exp(t:x:)z_k for k in range(self.n_values).
        '''
        if not hasattr(self, 'series'):
            raise RuntimeError("Map needs to be build before evaluation with 'self.build'.")
        if not check_zero(t - self.exponent_parameter) or not hasattr(self, 'exp'):
            self.exp = [sum(exp_ad_par(self.series[k], t=t)) for k in range(self.n_values)]
            self.exponent_parameter = t
    
    def __call__(self, z, **kwargs):
        self.eval(**kwargs) # This does nothing if self.exp already exists and t has not been changed.
        return [self.exp[k](z) for k in range(self.n_values)]
    



# This file contains the basic functionalities of the poly class. Since the class will be derived later,
# it is not recommended to construct poly objects directly from that file.
import numpy as np

from njet import derive, jetpoly, jet
from njet.common import check_zero, factorials, nCr
from njet.extras import general_faa_di_bruno

def _shift_component(shift_value, shift_component, terms, **kwargs):
    '''
    Shift the terms of a polynomial by a given value and component index, using binomial coefficients.
    
    Parameters
    ----------
    shift_value: float or complex
        The value to be shifted
    
    shift_component: int
        The index of the component to be shifted
        
    terms: dict
        A dictionary, mapping tuples to (e.g.) float or complex.
        The terms defining the indices and their exponents of a polynomial.
        
    Returns
    -------
    dict
        A dictionary defining the indices and powers of the shifted polynomial.
    
    Example
    =======
    shift_component(0.1, 3, {(0, 0, 0, 2): 7})
    # shifts the fourth component by 0.1. If we denote this component by x, then
    # the rule would be:    
    # 7*x**2 -> 7*(x + 0.1)**2 = 7*x**2 + 14*x*0.1 + 7*0.1**2 = 7*x**2 + 1.4*x + 0.07
    # The result is
    {(0, 0, 0, 2): 7.0, (0, 0, 0, 1): 1.4, (0, 0, 0, 0): 0.07}
    '''
    if check_zero(shift_value):
        return terms
    
    required_powers = set(k[shift_component] for k in terms.keys() if k[shift_component] > 0)
    if len(required_powers) == 0:
        return terms
    
    max_power = max(required_powers)
    shift_powers = np.arange(max_power + 1)
    new_shift_values = shift_value**shift_powers
    
    if 'binomials' not in kwargs.keys():
        binomials = nCr(max_power)
    else:
        binomials = kwargs['binomials']
    
    new_terms = {}
    for powers, value in terms.items():
        power_component = powers[shift_component] # the power of the component in question
        if power_component == 0:
            new_terms[powers] = new_terms.get(powers, 0) + value
            continue

        sp = shift_powers[:power_component + 1]
        new_keys = (tuple(powers[k] if k != shift_component else l for k in range(len(powers))) for l in sp[::-1])

        nsv = new_shift_values[:power_component + 1]
        b = binomials[power_component]
        nsvb = nsv*b
        try:
            new_values = value*nsvb
        except: # 'value' could be some numpy array. We try:
            new_values = np.outer(nsvb, value)

        j = 0
        for nk in new_keys:
            new_terms[nk] = new_terms.get(nk, 0) + new_values[j]
            j += 1
            
    return new_terms

class _poly:
    '''
    Class to model the Lie operator :p:, where p is a polynomial given in terms of
    complex (xi, eta)-coordinates. For the notation of these coordinates see Ref. [1] p. 33 onwards.
    In the literature this polynomial itself may be called a "Lie polynomial", see e.g. Ref. [2].
    
    Parameters
    ----------
    values: dict, optional
        A dictionary assigning the powers of the xi- and eta-variables to coefficients, modeling monomials.
        Note that it is internally assumed that every coefficient is non-zero and so zero-coefficients will
        be discarded. Powers which do not appear in the dictionary are assumed to be zero.
        
    a: tuple, optional
        If no values specified, then one can specify a tuple of integers a = (a1, ..., an) 
        to set the powers of the monomial xi = xi_1**a1 * ... * xi_n**an.
        
    b: tuple, optional
        Similar to 'a', this tuple will define the powers of the monomial belonging to the eta-variables.
    
    max_power: int, optional
        An integer beyond we do not perform calculations. In particular it is assumed for the poisson
        bracket that terms beyond that integer are zero.
        Attention:
        The poisson bracket of two homogeneous Lie polynomials of orders n and m has order K = m + n - 2. 
        For a general polynomial, we have in order K a contribution of m which could be e.g. of order K + 1 
        (with n = 1). This means that orders K + 1 will in general
        contribute to the K-th order in the poisson bracket, making repeated application of
        poisson brackets not reliable if max_power is set too small.

        Specificially, if we set max_power = K for two Lie polynomials a and b, then in {a, b} only those powers up and including K - 1 can be expected to be correct.
        
    dim: int, optional
        The number of xi- (or eta-) variables. Will be determined automatically from the input, if nothing
        specified.
        
    **kwargs
        Optional arguments passed to self.set_monimial and self.set_max_power
        
    Reference(s):
        [1] Titze, M. "Space Charge Modeling at the Integer Resonance for the CERN PS and SPS" (2019).
        [2] Casas, F. and Murua, M. "An efficient algorithm for computing the Baker-Campbell-Hausdorff series and some of its applications" (2008) arXiv:0810.2656.
    '''
    
    def __init__(self, **kwargs):
        # self.dim denotes the number of xi (or eta)-factors.
        self._poisson_factor = kwargs.get('poisson_factor', -1j) 
        # Factor to be multiplied with the canonical symplectic structure
        #
        #        / 0  1 \
        #   J = |       |
        #       \-1  0 /
        #
        # to determine the poisson bracket of two poly objects (see self.__matmul__). 
        # Therefore one can conveniently control if (e.g) the polynomial is given in terms 
        # of xi/eta or q/p variables.
        
        if 'values' in kwargs.keys():
            self._values = {k: v for k, v in kwargs['values'].items() if not check_zero(v)}
        elif 'a' in kwargs.keys() or 'b' in kwargs.keys(): # simplified building
            self.set_monomial(**kwargs)
        else:
            self._values = {}
            
        if len(self._values) == 0:
            self.dim = kwargs.get('dim', 0)
        else:
            self.dim = kwargs.get('dim', len(next(iter(self._values)))//2)
            
        self.set_max_power(**kwargs)
        
    def set_max_power(self, max_power=float('inf'), **kwargs):
        '''
        Set the maximal power to be taken into consideration.
        Attention: This operation will discard the current values *without* recovery.
        
        Parameters
        ----------
        max_power: int, optional
            A value > 0 means that any calculations leading to expressions beyond this 
            degree will be discarded. For binary operations the minimum of both 
            max_powers are used.
        '''
        self.max_power = max_power
        self._values = {k: v for k, v in self.items() if sum(k) <= max_power}
        
    def set_monomial(self, a=[], b=[], value=1, **kwargs):
        dim = max([len(a), len(b)])
        if len(a) < dim:
            a += [0]*(dim - len(a))
        if len(b) < dim:
            b += [0]*(dim - len(b))
        self._values = {tuple(a + b): value}
        
    def maxdeg(self):
        '''
        Obtain the maximal degree of the current polynomial. 
        '''
        if len(self._values) == 0:
            return 0
        else:
            return max([sum(k) for k, v in self.items()])
    
    def mindeg(self):
        '''
        Obtain the minimal degree of the current polynomial. 
        '''
        if len(self._values) == 0:
            return 0
        else:
            return min([sum(k) for k, v in self.items()])
        
    def copy(self):
        new_values = {}
        for k, v in self.items():
            if hasattr(v, 'copy'):
                v = v.copy()
            new_values[k] = v
        return self.__class__(values=new_values, dim=self.dim, max_power=self.max_power)
    
    def extract(self, *keys, key_cond=lambda x: True, value_cond=lambda x: True):
        '''
        Extract a Lie polynomial from the current Lie polynomial, based on a condition.
        
        Parameters
        ----------
        key_cond: callable, optional
            A function which maps a given tuple (an index) to a boolean. key_cond is used to enforce a condition
            on the keys of the current polynomial. For example 'key_cond = lambda x: sum(x) == k' would
            yield the homogeneous part of the current Lie polynomial (this is realized in 'self.homogeneous_part').

        value_cond: callable, optional
            A function which maps a given value to a boolean. value_cond is used to enforce a condition on the values of the
            current polynomial.

        Returns
        -------
        poly
            The extracted Lie polynomial.
        '''
        if len(keys) == 0:
            return self.__class__(values={keyc: valuec for keyc, valuec in self.items() if key_cond(keyc) and value_cond(valuec)}, 
                                  dim=self.dim, max_power=self.max_power)
        else:
            return self.__class__(values={key: self[key] for key in keys}, dim=self.dim, max_power=self.max_power)
    
    def homogeneous_part(self, k: int):
        '''
        Extract the homogeneous part of order k from the current Lie polynomial.
        
        Parameters
        ----------
        k: int
            The requested order.
            
        Returns
        -------
        poly
            The extracted polynomial.
        '''
        return self.extract(key_cond=lambda x: sum(x) == k)
    
    def above(self, tol: float):
        '''
        Drop values below a given threshold.
        
        Parameters
        ----------
        tol: float
            The threshold.
            
        Returns
        -------
        poly
            A polynomial having the same keys/values as the current polynomial, but the absolute values are larger
            than the requested threshold.
        '''
        def value_cond(x):
            result = abs(x) > tol
            if not hasattr(result, '__iter__'):
                return result
            else:
                return result.all() # assuming a numpy array
        return self.extract(value_cond=value_cond)
    
    def below(self, tol: float):
        '''
        Drop values above a given threshold.
        
        Parameters
        ----------
        tol: float
            The threshold.
            
        Returns
        -------
        poly
            A polynomial having the same keys/values as the current polynomial, but the absolute values are smaller
            than the requested threshold.
        '''
        def value_cond(x):
            result = abs(x) < tol
            if not hasattr(result, '__iter__'):
                return result
            else:
                return result.all() # assuming a numpy array
        return self.extract(value_cond=value_cond)
    
    def monomials(self):
        '''
        Return a list of the monomials of the current polynomial.
        '''
        return [self.__class__(values={k: v}, dim=self.dim, max_power=self.max_power) for k, v in self.items()]
    
    def truncate(self, k):
        '''
        Truncate the current polynomial with respect to a total power.
        '''
        return self.extract(key_cond=lambda x: sum(x) <= k)

    def project(self, *projection):
        '''
        Project the current polynomial to a specific subspace, by dropping monomials which contain indices belonging to other dimensions.

        Example:
        self.project(0)
        will leave terms containing powers of the form (i_0, 0, 0, ..., i_{self.dim}, 0, 0).
        '''
        # Input consistency checks
        new_dim = len(projection)
        assert new_dim > 0, 'Subspace not specified.'
        assert new_dim <= self.dim, 'Requested dimension too large.'
        assert max(projection) < self.dim, 'At least one dimension-index larger than current dimension.'

        # Perform the projection
        projection = list(projection) + [p + self.dim for p in projection] # the eta-components duplicate the indices.
        complement = [k for k in range(2*self.dim) if not k in projection]
        new_values = {}
        for k, v in self.items():
            if any([k[p] != 0 for p in complement]): # only keep those coefficients which do not couple to other directions
                continue
            new_values[tuple([k[p] for p in projection])] = v
        return self.__class__(values=new_values, dim=new_dim, max_power=self.max_power)
    
    def _prepare_input(self, *z):
        '''
        Consistency check & input preparation prior to evaluation.
        '''
        if len(z) == self.dim and self._poisson_factor == -1j:
            # for xi/eta-variables we conveniently conjugate the remaining components if they are not explicitly given
            z = [e for e in z] + [e.conjugate() for e in z]
        dim2 = self.dim*2
        assert len(z) == dim2, f'Number of input parameters: {len(z)}, expected: {dim2} (or {self.dim})'
        return z
    
    def insert(self, *z, **kwargs):
        '''
        Insert other polynomials into the current polynomial, to yield a new polynomial.
        
        This routine is called also by self.__call__, if lie polynomials are inserted.
        
        Parameters
        ----------
        *z: polynomials
            The polynomials at which the current polynomial should be evaluated at.
            
        max_order: int, optional
            An integer defining the maximal order up to which the result should be computed.
            
        Returns
        -------
        jet
            The jet corresponding to the composition of the polynomials.
        '''        
        if 'max_order' not in kwargs.keys():
            max_order = max([e.maxdeg() for e in z] + [self.maxdeg()])
        else:
            max_order = kwargs['max_order']
        
        # Combine the respective jets which represent the involved polynomials (this is much faster than using self.evaluate on polynomials):
        zjets = [e.to_jet(max_order=max_order) for e in z]
        return general_faa_di_bruno([self.shift([zj.array(0) for zj in zjets]).to_jet(max_order=max_order)], zjets)[0]        
        
    def evaluate(self, *z, **kwargs):
        '''
        Insert other objects into the current polynomial, to yield a new object.
        
        This routine is called also by self.__call__, depending on the input type.
        '''        
        # compute the occuring powers ahead of evaluation
        z_powers = {}
        j = 0
        for we in zip(*self.keys()):
            z_powers[j] = {k: z[j]**int(k) for k in np.unique(we)} # need to convert k to int, 
            # otherwise we get a conversion to some numpy array if z is not a float (e.g. an njet).
            j += 1
        
        # evaluate polynomial at requested point
        if isinstance(self, type(z[0])):
            # ensure that the result is of the same type as the input in case self.items is empty
            result = self.__class__(values={}, dim=self.dim, max_power=self.max_power)
        else:
            result = 0
            
        for k, v in self.items():
            prod = 1
            for j in range(self.dim):
                prod *= z_powers[j][k[j]]*z_powers[j + self.dim][k[j + self.dim]]
            result += prod*v # v needs to stay on the right-hand side here, because prod may be a jet class (if we compute the derivative(s) of the Lie polynomial)
            
        return result
    
    def __call__(self, *z, **kwargs):
        '''
        Evaluate the polynomial at a specific position z.
        
        Parameters
        ----------
        z: subscriptable
            The point at which the polynomial should be evaluated. It is assumed that len(z) == self.dim,
            in which case the components of z are assumed to be xi-values. Otherwise, it is assumed that len(z) == 2*self.dim,
            where z = (xi, eta) denote a set of complex conjugated coordinates.
        '''
        # prepare input vector
        z = self._prepare_input(*z)
        
        # check input type
        if all([isinstance(e, type(self)) for e in z]):
            return self.insert(*z, **kwargs)
        else:
            return self.evaluate(*z, **kwargs)
        
    def __add__(self, other):
        if check_zero(other):
            return self
        add_values = {k: v for k, v in self.items()}
        if not isinstance(self, type(other)):
            # Treat other object as constant.
            zero_tpl = (0,)*self.dim*2
            add_values[zero_tpl] = add_values.get(zero_tpl, 0) + other
            max_power = self.max_power
        else:
            assert self.dim == other.dim, f'Dimensions do not agree: {self.dim} != {other.dim}'
            for k, v in other.items():
                add_values[k] = add_values.get(k, 0) + v
            max_power = min([self.max_power, other.max_power])
        return self.__class__(values=add_values, dim=self.dim, max_power=max_power)
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return self.__class__(values={k: -v for k, v in self.items()}, 
                              dim=self.dim, max_power=self.max_power)
    
    def __sub__(self, other):
        return self + -other
    
    def __rsub__(self, other):
        return -self + other

    def __matmul__(self, other):
        return self.poisson(other)
        
    def poisson(self, other):
        '''
        Compute the Poisson-bracket {self, other}
        '''
        if not isinstance(self, type(other)):
            raise TypeError(f"unsupported operand type(s) for poisson: '{self.__class__.__name__}' and '{other.__class__.__name__}'.")
        assert self.dim == other.dim, f'Dimensions do not agree: {self.dim} != {other.dim}'
        max_power = min([self.max_power, other.max_power])
        poisson_values = {}
        for t1, v1 in self.items():
            power1 = sum(t1)
            for t2, v2 in other.items():
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
                    poisson_values[new_power] = v1*v2*det*self._poisson_factor + poisson_values.get(new_power, 0)
        return self.__class__(values=poisson_values, dim=self.dim, max_power=max_power)
    
    def __mul__(self, other):
        if isinstance(self, type(other)):
            assert self.dim == other.dim
            dim2 = 2*self.dim
            max_power = min([self.max_power, other.max_power])
            mult_values = {}
            for t1, v1 in self.items():
                power1 = sum(t1)
                for t2, v2 in other.items():
                    power2 = sum(t2)
                    if power1 + power2 > max_power:
                        continue
                    prod_tpl = tuple([t1[k] + t2[k] for k in range(dim2)])
                    mult_values[prod_tpl] = mult_values.get(prod_tpl, 0) + v1*v2 # it is assumed that v1 and v2 are both not zero, hence prod_val != 0.
            return self.__class__(values=mult_values, dim=self.dim, max_power=max_power)
        else:
            return self.__class__(values={k: v*other for k, v in self.items()}, dim=self.dim, max_power=self.max_power) # need to use v*other; not other*v here: If type(other) = numpy.float64, then it may cause unpredicted results if it stands on the left.
        
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        # implement '/' operator
        if not isinstance(self, type(other)):
            # Attention: If other is a NumPy array, there is no check if one of the entries is zero.
            return self.__class__(values={k: v/other for k, v in self.items()}, dim=self.dim, max_power=self.max_power)
        else:
            raise NotImplementedError('Division by Lie polynomial not supported.')
        
    def __pow__(self, other):
        assert type(other) == int
        assert other >= 0
        if other == 0:
            return self.__class__(values={(0,)*self.dim*2: 1}, 
                                  dim=self.dim, max_power=self.max_power) # N.B. 0**0 := 1
        elif other == 1:
            return self
        else:
            remainder = other%2
            half = self**(other//2)
            if remainder == 1:
                return self*half*half
            else:
                return half*half
        
    def conjugate(self):
        return self.__class__(values={k[self.dim:] + k[:self.dim]: v.conjugate() for k, v in self.items()},
                              dim=self.dim, max_power=self.max_power)
        
    def __len__(self):
        return len(self._values)
    
    def __eq__(self, other):
        if isinstance(self, type(other)):
            return self._values == other._values
        else:
            if self.maxdeg() != 0:
                return False
            else:
                return self.get(tuple([0]*2*self.dim), 0) == other
            
    def keys(self):
        return self._values.keys()
    
    def get(self, *args, **kwargs):
        return self._values.get(*args, **kwargs)
    
    def items(self):
        return self._values.items()
    
    def values(self):
        return self._values.values()
    
    def __iter__(self):
        for key in self._values.keys():
            yield self._values[key]
            
    def __getitem__(self, key):
        return self._values[key]
    
    def __setitem__(self, key, other):
        self._values[key] = other
        
    def pop(self, *args, **kwargs):
        new_values = {k: v for k, v in self._values.items()}
        new_values.pop(*args, **kwargs)
        return self.__class__(values=new_values, dim=self.dim, max_power=self.max_power)
        
    def update(self, d):
        new_values = {k: v for k, v in self.items()}
        new_values.update(d)
        return self.__class__(values=new_values, dim=self.dim, max_power=self.max_power)
        
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
        y: poly
            Lie polynomial which we want to evaluate on

        power: int, optional
            Number of repeated brackets (default: 1).


        Returns
        -------
        list
            List [x**k(y) for k in range(n + 1)], if n is the power requested.
        '''
        if not isinstance(self, type(y)):
            raise TypeError(f"unsupported operand type(s) for adjoint: '{self.__class__.__name__}' on '{y.__class__.__name__}'.")
        assert power >= 0
        
        # Adjust requested power if max_power makes this necessary, see comment above.
        max_power = min([self.max_power, y.max_power])
        mindeg_x = self.mindeg()
        if mindeg_x > 2 and max_power < float('inf'):
            mindeg_y = y.mindeg()
            power = min([(max_power - mindeg_y)//(mindeg_x - 2), power]) # N.B. // works as floor division
            
        result = self.__class__(values={k: v for k, v in y.items()}, 
                                dim=y.dim, max_power=max_power)
        all_results = [result]
        # N.B.: We can not set values = self.values, otherwise result.values will get changed if self.values is changing.
        for k in range(power):
            result = self@result
            if len(result) == 0:
                break
            all_results.append(result)
        return all_results
    
    def __abs__(self):
        return self.apply(abs)
    
    def __str__(self):
        out = ''
        for k, v in self.items():
            out += f'{k}: {str(v)} '
        if len(out) > 0:
            return out[:-1]
        else:
            return '0'

    def _repr_html_(self):
        return f'<samp>{self.__str__()}</samp>'
    
    def derive(self, **kwargs):
        '''
        Derive the current Lie polynomial.
        
        Parameters
        ----------
        order: int
            The order by which we are going to derive the polynomial.
            
        **kwargs
            Optional keyword arguments passed to njet.derive
            
        Returns
        -------
        derive
            A class of type njet.ad.derive with n_args=2*self.dim parameters.
            Note that a function evaluation should be consistent with the fact that 
            the last self.dim entries are the complex conjugate values of the 
            first self.dim entries.
        '''
        kwargs['n_args'] = kwargs.get('n_args', 2*self.dim)
        return derive(self, **kwargs)
        
    def construct(self, f, **kwargs):
        '''
        Let :x: represent the current Lie polynomial. Depending on the input,
        this routine will either return the map f(x) or the Lie polynomial :f(x):.
        
        Parameters
        ----------
        f: callable
            A function depending on a single parameter. It needs to be supported by the njet module.
            
        **kwargs
            Additional parameters passed to lie.construct routine.
            
        Returns
        -------
        callable or poly
            The output depends on the optional argument 'power'.
            
            If no argument 'power' has been passed, then it will
            be taken from the current value self.max_power.
            
            If power < float('inf'), then the Lie polynomial :f(x): is returned,
            where f has been expanded up to the specified power. If power == float('inf'),
            then the function f(x) is returned.
        '''
        if not 'power' in kwargs.keys():
            kwargs['power'] = self.max_power
        return construct([self], f, **kwargs)
    
    def to_jetpoly(self):
        '''
        Map the current Lie polynomial to an njet jetpoly class.
        
        Returns
        -------
        jetpoly
            A jetpoly class of self.dim*2 variables, representing the current Lie polynomial.
        '''
        # N.B. self.dim corresponds to the number of xi (or eta) variables.
        # Although xi and eta are related by complex conjugation, we need to treat them as being independently,
        # in line with Wirtinger calculus. However, this fact needs to be taken into account when evaluating those polynomials, so
        # a polynomial should be evaluated always at points [z, z.conjugate()] etc.

        constant_key = (0,)*self.dim*2
        jpvalues = {}
        if constant_key in self.keys():
            jpvalues[frozenset([(0, 0)])] = self._values[constant_key]
        for key, v in self.items():
            if sum(key) == 0: # we already dealt with the constant term.
                continue
            jpvalues[frozenset([(j, key[j]) for j in range(self.dim*2) if key[j] != 0])] = v
        return jetpoly(terms=jpvalues)

    def to_jet(self, max_order=np.inf, **kwargs):
        '''
        Map the current Lie polynomial to an njet jet class.

        Returns
        -------
        jet
            A jet class of self.dim*2 variables, representing the current Lie polynomial.
        '''
        # Development comment: 
        # TODO: 1) May add mult_prm and mult_drv options
        #       2) To think about:
        #          Do this in njet, so the flow would be: to_jetpoly -> to_jet? 
        #          and here the conversion to_jetpoly will be done, then to_jet?
        #       3) Note also that to_jet and to_jetpoly currently give different results due to the factorials involved here.
        dim2 = self.dim*2
        constant_key = (0,)*dim2
        if max_order == np.inf:
            deg = self.maxdeg()
        else:
            deg = max_order
        jet_array = [0] + [{} for k in range(deg)]
        
        if 'factorials' not in kwargs.keys():
            facts = factorials(deg)
        else:
            facts = kwargs['factorials']

        if constant_key in self.keys():
            jet_array[0] = self._values[constant_key]

        for key, v in self.items():
            order = sum(key)
            if order == 0 or order > deg: # we already dealt with the constant term.
                continue
            jet_array[order].update({frozenset((j, key[j]) for j in range(dim2) if key[j] != 0): v*facts[order]})
        return jet(*([jet_array[0]] + [jetpoly(terms=ja) for ja in jet_array[1:]]))

    def apply(self, operator, *args, **kwargs):
        '''
        Apply an operator to the coefficients of the current Lie-polynomial.
        
        Parameters
        ----------
        operator: callable
            The operator.
            
        *args
            Optional parameters passed to the operator.
            
        **kwargs
            Optional keyworded parameters passed to the operator.
            
        Returns
        -------
        poly
            A Lie-polynomial having the mapped values.
        '''
        new_values = {k: operator(v, *args, **kwargs) for k, v in self.items()}
        return self.__class__(values=new_values, dim=self.dim, max_power=self.max_power)
    
    def applyClassFunc(self, name, cargs={}, *args, **kwargs):
        '''
        Apply a class function of the coefficients of the current Lie-polynomial.
        
        Parameters
        ----------
        name: str
            The name of the class function 'name'.
            
        cargs: dict, optional
            Dictionary of keywords which may depend on self.values.keys(). This means that the keys of
            cargs must correspond to self.values.keys(). The items of cargs correspond to a set of keyworded
            arguments for the class function 'name'.
            
        *args:
            Arguments of the class function 'name'.
            
        **kwargs:
            Keyworded arguments of the class function 'name'.
            
        Returns
        -------
        poly
            A Lie-polynomial in which every entry in its values contain the result of the requested class function.
        '''
        if len(cargs) > 0:
            out = {key: getattr(v, name)(*args, **cargs[key]) for key, v in self.items()}
        else:
            out = {key: getattr(v, name)(*args, **kwargs) for key, v in self.items()}
        return self.__class__(values=out, dim=self.dim, max_power=self.max_power)
    
    def realBasis(self):
        '''
        Note that it holds xi = (q + 1j*p)/sqrt2, eta = (q - 1j*p)/sqrt2.
        '''
        sqrt2 = float(np.sqrt(2))
        def trf(q, p):
            xi = (q + p*1j)/sqrt2
            eta = (q - p*1j)/sqrt2 # = xi.conjugate()
            return xi, eta
        return self.transform(trf=trf, poisson_factor=1)
        
    def complexBasis(self):
        '''
        Note that it holds xi = (q + 1j*p)/sqrt2, eta = (q - 1j*p)/sqrt2.
        '''
        sqrt2 = float(np.sqrt(2))
        def trf(xi, eta):
            q = (xi + eta)/sqrt2
            p = (xi - eta)/sqrt2/1j
            return q, p
        return self.transform(trf=trf, poisson_factor=-1j)
        
    def transform(self, trf, poisson_factor, mult_drv=False, mult_prm=False):
        '''
        Cast the current polynomial into a different form.
        
        Parameters
        ----------
        trf: callable
            A linear transformation mapping a pair of new variables q2, p2 to original variables q1, p1.
            
        poisson_factor: The factor f so that {q1, p1}_{q1, p1} = f*{q1, p1}_{q2, p2} hold, where
        the brackets are given with respect to the old/new coordinates. For example, if q1 = q, p1 = p
        and q2 = xi = (q + p*1j)/sqrt2, p2 = eta = (q - p*1j)/sqrt2, then
        1 = {q1, p1}_{q1, p1} and -1j = {q1, p1}_{xi, eta}. Therefore, if transforming from q/p to xi/eta,
        the poisson factor f has to be -1j.
        
        At the moment only 2d transformations are supported; more sophisticated transformations require
        some overwork in the poisson bracket routine self.__matmul__.
        
        mult_drv: boolean, optional
            Control of factorial and permutation coefficients. See njet.poly.jetpoly.taylor_coefficients for details.
            
        mult_prm: boolean, optional
            Control of factorial and permutation coefficients. See njet.poly.jetpoly.taylor_coefficients for details.

        Returns
        -------
        dict
            The coefficients of the polynomial with respect to the real q and p coordinates.
            The keys correspond to the powers with respect to (q_1, ..., q_dim, p1, ..., p_dim).
        ''' 
        assert self.dim > 0
        sqrt2 = float(np.sqrt(2))
        q2, p2 = [], []
        for k in range(self.dim):
            # we insert polynomials with coefficients 1 and power 1 into the current Hamiltonian
            qk = jetpoly(1, index=k, power=1)
            pk = jetpoly(1, index=k + self.dim, power=1)
            q2k, p2k = trf(qk, pk)
            q2.append(q2k)
            p2.append(p2k)
        h1 = self(*(q2 + p2))
        if h1 == 0:
            rbv = {(0,)*self.dim*2: 0}
        else:
            rbv = h1.taylor_coefficients(2*self.dim, facts=factorials(self.maxdeg()), 
                                          mult_drv=mult_drv, mult_prm=mult_prm)
        return self.__class__(values=rbv, poisson_factor=poisson_factor, dim=self.dim, max_power=self.max_power)
    
    def split(self, keys, scheme, check=False, **kwargs):
        '''
        Split the polynomial with respect to a set of keys. 
        Return a list of polynomials according to the requested number of slices and the splitting.
        
        Parameters
        ----------
        scheme: list, optional
            A list of coefficients [a1, b1, a2, ...]
            
        Returns
        -------
        list
            A list F of poly objects so that if H = H1 + H2, where H is the current poly, H1 the poly
            according to the subset of keys and H2 the complement of H1, then:
            F = [a1*H1, b1*H2, a2*H1, b2*H2, ...]
            according to the requested input list above.
        '''
        ham1 = self.extract(key_cond=lambda x: x in keys)
        ham2 = self.extract(key_cond=lambda x: x not in keys)
        out = []
        c, d = [], []
        # the decomposition is (always) assumed to be alternating between two entries.
        for k in range(len(scheme)):
            f = scheme[k]
            if k%2 == 0:
                out.append(ham1*f)
                c.append(f)
            else:
                out.append(ham2*f)
                d.append(f)
                
        if check:
            # perform consistency check:
            check1, check2 = sum(c), sum(d)
            assert check1 == 1 and check2 == 1, f'Both sums need to be 1:\nsum1: {check1}, sum2: {check2}'
        return out
    
    def shift(self, shift, **kwargs):
        '''
        Shift the polynonmial by a given value, i.e. replace xi[k] by xi[k] + a[k] etc.
        
        This may be faster than using automatic differentiation at the point "a".
        
        Parameters
        ----------
        shift: subscriptable
            The vector by which the polynomial should be shifted.
            
        **kwargs
            Optional keyworded arguments passed to njet.poly._shift_component routine.
            
        Returns
        -------
        poly:
            A polynomial object corresponding to the shifted original polynomial.
        '''

        poly_terms = {k: v for k, v in self.items()}
        j = 0
        for shift_value in shift:
            poly_terms = _shift_component(shift_value, j, poly_terms, **kwargs)
            j += 1
        return self.__class__(values=poly_terms, dim=self.dim, max_power=self.max_power)
    
    
def construct(f, *lps, **kwargs):
    r'''
    Let z1, ..., zk be Lie polynomials and f an analytical function, taking k values.
    Depending on the input, this routine will either return the Lie polynomial :f(z1, ..., zk): or
    the map f(z1, ..., zk).
    
    Parameters
    ----------
    f: callable
        A function on which we want to apply the list of poly objects.
        It needs to be supported by the njet module.
        
    lps: poly
        The Lie polynomial(s) to be constructed.
        
    power: int, optional
        The maximal power of the resulting Lie polynomial (default: inf).
        If a value is provided, the routine will return a class of type poly, representing
        a Lie polynomial. If nothing is provided, the routine will return the function
        f(z1, ..., zk)
        
    max_power: int, optional
        See poly.__init__; only used if power < inf.
        
    point: list, optional
        Only relevant if power != inf. A point around f will be expanded. If nothing specified, 
        zeros will be used.
        
    Returns
    -------
    callable or poly
        As described above, depending on the 'power' input parameter, either the map f(z1, ..., zk) or
        the Lie polynomial :f(z1, ..., zk): is returned.
    '''
    n_args_f = len(lps)
    assert n_args_f > 0
    dim_poly = lps[0].dim
    
    assert n_args_f == f.__code__.co_argcount, 'Input function depends on a different number of arguments.'
    assert all([lp.dim == dim_poly for lp in lps]), 'Input polynomials not all having the same dimensions.'

    construction = lambda *z: f(*[lps[k](*z) for k in range(n_args_f)])   
    
    power = kwargs.get('power', float('inf'))
    if power == float('inf'):
        return construction
    else:
        point = kwargs.get('point', [0]*2*dim_poly)
        max_power = kwargs.get('max_power', min([l.max_power for l in lps]))
        dcomp = derive(construction, order=power, n_args=2*dim_poly)
        taylor_coeffs = dcomp(*point, mult_drv=False)
        return lps[0].__class__(values=taylor_coeffs, dim=dim_poly, max_power=max_power)
    
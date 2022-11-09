
# This file contains the basic functionalities of the poly class. Since the class will be derived later,
# it is not recommended to construct poly objects directly from that file.
import numpy as np

from njet import derive, jetpoly
from njet.common import check_zero, factorials

class _poly:
    '''
    Class to model the Lie operator :p:, where p is a polynomial given in terms of
    complex (xi, eta)-coordinates. For the notation of these coordinates see Ref. [1] p. 33 onwards.
    
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
        
    dim: int, optional
        The number of xi- (or eta-) variables. Will be determined automatically from the input, if nothing
        specified.
        
    **kwargs
        Optional arguments passed to self.set_monimial and self.set_max_power
        
    Reference(s):
        [1] "M. Titze: Space Charge Modeling at the Integer Resonance for the CERN PS and SPS" (2019).
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
    
    def extract(self, key_cond=lambda x: True, value_cond=lambda x: True):
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
        return self.__class__(values={key: value for key, value in self.items() if key_cond(key) and value_cond(value)}, 
                              dim=self.dim, max_power=self.max_power)
    
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
        return self.extract(value_cond=lambda x: abs(x) > tol)
    
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
        return self.extract(value_cond=lambda x: abs(x) < tol)
    
    def monomials(self):
        '''
        Return a list of the monomials of the current polynomial.
        '''
        return [self.__class__(values={k: v}, dim=self.dim, max_power=self.max_power) for k, v in self.items()]
        
    def __call__(self, *z):
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
        if len(z) == self.dim and self._poisson_factor == -1j:
            # for xi/eta-variables we conveniently conjugate the remaining components if they are not explicitly given
            z = [e for e in z] + [e.conjugate() for e in z]
        assert len(z) == 2*self.dim, f'Number of input parameters: {len(z)}, expected: {2*self.dim} (or {self.dim})'
        
        # compute the occuring powers ahead of evaluation
        z_powers = {}
        j = 0
        for we in zip(*self.keys()):
            z_powers[j] = {k: z[j]**int(k) for k in np.unique(we)} # need to convert k to int, 
            # otherwise we get a conversion to some numpy array if z is not a float (e.g. an njet).
            j += 1
        
        # evaluate polynomial at requested point
        result = 0
        for k, v in self.items():
            prod = 1
            for j in range(self.dim):
                prod *= z_powers[j][k[j]]*z_powers[j + self.dim][k[j + self.dim]]
            result += prod*v # v needs to stay on the right-hand side here, because prod may be a jet class (if we compute the derivative(s) of the Lie polynomial)
        return result
        
    def __add__(self, other):
        if other == 0:
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
        new_values = {k: abs(v) for k, v in self.items()}
        return self.__class__(values=new_values, dim=self.dim, max_power=self.max_power)
    
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
        return jetpoly(values=jpvalues)
    
    def apply(self, name, cargs={}, *args, **kwargs):
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
    
    def realBasis(self, **kwargs):
        '''
        Note that it holds xi = (q + 1j*p)/sqrt2, eta = (q - 1j*p)/sqrt2.
        '''
        sqrt2 = float(np.sqrt(2))
        def trf(q, p):
            xi = (q + p*1j)/sqrt2
            eta = (q - p*1j)/sqrt2 # = xi.conjugate()
            return xi, eta
        return self.transform(trf=trf, poisson_factor=1, **kwargs)
        
    def complexBasis(self, **kwargs):
        '''
        Note that it holds xi = (q + 1j*p)/sqrt2, eta = (q - 1j*p)/sqrt2.
        '''
        sqrt2 = float(np.sqrt(2))
        def trf(xi, eta):
            q = (xi + eta)/sqrt2
            p = (xi - eta)/sqrt2/1j
            return q, p
        return self.transform(trf=trf, poisson_factor=-1j, **kwargs)
        
    def transform(self, trf, poisson_factor, mult_drv=False, mult_prm=False, **kwargs):
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
            Control of factorial and permutation coefficients. See njet.poly.jetpoly.get_taylor_coefficients for details.
            
        mult_prm: boolean, optional
            Control of factorial and permutation coefficients. See njet.poly.jetpoly.get_taylor_coefficients for details.
            
        **kwargs
            Arguments passed to self.__class__.

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
            rbv = h1.get_taylor_coefficients(2*self.dim, facts=factorials(self.maxdeg()), 
                                          mult_drv=mult_drv, mult_prm=mult_prm)
        return self.__class__(values=rbv, poisson_factor=poisson_factor)
    
    def split(self, keys, scheme, check=False, **kwargs):
        '''
        Split the Hamiltonian with respect to a set of keys. 
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
    

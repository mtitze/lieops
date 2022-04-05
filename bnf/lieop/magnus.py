
from itertools import product
from scipy.special import bernoulli
from njet.jet import factorials

import numpy as np

'''
References:
[1]: S. P. Norsett, A. Iserles, H. Z. Munth-Kaas and A. Zanna: Lie Group Methods (2000).
[2]: T. Carlson: Magnus expansion as an approximation tool for ODEs (2005)
[3]: A. Iserles: Magnus expansions and beyond (2008)
'''

class hard_edge_model:
    
    '''
    Class to model hard-edge functions, given by piecewise polynomial functions, and their respective integrals.
    '''
    
    def __init__(self, positions, values):
        '''
        Parameters
        ----------
        positions: list
            A list of start and end positions of the hard edge values.
            
        values: list
            A list of values, where the values[k] denotes the value of the hard edge between position[k + 1] and position[k].
        '''
        assert len(values) + 1 == len(positions), f'len(values) = {len(values)}, len(positions) = {len(positions)}.'
        self.positions = positions # Attention: It is assumed that positions[i] < positions[j] hold.
        self.values = [[v] for v in values] # values[k] = [2, 0, -5] corresponds to f(x) = 2 + 0*x**1 - 5*x**2 later on etc.
        self.lengths = {(k, 1): self.positions[k + 1] - self.positions[k] for k in range(len(self.positions) - 1)} # self.lengths[(k, l)] = (s[k] - s[k - 1])**l
        
    def copy(self):
        result = self.__class__(positions=self.positions, values=range(len(self.positions) - 1))
        # overwrite/copy default
        result.values = [v for v in self.values]
        result.lengths = {k: v for k, v in self.lengths.items()}
        return result
    
    def __mul__(self, other):
        '''
        Multiply a given hard-edge function with another one.

        Attention: It is assumed that the positions of both hard-edge models agree.
        '''
        assert len(self.positions) == len(other.positions) # for the time being...
        # and all positions equal (no check at the moment)
        
        product = self.copy()
        for k in range(len(self.values)):
            vals1 = self.values[k]
            vals2 = other.values[k]
            
            max_order1, max_order2 = len(vals1), len(vals2)
            vals_mult = [0]*(max_order1 + max_order2)

            max_used = 1 # to drop unecessary zeros later on
            for order1 in range(len(vals1)):
                value1 = vals1[order1]
                if value1 == 0:
                    continue
                for order2 in range(len(vals2)):
                    value2 = vals2[order2]
                    if value2 == 0:
                        continue
                    vals_mult[order1 + order2] += value1*value2
                    max_used = max([max_used, order1 + order2])
                    
            product.values[k] = vals_mult[:max_used]
        return product
                
        
    def integral(self):
        '''
        Compute the integral
        
          x
         /
         | h(s) ds
         /
         p0
         
        where p0 = self.positions[0] and h(s) is the Hamiltonian of the hard-edge model.
        
        Returns
        -------
        integral: hard_edge
            A hard_edge object, representing the section-wise integrand of the current hard_edge object.
            
        float
            The result of the entire integration over the given range [self.positions[0], self.positions[-1]].
        '''
        result = self.copy()
        
        n_values = len(result.values)
        pos0 = result.positions[0]
        additional_summand = 0
        for i in range(n_values):
            n_coeffs = len(result.values[i])
            new_values_i = [additional_summand] + [result.values[i][k - 1]/k for k in range(1, n_coeffs + 1)]
            result.values[i] = new_values_i
            
            if (i, n_coeffs) not in result.lengths.keys():
                result.lengths[(i, n_coeffs)] = result.lengths[(i, 1)]**n_coeffs
            # n.B. len(new_values_i) == n_coeffs + 1
            additional_summand += sum([new_values_i[mu]*result.lengths[(i, mu)] for mu in range(1, n_coeffs + 1)])

        return result, additional_summand
    
    
class fourier_model:
    
    '''
    Model and compute a specific term in the integral of a tree, in case the original Hamiltonian has been decomposed into a Fourier series.
    '''
    
    def __init__(self, factors=[], exponent={}, sign=1):
        
        self.factors = factors
        self.exponent = exponent # the keys in 'exponent' denote the indices of the s-values, while the items denote the indices of the 'omega'-values. 
        self.sign = sign
        
    def __str__(self):
        if len(self.factors) == 0:
            return '1'
        
        if self.sign == -1:
            out = '-'
        else:
            out = ' '
            
        out += '[ '
        for flist in self.factors:
            out += '('
            for f in flist:
                out += f'f{f} + '
            out = out[:-3]
            out += ')*'
        out = out[:-1]
        expstr = ''
        
        for k, v in self.exponent.items():
            expstr += '('
            for e in v:
                expstr += f'f{e} + '
            expstr = expstr[:-3]
            expstr += ') + '
        expstr = expstr[:-3]
        
        out += f' ]**(-1) * [exp( {expstr} ) - 1]'
        return out
    
    def _repr_html_(self):
        return f'<samp>{self.__str__()}</samp>' 


class tree:
    '''
    A tree according to Refs. [1, 2, 3]
    '''
    
    def __init__(self, *branches, time_power=0):
        '''
        self.integration_scheme
        
        Example:
        Let the tree have the form
          [I([I(H), H]), [I([I(H), H]), H]]
        The function 'H' appears 5 times and each 'I' denotes a specific integration. 
        In addition to this, there is an integration over the entire expression (whose upper bound
        we shall denote by t_-1). 
        By bi-linearity of the Lie brackets, we can move out the interior integrals to obtain:
        
        I_0^{t_j4} [I_0^{t_j1} [I_0^{t_j0} H(t_0), H(t_1)], [I_0^{t_j3} [I_0^{t_j2} H(t_2), H(t_3)], H(t_4)]]
        
        The field self.integration scheme is a list L so that L[k] = jk. For this example:
        
        L = [1, 4, 3, 4, -1]
        
        So that the above integral is given by

        I_0^{t_-1} [I_0^{t_4} [I_0^{t_1} H(t_0), H(t_1)], [I_0^{t_4} [I_0^{t_3} H(t_2), H(t_3)], H(t_4)]]
        
        Parameters
        ----------
        *branches: None or (tree, tree)
            Either none or two trees which define the branches of the given tree.
        
        time_power: int, optional
            Only relevant at initialization if len(branches) == 0. 
            Defines the first order in time in the time-expansion of the operator (Hamiltonian).
        '''
        self._upper_bound_default = -1 # should be smaller than 0
        self.branches = branches
        if len(branches) == 0:
            self.index = 1 # we set the index of the fundamental object to one, to have the easiest way to increase the index for higher-orders.
            self.pivot_branches = [] # Contains the trees listen in e.g. Eq. (2.3), Ref. [3]
            self.factor = 1
            self.time_power = time_power # The power of the first coefficient in the Taylor expansion of the Hamiltonian with respect to time.
            self.integration_bounds = [self._upper_bound_default]
        else:
            assert len(branches) == 2
            self.index = branches[0].index + branches[1].index
            self.pivot_branches = [branches[0]] + branches[1].pivot_branches # Contains the trees listen in e.g. Eq. (2.3), Ref. [3]
            
            ### Keep track of the integration scheme ###
            # copy the integration schemes of the two branches
            bounds1 = [s for s in branches[0].integration_bounds]
            bounds2 = [s for s in branches[1].integration_bounds]
            
            # find their free variables, which are those unique indices with a '-1'.
            free_variable1 = bounds1.index(self._upper_bound_default)
            free_variable2 = bounds2.index(self._upper_bound_default)
            
            # relabel indices of scheme2 to fit in the larger setting
            bounds2 = [s + branches[0].index for s in bounds2]
            bounds2[free_variable2] = self._upper_bound_default # keep the free variable of scheme2
            
            # define integration over the free variable of scheme1 with upper bound of the free variable of scheme2
            bounds1[free_variable1] = free_variable2 + branches[0].index
            
            # put all together to define the integration scheme of the current tree
            self.integration_bounds = bounds1 + bounds2
            
    def _set_time_power(self, eb=None):
        '''
        Set the time power of the tree.
        
        Parameters
        ----------
        eb: boolean, optional
            self.branches[0] == self.branches[1]; if 'None' given, the check will be done.
        '''
        if eb == None:
            eb = self.branches[0] == self.branches[1]
            
        if eb:
            self.time_power = self.branches[0].time_power + self.branches[1].time_power + 2
        else:
            self.time_power = self.branches[0].time_power + self.branches[1].time_power + 1
            
    def integration_chain(self):
        '''
        Convert the integration bounds of the current tree into a multi-dimensional integral over a simplex. I.e. this routine
        will move the integrals in the example of self.__init__ in front of the entire expression.

        Returns
        -------
        list:
            A list of tuples, denoting the order and bounds of the multi-dimensional integral:
            [(j1, b1), (j2, b2), ..., (jk, bk)] corresponds to

            I_0^{b1} I_0^{b2} ... I_0^{bk} f(t_0, t_1, ..., t_k) dt_j1 ... dt_jk ,
            
            where f(t_0, t_1, ..., t_k) denotes the nested bracket expression of the tree (see self.__init__ for an example).
        '''
        # Input consistency check: In order to be able to move an integral into the front of the nested commutator expression, 
        # it is necessary that for every variable t_k, its corresponding upper bound b_k does not equal one of the
        # preceeding variables.
        assert all([all([j < self.integration_bounds[k] for j in range(k)]) for k in range(len(self.integration_bounds)) if self.integration_bounds[k] != self._upper_bound_default])
        
        # construct the ordering
        default_var = self.integration_bounds.index(self._upper_bound_default)
        level = {default_var: self._upper_bound_default}
        order = [(default_var, self._upper_bound_default)]
        integration_levels = [level]
        while len(level.keys()) > 0:
            level = {k: self.integration_bounds[k] for k in range(len(self.integration_bounds)) if self.integration_bounds[k] in level.keys()}
            order += [e[0] for e in zip(level.items())]
            integration_levels.append(level)
        return order, integration_levels[:-1]
    
    def hard_edge_integral(self, *args, **kwargs):
        '''
        Compute the nested chain of integrals in case the underlying Hamiltonian is given by a hard-edge model.
        
        Parameters
        ----------
        *args:
            Arguments passed to hard_edge_model.
            
        **kwargs:
            Keyworded arguments passed to hard_edge_model.
        '''
        hamiltonian = hard_edge_model(*args, **kwargs)        
        integrands = {k: hamiltonian for k in range(self.index)}
        ic, _ = self.integration_chain()
        for var, bound in ic[::-1]:
            integral_functions, I = integrands[var].integral()
            if bound == self._upper_bound_default:
                break
            integrands[bound] *= integral_functions
        return I
    
    def fourier_integral_terms(self, consistency_checks=False):
        '''
        If the original t-dependency reads exp(i sum_j (omega_j t_j)), for some variables omega_j, then
        a tree expression can be integrated immediately with respect to the t_j's. This routine will compute
        the resulting factor in front of the exponential, which will depend on the omega_j's. Their indices
        will be returned.
        
        Parameters
        ----------
        consistency_checks: boolean, optional
            If true, perform some consistency checks of the output.

        Returns
        -------
        list
            A list of fourier_model objects, having length 2**(self.index - 1). Each entry A hereby represents 
            one summand of the final integral.
        '''
        integral = [fourier_model(exponent={j: [j] for j in range(self.index)})] # the keys in 'exponent' denote the indices of the s-values, while the items denote the indices of the 'omega'-values.
        if self.index == 1:
            return integral
        
        chain, _ = self.integration_chain()
        for int_op in chain[::-1]:
            variable, bound = int_op
            
            # perform the integration for each summand represented by an entry in 'integral'
            new_integral = []
            for entry in integral:
                new_factors = [e for e in entry.factors] # copy required; otherwise entry['factors'] will be modified unintentionally
                coeffs = entry.exponent[variable]
                new_factors.append(coeffs)

                # lower bound (zero):
                exponent_lower = {k: [c for c in entry.exponent[k]] for k in entry.exponent.keys() if k != variable} # need to reset exponent[k] here as well, otherwise it will get modified unintentionally.
                # upper bound:
                exponent_upper = {a: [e for e in b] for a, b in exponent_lower.items()} # copy required; otherwise exponent_lower gets modified later
                if bound in exponent_upper.keys():
                    exponent_upper[bound] += coeffs
                else:
                    # the final integration
                    assert bound == self._upper_bound_default
                    exponent_upper[bound] = coeffs
                
                # add integration for upper and lower bound to the new set of integrals
                new_integral.append(fourier_model(factors=new_factors, exponent=exponent_upper, sign=entry.sign))
                new_integral.append(fourier_model(factors=new_factors, exponent=exponent_lower, sign=-1*entry.sign))

            integral = new_integral
            
        non_zero_terms = [integral[2*k] for k in range(len(integral)//2)] # by construction, the non-zero exponents (upper bounds) are computed first
        if consistency_checks:
            # consistency checks
            assert len(integral) == 2**self.index # each integration step yields two terms, one for the upper and one for the lower bound.
            zero_terms = [integral[2*k + 1] for k in range(len(integral)//2)] # in the final output, the lower bounds have zero in their exponents.
            assert all([len(e.exponent) == 1 for e in non_zero_terms]) # the non-zero terms must have only self._upper_bound_default as single key.
            assert all([len(e.exponent) == 0 for e in zero_terms]) # verify that the zero_terms are in fact zero in their exponents.
            assert all([e.exponent[self._upper_bound_default] == e.factors[-1] for e in non_zero_terms]) # the last factors must equal the final exponents.
            
        # prepare output; only factors are required for the full information
        return non_zero_terms
            
            
    def set_factor(self, b=[]):
        '''
        Set the factor associated with the coefficient of the tree.
        The factor will be given in self.factor.
        
        Attention: Calling this method requires that all the factors in self.pivot_branches have already been determined. There will be no check.
        
        Parameters
        ----------
        b: list, optional
            An optional argument so that the n-th element equals B_n/n!. If such a list is given,
            it must hold len(b) >= len(self.pivot_branches)
        '''
        n = len(self.pivot_branches)
        if len(b) == 0:
            f0 = bernoulli(n)[-1]/factorials(n)[-1]
        else:
            f0 = b[n]
            
        self.factor = f0
        for pbranch in self.pivot_branches:
            self.factor *= pbranch.factor
            
    def __eq__(self, other):
        if self.index != other.index:
            return False
        elif self.index == 1: # == other.index
            return True
        else:
            return self.branches[1] == other.branches[1] and self.branches[0] == other.branches[0]
        
    def __str__(self):
        if len(self.branches) == 0:
            return 'H'
        else:
            return f'[I({str(self.branches[0])}), {self.branches[1]}]'
    
    def _repr_html_(self):
        return f'<samp>{self.__str__()}</samp>' 
    
    
def forests(k, time_power=0):
    '''
    Construct a set of trees with respect to a given index, according to Refs. [1, 2, 3].
    
    Parameters
    ----------
    k: int
        The maximal index up to which we want to construct the trees.
        
    time_power: int, optional
        Optional initial order of time, of the first Taylor coefficient of the given operator.
        
    Returns
    -------
    dict
        A dictionary where the key j corresponds to a list containing all trees with index j.
        
    dict
        A dictionary representing the forests of trees for the requested time powers 
        (up to k*(time_power + 1) + 2, see the discussion in this code below).
    '''
    factors = bernoulli(k)/factorials(k)
    tree_groups = {0: [tree(time_power=time_power)]} # Representing the sets T_k in Refs. [1, 2, 3].
    forest_groups = [tree_groups[0]] # Representing the sets F_k in Refs. [1, 2, 3].
    for j in range(1, k + 1):
        # construct the set of trees with respect to index j from trees of indices q < j and p < j so that q + p = j
        treesj = []
        for q in range((j - 1)//2 + 1): # We don't have to iterate from 0 up to j - 1, but only half the way: The case j - q - 1 < q is already covered by some q later on (elements of trees_q and trees_p are exchanged and added, if they provide a new unique tree).
            p = j - q - 1
            # N.B. 
            # 1) q <= p
            # 2) Unfortunately there is no immediate relation to the building of trees and the factors: Even if the factor
            # of a specific tree is zero, it may happen that this tree is used later on in a tree with a higher index whose factor
            # is not zero. So for building the forest we need to take all trees into account even if their factors may be zero.
            # For example: Tree nr. 7 in Ref. [2] is zero due to B_3 = 0, but tree nr. 17 is not zero, but build from the tree nr. 7.
            for t1, t2 in product(tree_groups[q], tree_groups[p]):
                trees_equal = t1 == t2 # n.B. if q != p, then this will not go deep into the trees but return False immediately.
                
                t12 = tree(t1, t2)
                # t12.index == j + 1 with p + q == j - 1, so that t12.index == p + q + 2 == t1.index + t2.index
                t12.set_factor(factors)
                t12._set_time_power(trees_equal)
                treesj.append(t12)
                    
                if not trees_equal:
                    t21 = tree(t2, t1)
                    t21.set_factor(factors)
                    t21._set_time_power(trees_equal)
                    treesj.append(t21)

        tree_groups[j] = treesj
            
    time_powers = np.unique([t.time_power for tg in tree_groups.values() for t in tg])
    max_power = k*(time_power + 1) + 2 # Trees of index k can only contribute to forests of k + 2 at most (if multiplied by a tree with index 1).
    # Each additional time_power can be extracted out of the brackets and therefore acts as a flat addition to this value. So we have max_power = k + 2 + k*time_power = k*(1 + time_power) + 2. We do not include forests beyond this value.
    forest_groups = {power: [t for tg in tree_groups.values() for t in tg if t.time_power == power] for power in time_powers if power <= max_power}
    # N.B. in Ref. [2] it appears that the term belonging to "F_5" with coeff -1/24 is not correctly assigned. It should be in F_6. This code also predicts that it is in F_6. To be checked: In Ref. [1] #T_6 = 132 and #F_6 = 21, while here #T_6 = 136 and #F_6 = 21. Given the fact that there is agreement with #F_6, I am not sure whether there was a miscalculation of #T_6 in Ref. [1].
        
    return tree_groups, forest_groups



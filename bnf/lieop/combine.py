
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
    
    def exp_integral_factors(self, add_last_chain=False):
        '''
        If the original t-dependency reads exp(i sum_j (omega_j t_j)), for some variables omega_j, then
        a tree expression can be integrated immediately with respect to the t_j's. This routine will compute
        the resulting factor in front of the exponential, which will depend on the omega_j's. Their indices
        will be returned.

        Returns
        -------
        list
            A list of n-tuples, each n-tuple (j1, j2, ..., jk) representing a factor of the form
            1/(omega_{j1} + omega_{j2} + ... + omega_{jk}) 
            in the overall integrand.
            
        add_last_chain: bool, optional
            If set to true, then add the last chain to the result, which corresponds to a summation over all available indices.
        '''
        if self.index == 1:
            return []
        _, levels = self.integration_chain()
        chains = {(k,): v for k, v in levels[-1].items()}
        n_levels = len(levels)
        for level in levels[:0:-1]:
            chains.update({k1 + (chains[k1],): level[chains[k1]] for k1 in chains if chains[k1] in level})
            chains.update({(k2,): level[k2] for k2 in level if k2 not in chains.values()})
        if add_last_chain:
            # add the last chain, which is the sum over all indices
            chains[tuple(range(self.index))] = self._upper_bound_default # the value doesn't matter here
        return list(chains.keys())
            
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
        else:
            if self.index == 1: # == other.index
                return True
            else:
                return self.branches[0] == other.branches[0] and self.branches[1] == other.branches[1]
        
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
        # construct the set of Trees with respect to index j from trees of indices q < j and p < j so that q + p = j
        treesj = []
        for q in range((j - 1)//2 + 1): # the case j - q - 1 < q is already covered by q later on (elements of trees_q and trees_p are exchanged and added, if the provide a new unique tree).
            p = j - q - 1
            # N.B. q <= p
            for t1, t2 in product(tree_groups[q], tree_groups[p]):
                t12 = tree(t1, t2)
                t12.set_factor(factors)
                if t1 != t2:
                    t21 = tree(t2, t1)
                    t21.set_factor(factors)
                    time_power_12 = t1.time_power + t2.time_power + 1 # keep track of the power in time, see e.g. Ref. [2], Thm. 19
                    t21.time_power = time_power_12
                    t12.time_power = time_power_12
                    treesj.append(t21)
                else:
                    t12.time_power = t1.time_power + t2.time_power + 2 # keep track of the power in time, see e.g. Ref. [2], Thm. 19
                treesj.append(t12)
        tree_groups[j] = treesj
            
    time_powers = np.unique([t.time_power for tg in tree_groups.values() for t in tg])
    max_power = k*(time_power + 1) + 2 # Trees of index k can only contribute to forests of k + 2 at most (if multiplied by a tree with index 1).
    # Each additional time_power can be extracted out of the brackets and therefore acts as a flat addition to this value. So we have max_power = k + 2 + k*time_power = k*(1 + time_power) + 2. We do not include forests beyond this value.
    forest_groups = {power: [t for tg in tree_groups.values() for t in tg if t.time_power == power] for power in time_powers if power <= max_power}
    # N.B. in Ref. [2] it appears that the term belonging to "F_5" with coeff -1/24 is not correctly assigned. It should be in F_6. This code also predicts that it is in F_6. To be checked: In Ref. [1] #T_6 = 132 and #F_6 = 21, while here #T_6 = 136 and #F_6 = 21. Given the fact that there is agreement with #F_6, I am not sure whether there was a miscalculation of #T_6 in Ref. [1].
        
    return tree_groups, forest_groups


            
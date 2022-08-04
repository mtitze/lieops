

class symint:
    
    def __init__(self, scheme=[1, 1/2]):
        '''
        Model a symplectic integrator which is symmetric according to Yoshida [1].
        
        References
        ----------
        [1] H. Yoshida: "Construction of higher order symplectic integrators", 
        Phys. Lett. A 12, volume 150, number 5,6,7 (1990).
        '''
        self.scheme = scheme
        
    @staticmethod
    def branch_factors(m: int, scheme=[1, 1/2]):
        if m == 0:
            return scheme
        else:
            z0 = -2**(1/(m + 1))/(2 - 2**(1/(m + 1)))
            z1 = 1/(2 - 2**(1/(m + 1)))
            return z0, z1
        
    def build(self, n: int):
        '''
        Construct the coefficients for the symmetric Yoshida integrator.
        
        Parameters
        ----------
        n: int
            The order of the integrator.
        '''
        z0, z1 = self.branch_factors(m=n, scheme=self.scheme)
        steps_k = [z1, z0, z1]
        for k in range(n):
            new_steps = []
            for step in steps_k:
                z0, z1 = self.branch_factors(m=n - k - 1, scheme=self.scheme)
                new_steps += [z1*step, z0*step, z1*step]
            steps_k = new_steps
            
        # In its final step, steps_k has the form
        # [a1, b1, a1, a2, b2, a2, a3, b3, a3, ..., bm, am]
        # where the aj's belong to the first operator and the bj's belong to the second operator.
        # Therefore, we have to add the inner aj's together. They belong to the index pairs
        # (2, 3), (5, 6), (8, 9), (11, 12), ...
        pair_start_indices = [j for j in range(2, len(steps_k) - 3, 3)]
        out = []
        k = 0
        while k < len(steps_k):
            if k in pair_start_indices:
                out.append(steps_k[k] + steps_k[k + 1])
                k += 2
            else:
                out.append(steps_k[k])
                k += 1
        return out

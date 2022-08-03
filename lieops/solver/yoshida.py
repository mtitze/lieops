

class symint:
    
    def __init__(self, scheme=[1, 1/2]):
        '''
        A symplectic integrator which is symmetric.
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
        
    def build(self, n: int, step=1):
        z0, z1 = self.branch_factors(m=n, scheme=self.scheme)
        steps_k = [step*z1, step*z0, step*z1]
        for k in range(n):
            new_steps = []
            for step in steps_k:
                z0, z1 = self.branch_factors(m=n - k - 1, scheme=self.scheme)
                new_steps += [z1*step, z0*step, z1*step]
            steps_k = new_steps
        return steps_k

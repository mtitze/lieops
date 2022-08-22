import numpy as np
from njet import derive

from njet.functions import sin, cos

from .common import getRealHamiltonFunction

class integrator:
    
    def __init__(self, hamiltonian, order=2, omega=1, delta=1, real=False, **kwargs):
        '''
        Class to manage Tao's l'th order symplectic integrator for non-separable Hamiltonians.

        Reference(s):
        [1]: Molei Tao: "Explicit symplectic approximation of nonseparable Hamiltonians: 
                         algorithm and long time performance", PhysRevE.94.043303 (2016).
        '''
        self.dim = hamiltonian.dim
        if not real:
            hamiltonian = hamiltonian*-1j
            self.dhamiltonian = derive(hamiltonian, order=1, n_args=self.dim*2)
        else:
            self.realHamiltonian = getRealHamiltonFunction(hamiltonian, **kwargs)
            self.dhamiltonian = derive(self.realHamiltonian, order=1, n_args=self.dim*2)
        self.hamiltonian = hamiltonian

        self.delta = delta # the step size
        self.omega = omega # the coupling between the two phase spaces
        self.set_order(order) # the order of the integrator (must be even)
        
        # njet.poly keys required in dhamiltonian to obtain the gradient of the Hamiltonian for each Q and P index (starting from 0 to self.dim)
        self._component1_keys = {w: tuple(0 if k != w else 1 for k in range(2*self.dim)) for w in range(self.dim)}
        self._component2_keys = {w: tuple(0 if k != w + self.dim else 1 for k in range(2*self.dim)) for w in range(self.dim)}
            
    def set_order(self, order):
        '''
        Compute the scheme of Tao's integrator for the requested order.
        '''
        assert order%2 == 0, 'Order has to be even.'
        self.order = order
        scheme = {2: [self.delta]}
        for l in range(4, order + 1, 2):
            gamma_l = 1/(2 - 2**(1/(l + 1)))
            f1 = [delta*gamma_l for delta in scheme[l - 2]]
            f2 = [delta*(1 - 2*gamma_l) for delta in scheme[l - 2]]
            scheme[l] = f1 + f2 + f1
        self.scheme = scheme[order]
        self.cos_sin = [[np.cos(2*self.omega*delta), np.sin(2*self.omega*delta)] for delta in self.scheme]
                        
    def second_order_map(self, *qp, delta, w):
        q, p = list(qp[:self.dim]), list(qp[self.dim:])
        z0 = [q, p, q, p]
        z1 = self.phi_HA(*z0, delta=delta/2)
        z2 = self.phi_HB(*z1, delta=delta/2)
        z3 = self.phi_HC(*z2, w)
        z4 = self.phi_HB(*z3, delta=delta/2)
        z5 = self.phi_HA(*z4, delta=delta/2)
        out = []
        for z in z5[:2*self.dim]: # only return the first 2 coordinates, which are q & p
            out += z # the elements z are lists, we concatenate them together to have consistent output
        return out 
        
    def phi_HA(self, q, p, x, y, delta):
        dham = self.dhamiltonian(*(q + y))
        result2, result3 = [], []
        for k in range(self.dim):
            result2.append(p[k] - dham.get(self._component1_keys[k], 0)*delta)
            result3.append(x[k] + dham.get(self._component2_keys[k], 0)*delta)
        return q, result2, result3, y
    
    def phi_HB(self, q, p, x, y, delta):
        dham = self.dhamiltonian(*(x + p))
        result1, result4 = [], []
        for k in range(self.dim):
            result1.append(q[k] + dham.get(self._component2_keys[k], 0)*delta)
            result4.append(y[k] - dham.get(self._component1_keys[k], 0)*delta)
        return result1, p, x, result4
    
    def phi_HC(self, q, p, x, y, w):
        cos, sin = self.cos_sin[w]
        result1, result2, result3, result4 = [], [], [], []
        for k in range(self.dim):
            diff1 = q[k] - x[k]
            diff2 = p[k] - y[k]
            r1 = diff1*cos + diff2*sin
            r2 = diff1*-sin + diff2*cos
            sum1 = q[k] + x[k]
            sum2 = p[k] + y[k]
            result1.append((sum1 + r1)*0.5)
            result2.append((sum2 + r2)*0.5)
            result3.append((sum1 - r1)*0.5)
            result4.append((sum2 - r2)*0.5)
        return result1, result2, result3, result4
    
    def __call__(self, *xieta0):
        n = len(self.scheme)
        xieta = xieta0
        # TODO: combine adjacent phi_HA-maps
        for w in range(n):
            delta = self.scheme[n - 1 - w]
            xieta = self.second_order_map(*xieta, delta=delta, w=w)
        return xieta
        
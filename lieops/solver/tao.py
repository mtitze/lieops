import numpy as np
from njet import derive

from njet.functions import sin, cos

from .common import getRealHamiltonFunction

class integrator:
    
    def __init__(self, hamiltonian, order=2, omega=1, delta=1, **kwargs):
        '''
        Class to manage Tao's l'th order symplectic integrator for non-separable Hamiltonians.

        Reference(s):
        [1]: Molei Tao: "Explicit symplectic approximation of nonseparable Hamiltonians: 
                         algorithm and long time performance", PhysRevE.94.043303 (2016).
        '''
        
        self.hamiltonian = hamiltonian
        self.dim = self.hamiltonian.dim
        
        self.realHamiltonian = getRealHamiltonFunction(hamiltonian, **kwargs)
        self.dhamiltonian = derive(self.realHamiltonian, order=1, n_args=self.dim*2)
        
        self.omega = omega
        self.delta = delta
        self.set_order(order=order)
        
        # njet.poly keys required in dhamiltonian to obtain the gradient of the Hamiltonian for each Q and P index (starting from 0 to self.dim)
        self._component1_keys = {w: tuple(0 if k != w else 1 for k in range(2*self.dim)) for w in range(self.dim)}
        self._component2_keys = {w: tuple(0 if k != w + self.dim else 1 for k in range(2*self.dim)) for w in range(self.dim)}

        
    def set_order(self, order):
        self.order = order
        
        delta = self.delta ### TODO
        
        # TODO
        # the cos and sind terms are computed only once, here, but their values may depend on the order (delta)
        argument = 2*self.omega*delta
        
        # scheme 2
        self.cos, self.sin = {}, {}
        for k in [1, 2]:
            self.cos[k] = np.cos(argument/k)
            self.sin[k] = np.sin(argument/k)
            #self.cos[k] = cos(argument/k)
            #self.sin[k] = sin(argument/k)
        
    def second_order(self, *qp):
        q, p = qp[:self.dim], qp[self.dim:]
        z0 = [q, p, q, p]
        z1 = self.phi_HA(*z0, delta=self.delta/2)
        z2 = self.phi_HB(*z1, delta=self.delta/2)
        z3 = self.phi_HC(*z2, w=1)
        z4 = self.phi_HB(*z3, delta=self.delta/2)
        z5 = self.phi_HA(*z4, delta=self.delta/2)
        return z5[:2*self.dim] # only return the first 2 coordinates, which are q & p
        
    def phi_HA(self, q, p, x, y, delta=1):
        dham = self.dhamiltonian(*(q + y))
        result2, result3 = [], []
        for k in range(self.dim):
            result2.append(p[k] - dham.get(self._component1_keys[k], 0)*delta)
            result3.append(x[k] + dham.get(self._component2_keys[k], 0)*delta)
        return q, result2, result3, y
    
    def phi_HB(self, q, p, x, y, delta=1):
        dham = self.dhamiltonian(*(x + p))
        result1, result4 = [], []
        for k in range(self.dim):
            result1.append(q[k] + dham.get(self._component2_keys[k], 0)*delta)
            result4.append(y[k] - dham.get(self._component1_keys[k], 0)*delta)
        return result1, p, x, result4
    
    def phi_HC(self, q, p, x, y, w=1):
        result1, result2, result3, result4 = [], [], [], []
        for k in range(self.dim):
            diff1 = q[k] - x[k]
            diff2 = p[k] - y[k]
            r1 = diff1*self.cos[w] + diff2*self.sin[w]
            r2 = diff1*-self.sin[w] + diff2*self.cos[w]
            sum1 = q[k] + x[k]
            sum2 = p[k] + y[k]
            result1.append((sum1 + r1)*0.5)
            result2.append((sum2 + r2)*0.5)
            result3.append((sum1 - r1)*0.5)
            result4.append((sum2 - r2)*0.5)
        return result1, result2, result3, result4
        
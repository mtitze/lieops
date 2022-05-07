import numpy as np

class integrator:
    
    def __init__(self, hamiltonian, order=2, omega=1, testdelta=1):
        '''
        Class to manage Tao's l'th order symplectic integrator for non-separable Hamiltonians.
        
        In our setting of (xi, eta)-canonical variables, the symplectic structure is altered to -i*J.
        We therefore have replaced H by -i*H in Tao's paper, so that we can use (xi, eta) instead of his (q, p).
        
        Reference(s):
        [1]: Molei Tao: "Explicit symplectic approximation of nonseparable Hamiltonians: 
                         algorithm and long time performance", PhysRevE.94.043303 (2016).
        '''
        
        self.hamiltonian = hamiltonian
        self.dim = hamiltonian.dim
        self.dhamiltonian = hamiltonian.derive(order=1) # we just need the first-order derivatives (the gradient).
        self.omega = omega
        
        self.testdelta = testdelta
        self.set_order(order=order)
        
        # njet.poly keys required in dhamiltonian to obtain the gradient of the Hamiltonian for each Q and P index (starting from 0 to self.dim)
        self._component1_keys = {w: tuple(0 if k != w else 1 for k in range(2*self.dim)) for w in range(self.dim)}
        self._component2_keys = {w: tuple(0 if k != w + self.dim else 1 for k in range(2*self.dim)) for w in range(self.dim)}

        
    def set_order(self, order):
        self.order = order
        
        delta = self.testdelta ### TODO
        
        # TODO
        # the cos and sind terms are computed only once, here, but their values may depend on the order (delta)
        self.cos = {delta: np.cos(2*self.omega*delta)} # todo: 
        self.sin = {delta: np.sin(2*self.omega*delta)}
        
        self.cos[delta/2] = np.cos(2*self.omega*delta/2)
        self.sin[delta/2] = np.sin(2*self.omega*delta/2)
        
    def phi_HA(self, xi, eta, x, y, delta=1):
        dham = self.dhamiltonian(xi + y)
        result2, result3 = [], []
        for k in range(self.dim):
            result2.append(eta[k] + dham.get(self._component1_keys[k], 0)*delta*1j)
            result3.append(x[k] - dham.get(self._component2_keys[k], 0)*delta*1j)
        return xi, result2, result3, y
    
    def phi_HB(self, xi, eta, x, y, delta=1):
        dham = self.dhamiltonian(x + eta)
        result1, result4 = [], []
        for k in range(self.dim):
            result1.append(xi[k] - dham.get(self._component2_keys[k], 0)*delta*1j)
            result4.append(y[k] + dham.get(self._component1_keys[k], 0)*delta*1j)
        return result1, eta, x, result4
    
    def phi_HC(self, xi, eta, x, y, delta=1):
        result1, result2, result3, result4 = [], [], [], []
        for k in range(self.dim):
            diff1 = xi[k] - x[k]
            diff2 = eta[k] - y[k]
            r1 = diff1*self.cos[delta] + diff2*self.sin[delta]
            r2 = diff1*-self.sin[delta] + diff2*self.cos[delta]
            sum1 = xi[k] + x[k]
            sum2 = eta[k] + y[k]
            result1.append((sum1 + r1)*0.5)
            result2.append((sum2 + r2)*0.5)
            result3.append((sum1 - r1)*0.5)
            result4.append((sum2 - r2)*0.5)
        return result1, result2, result3, result4
        
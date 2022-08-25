import numpy as np
from njet import derive
import warnings
import itertools

from .common import getRealHamiltonFunction

class integrator:
    
    def __init__(self, hamiltonian, order: int=2, t=1, omega=0.5, real=False, **kwargs):
        '''
        Class to manage Tao's l'th order symplectic integrator for non-separable Hamiltonians.
        
        Parameters
        ----------
        hamiltonian: poly
            A poly class representing the Hamiltonian to be integrated.
        
        order: int, optional
            The order of the integrator.
            
        f: float, optional
            A factor f so that "delta << omega**(-1/order), then "f*delta == omega**(-1/order)". 
            This factor is only used in automatically selecting suitable omega and delta-values 
            for a given order. Any value given to omega or delta will ignore/overwrite the values
            suggested by f.
        
        omega: float, optional
            The coupling constant between the two phase spaces.
            
        delta: float, optional
            The step size of the integrator.
            
        real: boolean, optional
            Whether or not the Hamiltonian should be treated as dependent on the (real) q/p coordinates
            or (complex) xi/eta coordinates.

        Reference(s):
        [1]: Molei Tao: "Explicit symplectic approximation of nonseparable Hamiltonians: 
                         algorithm and long time performance", PhysRevE.94.043303 (2016).
        '''
        self.set_hamiltonian(hamiltonian, real=real, **kwargs)

        f = kwargs.get('f', 100) # An optional factor f so that "delta << omega**(-1/order), then "f*delta == omega**(-1/order)"
        self.omega = omega # the coupling between the two phase spaces
        self.delta = kwargs.get('delta', self.omega**(-1/order)/f) # the underlying step size; the default value here comes from the fact that delta << omega**(-1/order) should hold (see Ref. [1]). We consider a factor of 100 for "<<" to hold.
        self.set_order(order) # the order of the integrator (must be even)
        self.set_time(t=t)
    
        self.make_error_estimations()
        
        # njet.poly keys required in dhamiltonian to obtain the gradient of the Hamiltonian for each Q and P index (starting from 0 to self.dim)
        self._component1_keys = {w: tuple(0 if k != w else 1 for k in range(2*self.dim)) for w in range(self.dim)}
        self._component2_keys = {w: tuple(0 if k != w + self.dim else 1 for k in range(2*self.dim)) for w in range(self.dim)}
                
    def set_hamiltonian(self, hamiltonian, real=False, **kwargs):
        self.dim = hamiltonian.dim
        if not real:
            hamiltonian = hamiltonian*-1j
            self.dhamiltonian = derive(hamiltonian, order=1, n_args=self.dim*2)
        else:
            self.realHamiltonian = getRealHamiltonFunction(hamiltonian, **kwargs)
            self.dhamiltonian = derive(self.realHamiltonian, order=1, n_args=self.dim*2)
        self.hamiltonian = hamiltonian
        
    def set_time(self, t):
        self.t = t
        n_reps, r = divmod(self.t, self.delta)
        self.n_reps = int(n_reps)
        self.n_reps_r = r
        
    def set_n_reps(self, n_reps):
        self.t = n_reps*self.delta
        self.n_reps = n_reps
        self.n_reps_r = 0
            
    def set_order(self, order):
        '''
        Compute the scheme of Tao's integrator for the requested order, using a 'triple jump' scheme.
        '''
        # TODO: may use other (improved) schemes...
        assert order%2 == 0, 'Order has to be even.'
        self.order = order
        scheme = {2: [1]}
        for l in range(4, order + 1, 2):
            gamma_l = 1/(2 - 2**(1/(l + 1)))
            f1 = [f*gamma_l for f in scheme[l - 2]]
            f2 = [f*(1 - 2*gamma_l) for f in scheme[l - 2]]
            scheme[l] = f1 + f2 + f1
        self.scheme = scheme[order]
        self.cos_sin = [[np.cos(2*self.omega*f*self.delta), np.sin(2*self.omega*f*self.delta)] for f in self.scheme]

    def make_error_estimations(self, show=False, warn=True):
        '''
        Perform some checks to help deciding whether omega, delta and the order
        have been properly chosen for the given problem.
        
        After executing this routine, the error estimations are found in self.error_estimations.
        
        Parameters
        ----------
        show: boolean, optional
            Print the error estimates.
            
        warn: boolean, optional
            Warn in case that delta > omega**(-1/order) holds.
        '''
        l = self.order
        error_estimations = {}
        error_estimations['order'] = self.order
        error_estimations['delta'] = self.delta
        error_estimations['omega'] = self.omega
        error_estimations['error'] = self.delta**l*self.omega # the global error of the solution towards the exact result (for integrable systems)
        error_estimations['qx_py'] = 1/np.sqrt(self.omega) # the error q - x, resp. p - y.
        error_estimations['tmax_o'] = min([self.delta**(-l)/self.omega, np.sqrt(self.omega)])
        error_estimations['delta_vs_omega'] = [self.delta, self.omega**(-1/l)]
        if self.delta*10 > error_estimations['delta_vs_omega'][1] and warn:
            warnings.warn(f"It appears that {self.delta} = delta << omega**(-1/order) = {error_estimations['delta_vs_omega'][1]} is not satisfied.")
            
        self.error_estimations = error_estimations
        if show:
            print ('Integrator')
            print ('----------')
            print (f' order: {l}')
            print (f' omega: {self.omega}')
            print (f' delta: {self.delta}')
            print (f'  time: {self.t}')
            print (f'n_reps: {self.n_reps}')
            print ('\nError estimates')
            print ('---------------')
            print (f"    Against exact solution: O({error_estimations['error']}*t) = O({error_estimations['error']*self.t})")
            print (f"                t/t_max_o = {self.t/error_estimations['tmax_o']}")
            print (f"delta << omega**(-1/order): {error_estimations['delta_vs_omega']} (?)")
            print (f"         |q - x|, |p - y| ~ O({error_estimations['qx_py']})")
            
    def second_order_map(self, *qp, delta, cs):
        q, p = list(qp[:self.dim]), list(qp[self.dim:])
        z0 = [q, p, q, p]
        z1 = self.phi_HA(*z0, delta=delta/2)
        z2 = self.phi_HB(*z1, delta=delta/2)
        z3 = self.phi_HC(*z2, cs)
        z4 = self.phi_HB(*z3, delta=delta/2)
        z5 = self.phi_HA(*z4, delta=delta/2)
        return list(itertools.chain.from_iterable(z5[:2*self.dim])) # the elements of z5 are lists, we concatenate them together to have consistent output; see https://stackoverflow.com/questions/716477/join-list-of-lists-in-python
        
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
    
    def phi_HC(self, q, p, x, y, cs):
        cos, sin = cs
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
    
    def step(self, *xieta0, **kwargs):
        '''
        Integrate the equations of motion using a single delta-step of the integrator.
        '''
        if 'delta' in kwargs.keys():
            delta = kwargs['delta']
            cs = [[np.cos(2*self.omega*f*delta), np.sin(2*self.omega*f*delta)] for f in self.scheme]
        else:
            delta = self.delta
            cs = self.cos_sin

        cs = kwargs.get('cs', self.cos_sin)
        xieta = xieta0
        # TODO: combine adjacent phi_HA-maps
        n = len(self.scheme)
        for w in range(n):
            xieta = self.second_order_map(*xieta, delta=self.scheme[n - 1 - w]*delta, cs=cs[n - 1 - w])
        return xieta
    
    def __call__(self, *xieta0, trajectory=False, **kwargs):
        xieta_all = []
        xieta = xieta0
        for k in range(self.n_reps):
            xieta = self.step(*xieta)
            xieta_all.append(xieta)
        if self.n_reps_r > 0:
            xieta = self.step(*xieta, delta=self.n_reps_r)
            xieta_all.append(xieta)
            
        if trajectory:
            return xieta_all
        else:
            return xieta
        
        
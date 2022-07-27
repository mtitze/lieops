import heyoka as hy
import numpy as np

from .common import realHamiltonEqs

class heyoka_solver:

    def __init__(self, hamiltonian, **kwargs):
        self.hamiltonian = hamiltonian
        self.dim = self.hamiltonian.dim
        self.variables, self.hameqs, self.realHamiltonian = self.getHamiltonEqs(**kwargs)
        self.integrator = hy.taylor_adaptive(self.hameqs, [0, 0])
        self.t = kwargs.get('t', 1)
        
    def getHamiltonEqs(self, **kwargs):
        '''
        Compute the Hamilton equations from the given Hamiltonian (self.hamiltonian).
        

        Returns
        -------
        dict
            A dictionary containing the real Hamilton equations and the integration steps.
        '''
        qp = hy.make_vars(*([f"coord_q{k}" for k in range(self.dim)] + 
                            [f"coord_p{k}" for k in range(self.dim)]))
        kwargs['real'] = True # for the solver it is necessary to use a real-valued Hamiltonian
        hameqs, rham = realHamiltonEqs(self.hamiltonian, **kwargs)
        hameqs_hy = hameqs(*qp) # hameqs represents the Hamilton-equations for the real variables q and p.
        return qp, [e for e in zip(*[qp, hameqs_hy])], rham
        
    def ensemble_propagation(self, q0, p0, kind='until', **kwargs):
        '''
        Ensemble propagation according to https://bluescarni.github.io/heyoka/tut_ensemble.html
        
        Parameters
        ----------
        q0: array-like of shape (K, dim) 
            Input vector(s) denoting the coordinates. The number 'K' denotes the number of different
            vectors to be tracked, while 'dim' must correspond to the dimension of the current hamiltonian.
            
        p0: array-like of shape (K, dim)
            Input vector(s) denoting the momenta.
            
        kind: str, optional
            A string denoting the type of integration, calling the three possible integration routines
            in Heyoka (see https://bluescarni.github.io/heyoka/tut_ensemble.html)
            'until': track from 0 until t
            'for': track from t0 until t0 + t
            'grid': track a grid.
            
        **kwargs
            Optional keyworded arguments passed to the heyoka ensemble_propagate_* routines.
            
        Returns
        -------
        ret
            Object according to https://bluescarni.github.io/heyoka/tut_ensemble.html
        '''
        
        assert hasattr(q0, 'shape') and hasattr(p0, 'shape')
        assert q0.shape == p0.shape, 'Input vector shape mismatch.'
        assert len(q0.shape) == 2, 'Input vector shape length 2 required.'
        n_iter, dim = q0.shape
        assert dim == self.dim, 'Input vector dimension not consistent with dimension of given Hamiltonian.'
        
        kwargs['t'] = kwargs.get('t', self.t)
        
        # create a generator, taking a taylor adaptive object and modifying the state according to the
        # vector components
        def gen(tacp, i):
            tacp.state[:self.dim] = q0[i, :]
            tacp.state[self.dim:] = p0[i, :]
            return tacp

        if kind == 'until':
            return hy.ensemble_propagate_until(self.integrator, n_iter=n_iter, gen=gen, **kwargs)
        elif kind == 'for':
            return hy.ensemble_propagate_for(self.integrator, n_iter=n_iter, gen=gen, **kwargs) # todo
        elif kind == 'grid':
            return hy.ensemble_propagate_grid(self.integrator, n_iter=n_iter, gen=gen, **kwargs) # todo
        else:
            raise RuntimeError(f"Requested kind '{kind}' not recognized.")
            
    def _homgenInput(self, *qp0):
        '''
        Helper class to deal with user-defined input.
        '''
        assert len(qp0) == 2*self.dim, f'Length of input {len(qp0)}, expected: {2*self.dim}'
        q0, p0 = np.array(qp0[:self.dim][0]), np.array(qp0[self.dim:][0])
        
        if q0.shape == ():
            q0 = q0.reshape((1, 1))
        if p0.shape == ():
            p0 = p0.reshape((1, 1))
            
        if len(q0.shape) == 1:
            if self.dim == 1:
                q0 = q0.reshape((q0.shape[0], 1))
            else: 
                # here we will assume the user inputs one single vector according to the Hamiltonian dim.
                # a check will be done later
                q0 = q0.reshape((1, q0.shape[0]))
        if len(p0.shape) == 1:
            if self.dim == 1:
                p0 = p0.reshape((p0.shape[0], 1))                
            else:
                # here we will assume the user inputs one single vector according to the Hamiltonian dim.
                # a check will be done later
                p0 = p0.reshape((1, p0.shape[0]))
            
        assert q0.shape == p0.shape, f'Inconsistent input shapes: {q0.shape} != {p0.shape}.'
        return q0, p0
    
    def __call__(self, *xieta0, real=False, **kwargs):
        '''
        Apply the solver on the start coordinates.
        
        Parameters
        ----------
        *qp0: 
            Start vector(s)
        
        t: float, optional
            Integration length/intervall
            
        real: boolean, optional
            If True, assume the input are real-valued q and p values. If False, assume the input
            is given in terms of (complex) xi and eta values.
            
        **kwargs
            Optional keyworded arguments passed to self.ensemble_propagation routine.
        '''        
        # homogenization of input
        xi0, eta0 = self._homgenInput(*xieta0)
        if not real:
            # for the Heyoka solver we have to use the real-valued counterparts
            sqrt2 = float(np.sqrt(2))
            q0 = ((xi0 + eta0)/sqrt2).real # xi = (q + p*1j)/sqrt2, eta = (q - p*1j)/sqrt2 .
            p0 = ((xi0 - eta0)/sqrt2/1j).real
        else:
            q0 = xi0
            p0 = eta0
            
        # call the ensemble propagation
        self.results = self.ensemble_propagation(q0=q0, p0=p0, **kwargs)
        out = np.array([e[0].state for e in self.results])
        qf, pf = out[:,:self.dim][:,0], out[:,self.dim:][:,0]
        
        if not real:
            # transform back to xi/eta values
            xif = (qf + pf*1j)/sqrt2
            etaf = (qf - pf*1j)/sqrt2
            return xif, etaf
        else:
            return qf, pf
        
    

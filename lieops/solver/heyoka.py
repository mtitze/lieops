import heyoka as hy
import numpy as np

from .common import realHamiltonEqs

def trf(func):
    '''
    Return wrapper to perform transformations to/from complex coordinates within the heyoka_solver class.
    '''
    def xietaqp(self, *xieta0, real=False, **kwargs):
        if not real:
            # Transform to qp-coordinates and propagate their real and imaginary parts separately.
            qp0 = self._xieta2qp(*xieta0)
            if len(qp0.shape) == 1: # the case the user inputs a single point
                qp0 = qp0.reshape(qp0.shape[0], 1)
            qp1 = np.concatenate([qp0.real, qp0.imag], axis=1)
            out1 = func(self, qp1, **kwargs)
            out = out1[..., :out1.shape[-1]//2] + out1[..., out1.shape[-1]//2:]*1j # the dots here indicate either a single index running over the dimension -- or two indices: an index running over the dimension and another index corresponding to given t-values. The final index corresponds to the number of different start vectors.
            out = self._qp2xieta(*out)
        else:
            qp0 = np.array(xieta0).real
            if len(qp0.shape) == 1: # the case the user inputs a single point
                qp0 = qp0.reshape(qp0.shape[0], 1)
            out = func(self, qp0, **kwargs)
        return out
    return xietaqp


class heyoka_solver:
    
    def __init__(self, hamiltonian, **kwargs):
        self.hamiltonian = hamiltonian
        self.dim = self.hamiltonian.dim
        self.variables, self.hameqs, self.realHamiltonian = self.getHamiltonEqs(**kwargs)
        self.integrator = hy.taylor_adaptive(self.hameqs, [0]*self.dim*2)
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

    @trf
    def ensemble_propagation(self, qp0, kind='until', **kwargs):
        '''
        Ensemble propagation according to https://bluescarni.github.io/heyoka/tut_ensemble.html
        
        Parameters
        ----------
        qp0: array-like of shape (K, 2*dim) 
            Input vector(s) denoting the coordinates. The number 'K' denotes the number of different
            vectors to be tracked, while 'dim' must correspond to the dimension of the current hamiltonian.
                        
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
        
        assert len(qp0.shape) == 2
        dim2, n_iter = qp0.shape
        assert dim2//2 == self.dim, f'Input vector dimension {dim2//2} not consistent with dimension {self.dim} of given Hamiltonian.'
        
        t = kwargs.get('t', self.t)
        if hasattr(t, '__iter__'):
            kind = 'grid'
        
        # create a generator, taking a taylor adaptive object and modifying the state according to the
        # vector components.
        # The first index should be related to the dimension (i.e indexing the number of components of q and p),
        # while the second index should correspond to the number of variations. Therefore:
        def gen(tacp, i):
            tacp.state[:] = qp0[:, i]
            return tacp

        if kind == 'until':
            self.results = hy.ensemble_propagate_until(self.integrator, n_iter=n_iter, gen=gen, t=t, write_tc=True, c_output=True)
            return np.array([e[0].state for e in self.results]).transpose()        
        elif kind == 'for':
            self.results = hy.ensemble_propagate_for(self.integrator, n_iter=n_iter, gen=gen, delta_t=t, write_tc=True, c_output=True)
            return np.array([e[0].state for e in self.results]).transpose()
        elif kind == 'grid':
            self.results = hy.ensemble_propagate_grid(self.integrator, n_iter=n_iter, gen=gen, grid=t)
            return np.array([e[5] for e in self.results]).transpose()
        else:
            raise RuntimeError(f"Requested kind '{kind}' not recognized.")
                
    def _xieta2qp(self, *xieta, **kwargs):
        '''
        Helper class to deal with user-defined input.
        '''
        assert len(xieta) == 2*self.dim, f'Length of input {len(xieta)}, expected: {2*self.dim}'
        xi, eta = np.array(xieta[:self.dim]), np.array(xieta[self.dim:])     
        sqrt2 = float(np.sqrt(2))
        q = (xi + eta)/sqrt2
        p = (xi - eta)/sqrt2/1j
        return np.concatenate([q, p], axis=0)
    
    def _qp2xieta(self, *qp, **kwargs):
        '''
        Helper class to deal with user-defined input.
        '''
        assert len(qp) == 2*self.dim, f'Length of input {len(qp)}, expected: {2*self.dim}'
        q, p = np.array(qp[:self.dim]), np.array(qp[self.dim:])     
        sqrt2 = float(np.sqrt(2))
        xi = (q + p*1j)/sqrt2 
        eta = (q - p*1j)/sqrt2
        return np.concatenate([xi, eta], axis=0)
    
    def __call__(self, *xieta0, **kwargs):
        '''
        Apply the solver on start coordinates; this is a short-cut for calling
        self.ensemble_propagation and using self._xieta2qp and self._qp2xieta helper routines.
        
        For repeated application inside a loop, these helper routines should be 
        applied in front and behind of the loop.
        
        Parameters
        ----------
        *xieta0: 
            Start vector(s)
        
        t: float, optional
            Integration length/intervall
            
        real: boolean, optional
            If True, assume the input are real-valued q and p values. If False, assume the input
            is given in terms of (complex) xi and eta values.
            
        **kwargs
            Optional keyworded arguments passed to self.ensemble_propagation routine.
        '''  
        return self.ensemble_propagation(*xieta0, **kwargs)

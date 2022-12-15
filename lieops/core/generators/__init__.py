from njet import derive
from .genfunc import *

def create(generator, power: int):
    '''
    Generic routine to create a generator for a Lie operator out of a 
    function depending on a single variable. The generator coefficients are determined 
    from the Taylor coefficients of the input. 
    
    Parameters
    ----------
    generator: callable
        A function depending on a single parameter, which is supported by the njet module.
        
    power: int
        Define the maximal order up to which the Taylor coefficients will be determined.
    
    Returns
    -------
    list
        A list of coefficients a_k so that generator(x) ~ sum_k a_k x**k holds.
    '''
    assert generator.__code__.co_argcount == 1, 'Generator function needs to depend on a single variable.'
    dg = derive(generator, order=power)
    taylor_coeffs = dg(0, mult_drv=False)
    return [taylor_coeffs.get((k,), 0) for k in range(len(taylor_coeffs))]

import numpy as np

def make_random_cmplx(n):
    '''
    Create a random complex number with real and imaginary parts between -1 and 1.
    '''
    return [complex(e) for e in 1 - np.random.rand(n)*2 + (1 - np.random.rand(n)*2)*1j] # We change to default complex numbers to prevent numpy operator overloading if multiplying polynomials with those quantities from left.

def get_max(delta):
    '''
    Return the maximal value of the absolute values of a polynomial.
    '''
    if len(delta) > 0:
        diff = max(np.abs(list(delta.values())))
    else:
        diff = 0
    return diff

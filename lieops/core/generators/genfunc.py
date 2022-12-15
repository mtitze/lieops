# This script contains routines which returns the generating series for various functions. 
from scipy.special import bernoulli
from njet.jet import factorials

def genexp(power, t=1):
    '''
    Return the coefficients of the exponential function.
    '''
    facts = factorials(power)
    return [t**k/facts[k] for k in range(len(facts))]

def gen0(power, t=1):
    '''
    Return the coefficients of the function
    f(z) := z/(1 - exp(-z))
    
    Note that this series converges for all z if |z| < 2*pi.
    '''
    power2 = power*2
    bernoulli_numbers = bernoulli(power2)
    facts = factorials(power2)
    return [1, t/2] + [bernoulli_numbers[k]/facts[k]*t**k for k in range(2, power2, 2)]

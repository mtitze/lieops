# This script contains routines which returns the generating series for various functions. 
from njet.jet import factorials

def genexp(power):
    # The generator of exp(x)
    facts = factorials(power)
    return [1/facts[k] for k in range(len(facts))]
# This script contains routines which returns the generating series for various functions. 
from njet.jet import factorials

def genexp(power, t=1):
    # The generator of exp(t)
    facts = factorials(power)
    return [t**k/facts[k] for k in range(len(facts))]
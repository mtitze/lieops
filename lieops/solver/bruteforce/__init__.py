'''
Collection of routines to perform calculations involving a Lie operator, by directly summing up terms up to
specific orders (aka: "brute force").
'''

from .bruteforce import calcFlow as _calcFlow

def flow(lo, **kwargs):
    return _calcFlow(lo=lo, **kwargs)
    
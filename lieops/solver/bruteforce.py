'''
Collection of routines to perform calculations involving a Lie operator, by directly summing up terms up to
specific orders (aka: "brute force").
'''

def action(lo, y, **kwargs):
    '''
    Apply the Lie operator g(:x:) to a given lie polynomial y, to return the elements
    in the series of g(:x:)y.

    Parameters
    ----------
    lo: lieoperator
        The Lie operator to be used.
    
    y: poly
        The Lie polynomial on which the Lie operator should be applied on.

    Returns
    -------
    list
        List containing (g[n]*:x:**n)(y) if g = [g[0], g[1], ...] denotes the underlying generator.
        The list goes on up to the maximal power N determined by self.argument.ad routine (see
        documentation there).
    '''
    assert hasattr(lo, 'generator'), 'No generator set (check self.set_generator).'
    assert hasattr(lo, 'argument'), 'No Lie polynomial in argument set (check self.set_argument)'
    ad_action = lo.argument.ad(y, power=lo.power)
    assert len(ad_action) > 0
    # N.B. if self.generator[j] = 0, then k_action[j]*self.generator[j] = {}. It will remain in the list below (because 
    # it always holds len(ad_action) > 0 by construction).
    # This is important when calculating the flow later on. In order to check for this consistency, we have added 
    # the above assert statement.
    return [ad_action[j]*lo.generator[j] for j in range(len(ad_action))]

def calcOrbits(lo, components):
    '''
    Compute the summands in the series of the Lie operator g(:x:)y, for every requested y,
    by utilizing the routine self.action.

    Parameters
    ----------
    lo: lieoperator
        The Lie operator to be used.
        
    components: list
        List of poly objects on which the Lie operator g(:x:) should be applied on.
        If nothing specified, then the canonical xi-coordinates are used.

    Returns
    -------
    list
        A list containing the actions [(g[n]*:x:**n)(y) for n=0, ..., N] (see self.action) as elements, 
        where y is running over the requested Lie-operator components.
    '''
    return [action(lo, y) for y in components] # orbits: A list of lists

def calcFlow(orbits, t):
    '''
    Compute the Lie operators [g(t:x:)]y for a given parameter t, for every y in self.components.

    Parameters
    ----------
    orbits: list
        The orbits to be used.
        
    t: float (or e.g. numpy array), optional
        Parameter in the argument at which the Lie operator should be evaluated.

    Returns
    -------
    list
        A list containing the flow of every component function of the Lie-operator.
    '''
    return [sum([orbits[k][j]*t**j for j in range(len(orbits[k]))]) for k in range(len(orbits))]

def flowFunc(orbits, t, *z):
    '''
    Return the flow function phi(t, z) = [g(t:x:) y](z).

    Parameters
    ----------
    orbits: list
        The orbits to be used.
        
    t: float
        The requested t-value of the flow.

    z: subscriptable
        The point at which to evaluate the flow.

    Returns
    -------
    phi: callable
        The flow of the current Lie operator, as described above.
    '''
    return [sum([orbits[k][j](*z)*t**j for j in range(len(orbits[k]))]) for k in range(len(orbits))]

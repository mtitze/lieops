def nc(*args):
    '''
    Returns the nested commutator expression (sometimes called Chibrikov basis)
    x1@(x2@(x3@(x4@ .... )))
    '''
    out = args[-1]
    for x in args[len(args) - 2::-1]:
        out = x@out
    return out

def bch(x, y):
    '''
    Compute the explicit commutators of the Baker-Campbell-Hausdorff series for
    orders up and including 7.
    '''
    result = {}
    result[1] = x + y
    result[2] = 0.5*(x@y)
    result[3] = 1/12*(nc(x, x, y) + nc(y, y, x))
    result[4] = -1/24*nc(y, x, x, y)
    #result[5] = 1/180*nc(y, y, x, x, y) - 1/360*(x@y)@nc(y, x, y) - 1/180*nc(y, x, x, x, y) - 1/120*(x@y)@nc(x, x, y) + 1/720*nc(y, y, y, x, y) - 1/720*nc(x, x, x, x, y)
    result[5] = -1/720*(nc(y, y, y, y, x) + nc(x, x, x, x, y)) + \
                1/360*(nc(x, y, y, y, x) + nc(y, x, x, x, y)) + \
                1/120*(nc(y, x, y, x, y) + nc(x, y, x, y, x))
    # From Wiki (fails at symmetry test!)
    #result[6] = 1/240*nc(x, y, x, y, x, y) + \
    #            1/720*(nc(x, y, x, x, x, y) - nc(x, x, y, y, x, y)) + \
    #            1/1440*(nc(x, y, y, y, x, y) - nc(x, x, y, x, x, y))
    # Instead, we will use the results from the BCH routine of H. Hofstaetter (https://github.com/HaraldHofstaetter/BCH)
    result[6] = -1/1440*nc(y, x, x, x, y, x) + 1/720*nc(y, y, x, x, y, x) - 1/240*nc(y, x, y, x, y, x) + 1/1440*nc(y, y, y, x, y, x) - 1/720*nc(y, x, y, y, y, x)
    result[7] = -1/30240 * nc(x, x, x, x, x, y, x) + \
                 1/10080 * nc(y, x, x, x, x, y, x) - \
                 1/10080 * nc(x, y, x, x, x, y, x) - \
                 1/3360  * nc(y, y, x, x, x, y, x) - \
                 1/5040 *  nc(x, x, y, x, x, y, x) + \
                 1/1260 *  nc(y, x, y, x, x, y, x) + \
                 1/7560 *  nc(x, y, y, x, x, y, x) - \
                 1/7560 *  nc(y, y, y, x, x, y, x) + \
                 1/10080 * nc(x, x, y, y, x, y, x) - \
                 1/1008 *  nc(x, y, x, y, x, y, x) + \
                 1/3360 *  nc(y, y, x, y, x, y, x) + \
                 1/1680 *  nc(y, x, y, y, x, y, x) - \
                 1/3360 *  nc(x, y, y, y, x, y, x) - \
                 1/10080 * nc(y, y, y, y, x, y, x) - \
                 1/5040 *  nc(x, y, x, y, y, y, x) + \
                 1/2520 *  nc(y, y, x, y, y, y, x) - \
                 1/10080 * nc(x, y, y, y, y, y, x) + \
                 1/30240 * nc(y, y, y, y, y, y, x)
    return result

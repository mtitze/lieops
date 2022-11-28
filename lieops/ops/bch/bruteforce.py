import os
import pandas as pd
from tqdm import tqdm
import warnings

def _load_bch_data(filename):
    '''
    Load a pre-calculated file for the Baker-Campbell-Hausdorff coefficients up and
    including order 20. The code which has been used to generate the raw data
    is credited to H. Hofstaetter in https://github.com/HaraldHofstaetter/BCH.
    
    This routine converts the .txt file bch20.txt created with the command
    "./bch N=20 table_output=1 > bch20.txt" to a Pandas dataframe object.
    
    A .csv file can subsequently be generated with the command (e.g. data denotes the dataframe)
    data.to_csv('bch20.csv', index=False).
    
    Parameters
    ----------
    filename: str
        The name of the output .txt file as described above.
        
    Returns
    -------
    dataframe
        A Pandas dataframe object whose columns have the following meaning:
        Let A and B be two operators, whose BCH expression Z(A, B) = log(exp(A) o exp(B)) has
        to be computed. Then Z(A, B) = sum_(j = 0)^N x_j*c_j(A, B), where the operators c_j
        are given in terms of nested commutator expressions of A and B, and x_j is some rational
        coefficient.
        
        'index': The index j of the basis element c_j.
        'order': The number of involved operators A and B in the expression of c_j.
         'left': The index k so that c_j = [c_k, c_l] holds.
        'right': The index l so that c_j = [c_k, c_l] holds.
          'nom': The nominator of x_j.
        'denom': The denominator of x_j.
        'coeff': The coefficient x_j.
    '''
    data = pd.read_csv(raw_data, sep="\t| |/", header=None, engine='python')#, delimiter = "\t")
    data.columns = ["index", "order", "left", "right", "nom", "denom"]
    data.drop(columns=["index"], inplace=True) # drop unecessary information
    data.drop(data[data.nom == 0].index, inplace=True) # drop items which are zero anyways
    data = data.astype({"nom": float, "denom": float})
    data['coeff'] = data.apply(lambda x: x.nom/x.denom, axis=1) # compute the coefficients
    return data

def bch(A, B, order, database=[], output_format='order', **kwargs):
    '''
    Compute the terms in the Baker-Campbell-Hausdorff series for
    orders up and including "order".
    
    Z(A, B) = log(exp(A) o exp(B))
    
    Parameters
    ----------
    A: poly
        The first operator.
        
    B: poly
        The second operator.
        
    database: Pandas dataframe, optional
        A dataframe which can hold the BCH coefficients up to a desired order. See _load_bch_data routine for more
        information.
        
    output_format: str, optional
        Define how the output should be returned:
             "order": Return a dictionary, where the key 'k' denotes the contribution of the k-th order to Z.
        "individual": Return a dictionary holding the individual non-zero entries. The keys are hereby deduced from
                      the indices of the given database.
                      
    **kwargs
        Optional keyworded arguments to control the progress bar output.
        
    Returns
    -------
    dict
        A dictionary of poly objects which contribute to Z(A, B), as described above.
    '''
    # check & prepare input
    assert output_format in ['order', 'individual'], f'Requested output format {output_format} not understood.'
    if len(database) == 0:
        database_filename = os.path.join(os.path.dirname(__file__), 'bch20.csv')
        database = pd.read_csv(database_filename)
    max_order = int(database.iloc[-1]['order'])
    if order > max_order:
        warnings.warn(f"Requested order {order} exceeds the maximal supported order {max_order} in the database.")
    database = database.drop(database[database.order > order].index)
    
    # initiate calculation
    commutators = {0: A, 1: B} # the commutators needs to be stored without their coefficients, because some 
                               # higher-order expressions are not zero, while they depend on lower-order 
                               # expressions which may have zero as coefficients.
    results = {0: A, 1: B} # here the results are stored *with* their coefficients
    pbar = tqdm(range(2, len(database)), 
                leave=kwargs.get('leave_tqdm', True), 
                disable=kwargs.get('disable_tqdm', False)) # a progress bar to show the progress of the calculation
    for j in pbar:
        data_j = database.iloc[j]
        index = int(data_j['index'])
        k = int(data_j['left'])
        l = int(data_j['right'])
        coeff = data_j['coeff']
        commutators[index] = commutators[k]@commutators[l]
        if coeff != 0:
            results[index] = commutators[index]*coeff
    if output_format == 'individual':
        return results
    elif output_format == 'order':
        # return a result which gives the individual contributions for each order
        return {k: sum([results[w] for w in database.loc[(database['order'] == k) & (database['coeff'] != 0)]['index']]) for k in database['order'].unique()}

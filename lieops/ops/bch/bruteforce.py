import os
import pandas as pd
from tqdm import tqdm
import warnings

def _load_bch_data(filename):
    '''
    Load a pre-calculated file for the Baker-Campbell-Hausdorff coefficients up and
    including order 20. The code which has been used to generate the raw data
    is credited to H. Hofstaetter in https://github.com/HaraldHofstaetter/BCH.
    
    This routine converts the .txt file bch20.txt contained in lieops.ops.bch into
    a pandas dataframe and returns it. This file has been created with the command
    "./bch N=20 table_output=1 > bch20.txt"
    
    A .csv file can then be generated with the command (e.g. data denotes the dataframe)
    data.to_csv('bch20.csv', index=False).
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

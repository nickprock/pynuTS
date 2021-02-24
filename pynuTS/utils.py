"""
Created on Sat Feb 20 2021

@project: pynuTS
@author: nicola procopio
@description: some utils functions
@last_update: 20/02/2021
"""

import numpy as np

def split_sequence(sequences_in, sequence_out: int = None, n_steps_in: int, n_steps_out: int = 1, univariate: bool = True):
	X, y = list(), list()
    if univariate==True:
	    for i in range(len(sequences_in)):
		    end_ix = i + n_steps_in
		    out_end_ix = end_ix + n_steps_out
		    if out_end_ix > len(sequences_in):
			    break
		    seq_x, seq_y = sequences_in[i:end_ix], sequences_in[end_ix:out_end_ix]
		    X.append(seq_x)
		    y.append(seq_y)
    else:
        for i in range(len(sequences_in)):
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            if out_end_ix > len(sequences_in):
               break
            seq_x, seq_y = sequences[i:end_ix].remove(sequence_out), sequences_in[end_ix:out_end_ix, sequence_out]
            X.append(seq_x)
            y.append(seq_y)
    
	return np.array(X), np.array(y)
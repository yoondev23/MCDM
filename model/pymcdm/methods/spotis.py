# Copyright (c) 2020 Andrii Shekhovtsov

import numpy as np
from .. import normalizations
from .mcda_method import MCDA_method

class SPOTIS(MCDA_method):
    def __init__(self):
        """Create SPOTIS method object."""
        pass

    def __call__(self, matrix, weights, types, bounds, *args, **kwargs):
        """Rank alternatives from decision matrix `matrix`, with criteria weights `weights` and criteria types `types`.

Parameters
----------
    matrix : ndarray
        Decision matrix / alternatives data.
        Alternatives are in rows and Criteria are in columns.

    weights : ndarray
        Criteria weights. Sum of the weights should be 1. (e.g. sum(weights) == 1)

    types : ndarray
        Array with definitions of criteria types:
        1 if criteria is profit and -1 if criteria is cost for each criteria in `matrix`.

    bounds : ndarray
        Each row should contain min and max values for each criterion. Min and max should be different values!

    *args and **kwargs are necessary for methods which reqiure some additional data.

Returns
-------
    ndarray
        Preference values for alternatives. Better alternatives have smaller values.
"""
        SPOTIS._validate_input_data(matrix, weights, types)
        if np.any(bounds[:, 0] == bounds[:, 1]):
            eq = np.arange(bounds.shape[0])[bounds[:, 0] == bounds[:, 1]]
            raise ValueError(
                    f'Bounds for criteria {eq} are equal. Consider changing min and max values for this criterion, delete this criterion or use another MCDA method.'
                )

        # Determine Ideal Solution Point based on criteria bounds
        isp = bounds[np.arange(bounds.shape[0]), ((types+1)//2).astype('int')]
        return SPOTIS._spotis(matrix, weights, isp, bounds)

    @staticmethod
    def _spotis(matrix, weights, isp, bounds):
        nmatrix = matrix.astype(float)
        # Normalized distances matrix (d_{ij})
        nmatrix = np.abs((nmatrix - isp)/
                         (bounds[:,0] - bounds[:,1]))
        # Distances to ISP (smaller means better alt)
        raw_scores = np.sum(nmatrix * weights, axis=1)
        return raw_scores

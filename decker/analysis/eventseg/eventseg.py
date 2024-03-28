from sklearn.model_selection import LeaveOneOut, KFold, train_test_split
from brainiak.eventseg.event import EventSegment
import numpy as np
import pandas as pd


def innerloop_optimize_K(train_data, test_data, k_vals:np.ndarray):
    """Inner loop for NCV
    
    Parameters
    ----------
    train_data
        Training data, parsed by sklearn.model_selection.LeaveOneOut()
    test_data
        Test data, parsed by sklearn.model_selection.LeaveOneOut()    
        
    k_vals: np.ndarray
        Numpy array which contains all possible k values
    """

    log_likelihoods = np.zeros_like(k_vals)

    # test all the K values
    for i,K in enumerate(k_vals):

        # instantiate
        HMM = EventSegment(K)

        # fit and get log likely hoods
        HMM.fit(train_data)
        _, ll = HMM.find_events(test_data)
        log_likelihoods[i] = ll

    # find the best K value
    bestK = k_vals[np.argmax(log_likelihoods)]

    return bestK, np.max(log_likelihoods)
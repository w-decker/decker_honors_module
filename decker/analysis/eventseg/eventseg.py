from sklearn.model_selection import LeaveOneOut, KFold, train_test_split
from brainiak.eventseg.event import EventSegment
import numpy as np
import pandas as pd


def tuneK(train_data, test_data, k_vals:np.ndarray):
    """Tune K inside fold
    
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

    print('\t\tTuned on k')

    return bestK, np.max(log_likelihoods)

def innerLoop(n_splits: int, idx: np.ndarray, outer_train_idx:np.ndarray, data:np.ndarray, k_vals:np.ndarray) -> int:

    # inner loop fold
    subj_id_all_inner = idx[outer_train_idx]
    kf = KFold(n_splits=n_splits)
    kf.get_n_splits(outer_train_idx)

    # track
    out = []
            
    # run loop
    for subj_id_train_inner, subj_id_test_inner in kf.split(subj_id_all_inner):

        # format data
        train_data = np.mean(data[subj_id_train_inner, :, :],axis=0)
        test_data = data[int(subj_id_test_inner[0]), :, :]

        # # run inner loop
        bestK, maxLL = tuneK(train_data=train_data, test_data=test_data, k_vals=k_vals)

        # append to tracker
        out.append((bestK, maxLL))

        print('\tFold complete')

    # get best k
    k = [i[0] for i in out]
    ll = [i[1] for i in out]
    best = k[np.argmax(ll)]

    return best

def outerLoop(idx:np.ndarray, n_splits:int, data, k_vals:np.ndarray) -> pd.DataFrame:

    print('\tBeginning loop!\n')
    print('\t---------------\n')

    # initialize dataframe
    results = pd.DataFrame(columns=['subject', 'k', 'll'])

    # set up loo
    loo_outer = LeaveOneOut()
    subj_id_struct = idx
    loo_outer.get_n_splits(subj_id_struct)

    for subj_id_train_outer, subj_id_test_outer in loo_outer.split(subj_id_struct):

        # run inner loop
        k2test = innerLoop(n_splits=n_splits, outer_train_idx=subj_id_train_outer, data=data, k_vals=k_vals, idx=idx)

        # test on outer subject
        HMM = EventSegment(k2test)

        # outer_test
        outer_test_data = data[int(subj_id_test_outer[0]), :, :]

        # fit and get log likely hoods
        print(f'\tFit to outer loop testing data\n')
        HMM.fit(outer_test_data)
        _, ll = HMM.find_events(outer_test_data)

        # aggregate values
        out = [str(subj_id_test_outer), k2test, ll]

        # append to dataframe
        results.loc[len(results)] = out

    return results
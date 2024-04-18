from sklearn.model_selection import LeaveOneOut, KFold, train_test_split
from brainiak.eventseg.event import EventSegment
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from collections import namedtuple

def tuneK(train_data, test_data, k_vals:np.ndarray, split_merge:bool=False, verbose:bool=False):
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
        if split_merge:
            HMM = EventSegment(K, split_merge=split_merge)
        else:
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
        test_data = np.mean(data[subj_id_test_inner, :, :], axis=0)

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

def eventseg(data, idx:np.ndarray, k_vals:np.ndarray, n_splits:int, n_TR:int, split:bool=True, verbose:bool=False, concat:bool=False):
    """Single function that runs full leave-one-out nested cross validation
    
    Parameters
    ----------
    data:np.ndarray
        Stacked data in which subjects = axis(0), TRs = axis(1), ROIs = axis(2)
        
    idx:np.ndarray
        Array of ascending integers with length equal to total number of subjects.
        
    k_vals:np.ndarray
        All k-values to test
        
    n_splits:int
        How many splits to run on the inner loop?
        
    n_TR:int
        How many TRs?
    
    split:bool=True
        Use split_merge
        
    verbose:bool=False
    
    concat:bool=False
        If true, computes model fit over average of training sets instead of over each training set individually.
        Helps with data managing and run time.
        
    """
    # print
    if verbose:
        print()
        print('Running: Leave-one-out nested cross validation')
        print('----------------------------------------------\n')

    # some variables needed
    outer = LeaveOneOut()
    best_k = np.zeros_like(idx)
    best_ll = np.zeros((len(idx), len(idx)))
    ll_outer = np.zeros_like(idx)
    events = np.zeros((len(idx), n_TR))
    loglik = np.zeros((len(idx),len(idx),len(k_vals)))
    all_ll = np.zeros((n_splits,len(k_vals)))

    # start outer loop
    for outer_train_idx, outer_test_idx, in outer.split(idx):

        # print
        if verbose:
            print("Outer:\tTrain:", outer_train_idx, "Test:", outer_test_idx)

        # set up inner loop variables
        inner_idx = idx[outer_train_idx]
        kf = KFold(n_splits=n_splits)
        kf.get_n_splits(outer_train_idx)
        jj = 0 # update variable

        # print
        if verbose:
            print('\nInner:')

        # start inner loop
        for inner_train_idx, inner_test_idx in kf.split(inner_idx):

            # parse data by splits (is this redundant?)
            inner_train_idx = inner_idx[inner_train_idx]
            inner_test_idx = inner_idx[inner_test_idx]

            # subset data
            data_train = data[inner_train_idx]
            data_val = data[inner_test_idx]
            data_test = np.squeeze(data[outer_test_idx], axis=0)
        
            # print
            if verbose:
                print("-Train:", inner_train_idx,"Validate:", inner_test_idx)

            # begin loop to tune K
            for i, k in enumerate(k_vals):

                # instantiate EventSegment class
                if split:
                    HMM = EventSegment(n_events=k, split_merge=True)
                else:
                    HMM = EventSegment(n_events=k)

                # fit
                if concat:
                    HMM.fit(data_train.mean(0))
                else:
                    data_train_list = []
                    for i in range(len(data_train)):
                        data_train_list.append(data_train[i, :, :])
                        
                    HMM.fit(data_train_list)

                # test
                _, ll, = HMM.find_events(data_val.mean(0))

                # store ll
                loglik[outer_test_idx,jj,i] = ll

            # update other trackers
            all_ll[jj,:]=loglik[outer_test_idx,jj,:] # all log-likelihoods
            best_ll[outer_test_idx, jj] = np.max(loglik[outer_test_idx,jj,:]) # best log-likelihood
            jj += 1

        # update more trackers
        mean_ll = np.mean(all_ll, axis=0) # mean of log-likelihood
        fill = mean_ll.argsort() # sort by log-likelihood value
        fill = fill[len(k_vals)-1] # find best K
        best_k[outer_test_idx] = k_vals[fill] # store best K for test

        # get k to test
        k2test = best_k[outer_test_idx]

        # instantiate EventSegment class
        if split:
            HMM = EventSegment(n_events=k2test[0], split_merge=True)
        else:
            HMM = EventSegment(n_events=k2test[0])


        # # test and find events
        HMM.fit(data_test)
        _, ll = HMM.find_events(data_test)
        ll_outer[outer_test_idx] = np.max(HMM.ll_)
        events[outer_test_idx, :] = np.argmax(HMM.segments_[0], axis=1)

    if verbose:
        print("\nDone!")

    return {"Best_k": best_k, "events": events, "Best_LL":best_ll, "LL":loglik}

def eventseg_bounds(data:np.ndarray, k:int, w:int, n_perm:int, n_subs:int, n_TR:int, verbose:bool=False) -> namedtuple:
    """Statistically test event boundaries across subject
    
    Parameters
    ----------
    data: np.ndarray
        Axis 0 must be number of subjects
        
    k: int
        Number of events to fit
    
    w: int
        Window of TRs to test
        
    n_perm: int
        Number of permutations
    
    n_subs: int
        How many subjects do you have?
        
    n_TR: int
        How many TRs?

    verbose:bool, defalt = False
        Print individual correlations?
        
    Return
    -------
    out: namedtuple
        Field with data computed"""
    
    # empty mat
    within_across = np.zeros((n_subs, n_perm+1))

    # leave-one-out simple
    for left_out in range(n_subs):

        # Fit to all but one subject
        ev = EventSegment(k, split_merge=True)
        ev.fit(data[np.arange(n_subs) != left_out, :, :].mean(0))
        events = np.argmax(ev.segments_[0], axis=1)

        # Compute correlations separated by w in time
        corrs = np.zeros(n_TR-w)
        for t in range(n_TR-w):
            corrs[t] = pearsonr(data[left_out, t, :],data[left_out,t+w,:])[0]
        _, event_lengths = np.unique(events, return_counts=True)

        # Compute within vs across boundary correlations, for real and permuted bounds
        np.random.seed(0)
        for p in range(n_perm+1):
            within = corrs[events[:-w] == events[w:]].mean()
            across = corrs[events[:-w] != events[w:]].mean()
            within_across[left_out, p] = within - across
            # 
            perm_lengths = np.random.permutation(event_lengths)
            events = np.zeros(n_TR, dtype=np.int)
            events[np.cumsum(perm_lengths[:-1])] = 1
            events = np.cumsum(events)

        if verbose:
            print('Subj ' + str(left_out+1) + ': within vs across = ' + str(within_across[left_out,0]))

        # create output in namedtuple class
        BoundaryTest = namedtuple('BoundaryTest', ['within', 'across', 'within_across', 'events'])
        BoundaryTest = BoundaryTest(within=within, across=across, within_across=within_across, events=event_lengths)

        return BoundaryTest

def across_subs_event_corr(data: np.ndarray, k: int, n_subs:int, split_merge:bool=False) -> "list[np.ndarray]":
    """Calculate boundary correlations within and across subjects
    
    Parameters
    ----------
    data: np.ndarray
        Data where axis 0 is subjects
        
    k: int 
        Common k to fit all subjects too
        
    n_subs: int
        How many subjects?
        
    split_merge:bool, default = False
        Use split_merge in fitting procedure?
        
    Return
    -------
    out: list[np.ndarray]
        List with event labels, correlations and aggregate correlations"""

    # calculate events on common K
    subwise_events = {}

    # loop over each sub to calculate events
    for i in range(n_subs):
        if split_merge:
            HMM = EventSegment(k, split_merge=True)
        else:
            HMM = EventSegment(k, split_merge=False)

        HMM.fit(data[i])
        events = np.argmax(HMM.segments_[0], axis=1)
        subwise_events[i] = events

    # aggregate tracker
    ag = {}

    # loop over each subject's events
    for sub in range(n_subs):

        # extract their events and respective indices
        ev, ev_idx = np.unique(subwise_events[sub], return_index=True)

        # make event-level dict
        event_level = {}

        # get average timecourse of that event
        for i in np.arange(len(ev) - 1):
            _subset_data = data[sub, ev_idx[i]:ev_idx[i + 1], :]
            _avg = _subset_data.mean(0)

            # append
            event_level[i] = _avg

        # append to subject-level dict
        ag[sub] = event_level

    # aggregate over subs one more time
    ag2 = {}

    # loop over subjects
    for sub in range(n_subs):

        # create event-level storage
        l = []
        for i in ag[sub]:
            l.append(ag[sub][i])

        # calculate correlations
        cor = np.corrcoef(l)

        # append
        ag2[sub] = cor

    out = [subwise_events,ag, ag2]

    return out
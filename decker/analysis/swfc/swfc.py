from pydfc import data_loader
import numpy as np
from pydfc.dfc_methods import SLIDING_WINDOW
from pydfc import time_series
from nilearn.connectome import ConnectivityMeasure
from decker.analysis.utils.utils import vectorize_matrices, devectorize_centers
from sklearn.cluster import KMeans
import timecorr as tc

def swfc_combine(BOLD_multi_object:time_series.TIME_SERIES, subids:"list[str]") -> "dict[np.ndarray]":
    """Combine SWFC using Tobari et. al package
    
    Returns
    -------
    cswfc: dict[np.ndarray]
        Dictionary, "combined swfc", with key-value pairs are subject-data pairs.
    """

    # predefined parameters
    params_methods = {
        # W is window length in sec
        "W": 44,
        "n_overlap": 0.5,
        "sw_method": "pear_corr",
        "tapered_window": True,
        # data Parameters
        "normalization": True,
        "num_select_nodes": None,  # you can make the number of nodes smaller, e.g. by setting to 50, for faster computation
    }

    # initialize
    cswfc = {}

    print("Beginning...")
    print(f'------------\n')

    for i in subids:

        # compute SWFC
        measure = SLIDING_WINDOW(**params_methods)
        dFC = measure.estimate_dFC(time_series=BOLD_multi_object.get_subj_ts(subjs_id=i))

        # output
        print(f'SWFC computed for {i}\n')

        # append subject-wise corr matrices
        cswfc[i] = dFC.get_dFC_mat()

        print(f'\tAppended to average\n')

    print(f'------------\n')
    print('Done')

    return cswfc

def basic_sliding_window(data:np.ndarray, k:int, overlap:float=0.5) -> np.ndarray:
    """Calculate basic sliding window functional connectivity with optional overal
    
    Paramters
    ---------
    data: np.ndarray
        Two-dimensional array where axis 0 is number of samples (time points)

    k: int
        Window size. Will equate to number timepoints. So make sure you know how many TRs are in your data and the length of TRs

    overlap: float, default = 0.5
        How much overlap should each window have?
    """

    # get number of time points
    n_samples = data.shape[0]

    # extract windows
    n_windows = int(np.ceil((n_samples - k) / (k * (1 - overlap)))) + 1

    # instantiate CM class and tracker
    connectivity_measure = ConnectivityMeasure(kind="correlation", standardize="zscore_sample")
    dynamic_connectivity = []

    # oop over windows
    for i in range(n_windows):
        start_index = int(i * k * (1 - overlap))
        end_index = min(start_index + k, n_samples)
        window_data = data[start_index:end_index]
        
        # calculate FC at window and append
        corr_mat = connectivity_measure.fit_transform([window_data])[0]
        dynamic_connectivity.append(corr_mat)

    return np.array(dynamic_connectivity)

def run_kmeans(data:np.ndarray, n_clusters:int):
    """Run k-means clustering on sliding window connectivity
    
    Parameters
    -----------
    data: np.ndarray
        Out put from basic_slidind_window()

    n_clusters: int
    """
    # vectorize mats
    vectorized_data = vectorize_matrices(data)
    
    # k-means
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(vectorized_data)
    cluster_centers = kmeans.cluster_centers_
    
    # Devectorize centroids
    n_rois = data.shape[1]
    cluster_centers_devectorized = devectorize_centers(cluster_centers, n_rois)
    
    return labels, cluster_centers_devectorized

def momentcorr(data: "list[np.ndarray]", width:int, 
               gaussian:dict=None) -> np.ndarray:
    
    """Calculate moment-by-moment functional connectivity with timecorr
    
    Parameters
    ---------
    data: list[np.ndarray]
        Data

    width: int
        How wide should the window be?
    
    gaussian: dict, default = ={'name': 'Gaussian', 'weights': tc.gaussian_weights, 'params': {'var': width}}
        Parameters to give to timecorr. 
        Recommended to use default

    Return
    ------
    struc_vec_corrs
        Vectorized data
    """

    # check gaussian
    if gaussian is None:
        gaussian = {'name': 'Gaussian', 'weights': tc.gaussian_weights, 'params': {'var': width}}

    
    # compute 
    struc_vec_corrs = tc.timecorr(data, 
                              weights_function=gaussian['weights'], 
                              weights_params=gaussian['params'], 
                              combine=tc.corrmean_combine)
    
    return struc_vec_corrs
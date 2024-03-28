from pydfc import data_loader
import numpy as np
from pydfc.dfc_methods import SLIDING_WINDOW
from pydfc import time_series


def swfc_combine(BOLD_multi_object:time_series.TIME_SERIES, subids:"list[str]") -> "dict[np.ndarray]":
    """Combine SWFC
    
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
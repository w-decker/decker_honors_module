from nilearn.maskers import NiftiLabelsMasker
from nilearn import datasets
import matplotlib.pyplot as plt 
import numpy as np
import nibabel as nib

def nilearn_mask_data(atlas: str, file: str, report:bool=False, plot:bool=False):
    """Mask an fMRI data using Harvard-Oxford probabilistic atlas
    
    Parameters
    ----------
    atlas: str
        See https://nilearn.github.io/dev/modules/generated/nilearn.datasets.fetch_atlas_harvard_oxford.html

    file: str
        Absolute path to functional data

    report: bool, default = False
        Display masking report

    plot: bool, default = False
        Show data
    """

    # get atlas
    atlas = datasets.fetch_atlas_harvard_oxford(atlas)

    # Instantiate the masker with label image and label values
    masker = NiftiLabelsMasker(
        atlas.maps,
        labels=atlas.labels,
        standardize="zscore_sample",
    )

    # output report for people to view
    masker.fit(file)
    if report is True:
        masker.generate_report()

    # mask to data
    data = masker.transform(file)

    # plot for people to view
    if plot is True:
        plt.imshow(data.T, aspect='auto')

    return data, atlas.labels

def stack_data(data: "list[str]", group=False, **kwargs) -> "dict[str, np.ndarray]":
    """Stack data
    
    Parameters
    ----------
    data: list[str]
        List of file paths
        
    group: bool, default = False
        Group data by condition?
        
    Optional Keyword Parameters
    ---------------------------
    cond: dict
        Dictionary output by decker.utils.io.parse_pdb()
        
    Return
    ------
    stacked: dict[str, np.ndarray]
        Stacked data as dictionary
        Keys are either conditions or just "data"
        """
    
    # don't group data by condition
    if group is False:
        d = []

        # load data and append it to list
        for i in data:
            _d = nib.load(i).get_fdata()
            d.append(_d)

        # stack and return
        d = np.stack(d)
        stacked = {"data":d}

    # group data by condition
    elif group is not False:

        # get condition table 
        cond_table = kwargs.get("cond")
        d = {}

        for condition, subject_ids in cond_table.items():
            condition_data = []
            for sub_id in subject_ids:

                # Construct file path based on subject ID
                file_path = [f for f in data if sub_id in f]
                
                if file_path:  # Check if file exists
                    condition_data.append(nib.load(file_path[0]).get_fdata())
            d[condition] = np.stack(condition_data)
            
        stacked = d

    return stacked
    
    

    
    

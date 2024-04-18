from nilearn.maskers import NiftiLabelsMasker
from nilearn import datasets
import matplotlib.pyplot as plt 
import numpy as np
import nibabel as nib
from decker.utils.io.io import get_subid
import os
from pathlib import Path
from nilearn.interfaces.fmriprep import load_confounds_strategy


def nilearn_mask_single_data(atlas: str, file: str, report:bool=False, plot:bool=False, confounds:bool=False):
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
        report = masker.generate_report()
        report
    else: 
        report = None

    # mask to data
    if confounds:

        # get confounds
        confounds, _ = load_confounds_strategy(
        file, denoise_strategy="simple", motion="basic")

        # mask
        data = masker.fit_transform(file, confounds=confounds)
    else:
        
        # mask
        data = masker.fit(file)

    # plot for people to view
    if plot is True:
        plt.figure(figsize=(6, 3))
        plt.imshow(data.T, aspect='auto')
        plt.ylabel("Regions")
        plt.xlabel("TRs")
        plt.title(f'{get_subid(file)}')

    return data, atlas.labels, report

def nilearn_mask_group_by_condition(cond, atlas:str, files:"list[str]", confounds:bool=False) -> "dict[str, np.ndarray]":
    """Mask all data in a directory using Harvard-Oxford probabilistic atlas
    
    Parameters
    ----------
    atlas: str
        See https://nilearn.github.io/dev/modules/generated/nilearn.datasets.fetch_atlas_harvard_oxford.html

    files: list[str]
        output from decker.utils.io.io.BIDSio.get_func()

    cond: dict
        Dictionary output by decker.utils.io.parse_pdb()
        """
    
    # empty dict for grouped masked data
    d = {}

    print('Beginning masking procedure...')
    print(f'------------------------------\n')

    # loop over keys and values in condition table
    for condition, subject_ids in cond.items():
        condition_data = []

        # loop over values in condition table
        for sub_id in subject_ids:
            file_path = [f for f in files if sub_id in f]

            # parse and and store
            if file_path:
                roidat, _, _ = nilearn_mask_single_data(atlas=atlas, file=file_path[0], confounds=confounds)
                condition_data.append(roidat)
                print(f'Masking: {get_subid(file_path[0])} and adding to group: {condition} \n')

            # put data in back in dictionary and stack it
            d[condition] = np.stack(condition_data)

    print(f'------------------------------\n')
    print('Done!')

    return d

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

def mask_anat(data: str, mask: str, output_dir: str) -> "dict[np.ndarray, nib.nifti1.Nifti1Image]":
    """Mask a subject's anatomical scan with subject's T1w mask
    
    Parameters
    ----------
    data: str
        Path to anatomical data in nii.gz format
        
    mask: str
        Path to mask
        
    output_dir: str
        Path (with intended filename) to output directory to save the masked data
        
    Return
    ------
    data: dict[np.ndarray, nib.nifti1.Nifti1Image]
        Returns a dict with the data as a numpy.ndarray and Nifti image"""

    # assert a few things
    assert data.endswith(".nii.gz")
    assert data.endswith(".nii.gz")
    assert os.path.exists(output_dir)

    # load in the data
    _data = nib.load(data).get_fdata()
    _mask = nib.load(mask).get_fdata()

    # mask data
    mask_data = np.multiply(_data, _mask)  

    # save
    out = nib.nifti1.Nifti1Image(mask_data, affine=mask_data.affine, header=mask_data.header)
    nib.save(out, output_dir)

    return {"numpy": mask_data, "Nifti": out}

def export(data, filename:str, output_path:str = Path.cwd(), verbose:bool=False):
    """Export data
    
    Description
    -----------
    Converts data into an a priori specified format (nifti or binary numpy). 
    If you are converting to nifti format, then you may only create a single dataset. If you are converting to binary numpy, then you may export a stacked/grouped data of type np.ndarray or list[np.ndarray], dict[np.ndarray], etc. 

    Parameters
    ----------
    data:
        Can be a dictionary or numpy array of data

    filename: str
        Name to export, with extension ['.nii.gz' or '.npy']

    output_path: str, default = pathlib.Path.cwd()
        Where do you want to save the files?

    verbose: bool
        Display where files are output
    """

    # just to make sure
    assert filename.endswith('.nii.gz') or filename.endswith('.npy')

    if filename.endswith('.nii.gz'):
        img = nib.nifti1.Nifti1Image(data, affine=None)
        nib.save(img, os.path.join(output_path, filename))
        
        # assure
        if verbose is True:
            print(f'{filename} saved to {output_path} in nifti format.')

    elif filename.endswith('.npy'):
        np.save(os.path.join(output_path, filename), data)

        # assure
        if verbose is True:
            print(f'{filename} saved to {output_path} in numpy binary format.')

    
def vectorize_matrices(data:np.ndarray) -> np.ndarray:
    """Vectorize three dimensional data"""

    # Make matrix a vector
    return data.reshape(data.shape[0], -1)

def devectorize_centers(cluster_centers, n_rois):
    """Devectorize clusters from k-means algorithm"""

    # Reshape cluster centers back into correlation matrices
    return cluster_centers.reshape(cluster_centers.shape[0], n_rois, n_rois)

def corr2cov(x:np.ndarray, axis:int=0) -> np.ndarray:
    """Convert correlation matrix to covariance matrix"""

    sigma = np.std(x, axis=axis)
    covmat = x @ np.outer(sigma, sigma)

    return covmat


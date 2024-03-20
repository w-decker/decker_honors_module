from bids import BIDSLayout
import bids.layout.layout
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import urllib.request
import shutil
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)  # Set the logging level as desired

# Attempt to import deepdish
try:
    import deepdish as dd
    DEEPDISH_IMPORTED = True
except ModuleNotFoundError:
    logging.warning("deepdish is not installed. Some functionalities may not be available.")


def download_sherlock(out_dir: Path):
    """Download sherlock dataset"""
    out_dir.mkdir(exist_ok=True)

    # sherlock data URLs
    url_sherlock = 'https://ndownloader.figshare.com/files/9017983'
    url_AG_movie = 'https://ndownloader.figshare.com/files/9055612'

    # download data
    with urllib.request.urlopen(url_sherlock) as response, open(out_dir / 'sherlock.h5', 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

    with urllib.request.urlopen(url_AG_movie) as response, open(out_dir / 'AG_movie_1recall.h5', 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

def load_sherlock(path: Path, dataset: str) -> np.ndarray:
    """Load sherlock data
    
    Parameters
    ----------
    path: str or os.path.object
        Path to data directory

    dataset: str
        'sherlock' or 'AG_movie_1_recall'

    Return
    -------
    D: numpy.ndarray
    """

    D = dd.io.load(path / f"{dataset}.h5")

    print(f'Access variables like \n D["BOLD"] \n D["coords"] \n D["human_bounds"]')

    return D

def parse_pdb(path:str, subid_col: str, cond_col: str) -> dict:
    """Read participant database and parse by condition
    
    Parameters
    ----------
    path: str
        Path to wide formatted database

    subid_col: str
        Column name for subject identifiers

    cond_col: str
        Column name for condition identifiers
        
    Returns
    -------
    grouped: dict
        Dictionary containg key-value pairs in which keys are condition factors and values are subject IDs"""

    # get participant database
    if path.endswith(".xlsx"):
        df = pd.read_excel(path, header=0)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)

    # pivot with only subid and condition column
    _df = df.pivot(columns=cond_col, values=subid_col)

    # get conditions and create new dataframe
    _, c = pd.factorize(df[cond_col])

    # create empty dictionary
    grouped = {}
    for i in c.to_list():
        grouped[i] = list(_df[i].dropna().unique())
    
    return grouped

def get_subid(path: str) -> str:
    """Extract the subject ID from a full BIDS file path to a func file in derivatives folder
    
    Parameters
    ----------
    path: str
        Full path to file following bids naming and organization conventions"""

    subid = path.split("/")[-3].split("-")[1]

    return subid

@dataclass
class BIDSio(object):
    """Parent class for handeling BIDS formatted data
    
    Parameters
    ----------
    bids_path: str, default = current directory"""

    bids_path: str = Path.cwd()

    def init(self) -> bids.layout.layout.BIDSLayout:
        """Generate BIDS layout
        
        Return
        ------
        BIDS: bids.layout.layout.BIDSLayout"""

        self.BIDS = BIDSLayout(self.bids_path, derivatives=True)

        return self
    
    def info(self):
        """Print dataset info to console"""

        print(f'Subjects: \n\t {self.BIDS.get_subjects()}')
        print()
        print(f'Total subjects: \n\t {len(self.BIDS.get_subjects())}')
        print()
        print(f'Tasks: \n\t {self.BIDS.get_tasks()}')
        print()
        print(f'Total tasks: \n\t {len(self.BIDS.get_tasks())}')
        print()

    def get_signle_func(self, suffix: str, **kwargs):
        """Acquires single functional derivatives datafile.
        Expects that you have processed with fMRIPrep with --level full as it will search for ...desc-preproc_bold.nii.gz

        Parameters
        ----------
        suffix: str
        
        Keyword Parameters
        -----------------
        task: str, default = bids.layout.layout.BIDSLayout.get_tasks()[0]

        subid: str, default = bids.layout.layout.BIDSLayout.get_subjects()[0]
            Specify the subject ID. Default is first subject
        """

        if kwargs.get('subid') is None:
            subid = self.BIDS.get_subjects()[0]
        else:
            subid = kwargs.get('subid')
        
        if kwargs.get('task') is None:
            task = self.BIDS.get_tasks()[0]
        else:
            task = kwargs.get('task')

        D = self.BIDS.get(subject=subid, scope='derivatives', task=task, extension='nii.gz', suffix=suffix, return_type='filename')

        return D

    def get_func(self, space: str, suffix: str, **kwargs):
        """Acquires all functional derivatives datafiles.
        Expects that you have processed with fMRIPrep with --level full as it will search for ...desc-preproc_bold.nii.gz

        Parameters
        ----------
        suffix: str
        
        Keyword Parameters
        -----------------
        task: str, default = bids.layout.layout.BIDSLayout.get_tasks()[0]
        """

        if kwargs.get('task') is None:
            task = self.BIDS.get_tasks()[0]
        else:
            task = kwargs.get('task')

        if kwargs.get("space") is None:
            space = "MNI152NLin2009cAsym"
        else:
            space = kwargs.get("space")

        D = self.BIDS.get(task=task, space=space, suffix="bold", extension=".nii.gz", return_type='filename')

        return D
from bids import BIDSLayout
import bids.layout.layout
import os
from pathlib import Path
from dataclasses import dataclass

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

    def get_func(self, suffix: str, **kwargs):
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

        D = self.BIDS.get(scope='derivatives', extension='nii.gz', task=task, suffix=suffix, return_type='filename')

        return D
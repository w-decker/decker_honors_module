# decker_honors_module

Modular code for Will Decker's Honors Thesis (2024).

Real examples of deployment can be be found in the [analysis capsule](https://github.com/w-decker/Honors-Thesis) for this project.

# Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)

   2a. [`decker.utils.io.io`](#deckerutilsioio)

   2b. [`decker.analysis.utils.utils`](#deckeranalysisutilsutils)

   2c. [`decker.analysis.behavioral.scoring`](#deckeranalysisbehvarioralscoring)

   2d. [`decker.analysys.eventseg.eventseg`](#deckeranalysiseventsegeventseg)

3. [Contributing](#contributing)

# Installation

It is recommended that you clone the repository

```bash
git clone https://github.com/w-decker/decker_honors_module.git
```

And install the module

```bash
pip install .
```

To update the module

```bash
git pull origin main
pip install . --upgrade
```

# Usage

This code is broken down into a few submodules

```bash
decker
├── analysis
│   ├── behavioral
│   ├── eventseg
│   ├── swfc
│   └── utils
└── utils
    └── io
```

The following usage description will be divided by each submodule.

## `decker.utils.io.io`

I/O functions and classes for handeling neuroimaging data with an emphasis on [BIDS](https://bids.neuroimaging.io/) formatted data.

### `download_sherlock()`

Download the sherlock dataset and more, referenced by [Chen et al., 2016](https://www.biorxiv.org/content/10.1101/035931v1).

```py
from pathlib import Path
path = "/path/to/desired/directory"
download_sherlock(out_dir=Path(path))
```

### `load_sherlock()`

Load sherlock dataset and more into Python environment.

```py
from pathlib import Path
path = "/path/to/desired/directory"
data = load_sherlock(path=Path(path), dataset='sherlock')
```

### `parse_pdb()`

Parse a given database (e.g., participant database) by identifying elements belonging to a given condition. Most valuable for subject X condition transpose.

```py
path = "path/to/database.xlsx" # can also handle .csv
grouped_by_variable = parse_pdb(path=path, subid_col="Subject ID", cond_col="Condition")
```

> This function recruits `pandas` functionality. Specifying `subid_col` and `cond_col` require exact column names. Returns a dictionary with key-value pairs of condition-element(s).

### `get_subid()`

Identify the subject ID from a full path BIDS formatted .nii.gz file.

```py
path = "path/to/bids/file.nii.gz"
subid = get_subid(path)
```

### `BIDSio`

Class for querying BIDS formatted datasets, including derivatives.

```py
path = "/path/to/root/bids/directory"
bids = BIDSio(bids_path=path)
```

#### `BIDSio.init()`

Method to bind BIDS root directory with `BIDSio`.

```py
bids.init()
```

#### `BIDSio.info()`

Method to display dataset information

```py
bids.info()
```

#### `BIDSio.get_single_func()`

Method to acquire a single functional derivative.

```py
bids.get_single_func(suffic="preproc", task="rest", subid="001")
```

> This method contains defaults arguments. See docstring for more information and optional keyword parameters.

#### `BIDSio.get_func()`

Method to acquire entire functional derivatives.

```py
bids.get_func(suffix="preproc", task="rest")
```

> This method contains defaults arguments. See docstring for more information and optional keyword parameters.

## `decker.analysis.utils.utils`

Versatile utility functions to specifically streamline/aid analyses.

### `nilearn_mask_single_data()`

Extract ROIs from the Harvard-Oxford probabilistic atlases from a single .nii.gz file.

```py
atlas = "cortl-maxprob-thr0-2mm"
file = "path/to/nii.gz"
data, labels, report = nilearn_mask_single_data(atlas=atlas, file=file)
```

> This function contains defaults arguments. See docstring for more information and optional keyword parameters.

### `nilearn_mask_group_by_condition()`

Extract ROIs from the Harvard-Oxford probabilistic atlases from a group of .nii.gz files.

```py
# pps = ... # load in participant data created by decker.utils.io.io.parse_pdb()
# data = ... # load in list of data created by decker.utils.io.io.BIDSio
data = nilearn_mask_group_by_condition(cond=pps, atlas="cort-maxprob-thr25-1mm", files=data)
```

### `stack_data()`

Stack multiple .nii.gz files into `numpy.ndarray`s. You can also parse files based on the output from `decker.utils.io.parse_pdb()`.

```py
data = ["path/to/nii.gz", "path/to/nii.gz", "path/to/nii.gz"]
stacked = stack_data(data=data)
```

> This function contains defaults arguments. See docstring for more information and optional keyword parameters.

### `mask_anat()`

Mask anatomical data using fMRIPrep outputs. Returns a dictionary with numpy (`.npz`) and nifti (`.nii.gz`) data. Saves data to specified directory.

```py
data = ... # path to datafile
mask = ... # path to mask file
output_dir = ... # path to output directorry
masked_anat = mask_anat(data=data, mask=mask, output_dir=output_dir)
```

### `export()`

Converts data to specified data type (numpy, `.npy`; nifti, `.nii.gz`) and saves to directory.

```py
data = ... # path to datafile
filename = 'mydata.nii.gz'
export(data=data, filename=filename)
```

> This function contains defaults arguments. See docstring for more information and optional keyword parameters.

## `decker.analysis.behvarioral.scoring`

Classes and methods for analyzing behavioral data from Will Decker's honors thesis project. Functionality may not transfer to other applications without modifications.

### `Data`

Class for loading behavioral data

```py
path = "/path/to/data"
data = Data(path=path)
```

#### `Data.parse_files()`

Method to extract only relevant files and data.

```py
data.parse_files()
```

#### `Data.rm_subs()`

Method to remove a subject's data

```py
subid = ("sub-001")
data.rm_subs(subid=subid)
```

> Warning: This action cannot be undone.

#### `Data.score()`

Method to score behavioral data.

```py
data.score()
```

#### `Data.indiv_score()`

Method to view results of a single subject

```py
subid = "sub-001"
data.indiv_score()
```

#### `Data.cat_score()`

Method to calculate category level scores (scores based on trial type).

```py
data.cat_score()
```

### `Stats`

Class to calcualate bare-metal statistical test on behvaioral data.

```py
stats = Stats(scores=data.scores) # requires Data.scores, an attributed of the Data class
```

#### `Stats.compute()`

Method for computing desired statistical tests

```py
stats.compute(test='ind')
```

> See docstring for more information and optional keyword parameters.

## `decker.analysis.eventseg.eventseg`

Assortment of functions concatenating the _leave-one-out nested cross validation_ computations.

### `tuneK()`

Tune $k$ parameters (number of estimated event segments). Returns the $k$ value with the largest log-likelihood of model fit.

```py
train_data = ... # numpy.ndarray of training data
train_data = ... # numpy.ndarray of training data
k_vals = np.arange(3, 10, 1)
bestK, LL = tuneK(train_data=train_data, test_data=test_data, k_vals=k_vals)
```

### `innerLoop()`

Run inner loop of nested CV over $n$ folds and returns best $k$ after all folds.

```py
n_splits = 4
idx = ... # numpy.ndarray of ints corresponding to each subject (e.g., np.arange(0, 10, 1))
outer_train_idx = ... # numpy.ndarray of outer train indices
data = ... # numpy.ndarray of data minus outer left out
k_vals = np.arange(3, 10, 1)

bestK = innerLoop(n_splits=n_splits, idx=idx, outer_train_idx=outer_train_idx, data=data, k_vals=k_vals)
```
### `outerLoop()`

Runs entirety of nested CV. Returns `pd.DataFrame` of $k$ and log-likelihood of final test on left-out data.
```py
idx = ... # numpy.ndarray of ints corresponding to each subject (e.g., np.arange(0, 10, 1))
n_splits = 4
data = ... # numpy.ndarray of data
k_vals = np.arange(3, 10, 1)

results = outerLoop(data=data, idx=idx, n_splits=n_splits, k_vals=k_vals)
```

# Contributing

If you wish to contribute to this code, please submit a pull request to the main branch with adequeate descriptions of the requested/intended changes.

Please email [deckerwill7@gmail.com](mailto:deckerwill7@gmail.com) with any additional questions.

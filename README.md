# decker_honors_module

Modular code for Will Decker's Honors Thesis (2024)

# Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [`decker.utils.io.io`](#deckerutilsioio)
4. [`decker.analysis.utils.utils`](#deckeranalysisutilsutils)
5. [`decker.analysis.behavioral.scoring`](#deckeranalysisbehvarioralscoring)

# Installation

It is recommended that you clone the repository

```bash
git clone https://github.com/w-decker/decker_honors_module.git
```

And install the module

```bash
pip install .
```

# Usage

This code is broken down into a few submodules

```bash
.
├── analysis
│   ├── behavioral
│   ├── eventseg
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

### `nilearn_mask_data()`

Extract ROIs from the Harvard-Oxford probabilistic atlases.

```py
atlas = "cortl-maxprob-thr0-2mm"
file = "path/to/nii.gz"
data, labels = nilearn_mask_data(atlas=atlas, file=file)
```

> This function contains defaults arguments. See docstring for more information and optional keyword parameters.

### `stack_data()`

Stack multiple .nii.gz files into `numpy.ndarray`s. You can also parse files based on the output from `decker.utils.io.parse_pdb()`.

```py
data = ["path/to/nii.gz", "path/to/nii.gz", "path/to/nii.gz"]
stacked = stack_data(data=data)
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

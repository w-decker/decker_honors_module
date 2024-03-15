import urllib.request
import shutil
import numpy as np
import deepdish as dd
from pathlib import Path

def download_data(out_dir: Path):
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



def load_data(path: Path, dataset: str) -> np.ndarray:
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
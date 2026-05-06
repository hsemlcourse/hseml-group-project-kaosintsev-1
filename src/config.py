from pathlib import Path


def project_rootect_root():
    here = Path(__file__).resolve().parent.parent
    if (here / "data" / "raw" / "train.csv").exists():
        return here
    cwd = Path.cwd()
    if (cwd / "data" / "raw" / "train.csv").exists():
        return cwd
    if (cwd.parent / "data" / "raw" / "train.csv").exists():
        return cwd.parent
    if (here / "train.csv").exists():
        return here
    if (cwd / "train.csv").exists():
        return cwd
    return here


RANDOM_STATE = 42
N_CV_SPLITS = 5
MISSING_TOKEN = "missing"

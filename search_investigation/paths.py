from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "SEARCH"
CONFIG_DIR = REPO_ROOT / "configs"
OUTPUTS_DIR = REPO_ROOT / "outputs"
MANIFESTS_DIR = OUTPUTS_DIR / "manifests"
SMOKE_TESTS_DIR = OUTPUTS_DIR / "smoke_tests"
LOGS_DIR = OUTPUTS_DIR / "logs"
SLURM_DIR = LOGS_DIR / "slurm"
DERIVED_DIR = OUTPUTS_DIR / "derived"
ARRAY_DERIVED_DIR = DERIVED_DIR / "arrays"


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_project_dirs():
    for path in [
        OUTPUTS_DIR,
        MANIFESTS_DIR,
        SMOKE_TESTS_DIR,
        LOGS_DIR,
        SLURM_DIR,
        DERIVED_DIR,
        ARRAY_DERIVED_DIR,
        REPO_ROOT / "docs",
        REPO_ROOT / "scripts" / "slurm",
    ]:
        ensure_dir(path)

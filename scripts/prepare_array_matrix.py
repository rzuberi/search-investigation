#!/usr/bin/env python3
from __future__ import print_function

import argparse
import subprocess
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search_investigation.data import (
    DATA_DIR,
    ICOGS_LOOKUP_XLSX,
    ONCO_LOOKUP_XLSX,
    _read_lookup,
    find_plink_binary,
)
from search_investigation.paths import ARRAY_DERIVED_DIR, ensure_dir, ensure_project_dirs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["icogs", "oncoarray"], required=True)
    parser.add_argument("--plink-binary")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_project_dirs()
    plink_binary = args.plink_binary or find_plink_binary()
    if not plink_binary:
        raise RuntimeError("plink or plink2 is required to prepare array matrices.")

    if args.modality == "icogs":
        bfile = DATA_DIR / "search_zuberi_icogs"
        lookup = _read_lookup(ICOGS_LOOKUP_XLSX)
    else:
        bfile = DATA_DIR / "search_zuberi_oncoarray"
        lookup = _read_lookup(ONCO_LOOKUP_XLSX)

    outdir = ensure_dir(ARRAY_DERIVED_DIR / args.modality)
    raw_prefix = outdir / args.modality
    raw_matrix = Path(str(raw_prefix) + ".raw")
    final_matrix = ARRAY_DERIVED_DIR / "{0}_matrix.tsv.gz".format(args.modality)

    if args.force or not raw_matrix.exists():
        command = [plink_binary, "--bfile", str(bfile), "--recode", "A", "--out", str(raw_prefix)]
        print("Running:", " ".join(command))
        subprocess.check_call(command)

    frame = pd.read_csv(str(raw_matrix), sep=r"\s+")
    frame = frame.rename(columns={"IID": "assay_id"})
    frame["assay_id"] = frame["assay_id"].astype(str)
    frame = frame.merge(lookup, on="assay_id", how="left")
    frame = frame[frame["RZ_ID"].notnull()].copy()
    drop_columns = [column for column in ["FID", "PAT", "MAT", "SEX", "PHENOTYPE", "assay_id"] if column in frame.columns]
    frame = frame.drop(columns=drop_columns)
    frame["RZ_ID"] = frame["RZ_ID"].astype(str)
    frame = frame.groupby("RZ_ID", as_index=False).mean()
    frame.to_csv(str(final_matrix), sep="\t", index=False, compression="gzip")
    print("Saved:", final_matrix)


if __name__ == "__main__":
    main()

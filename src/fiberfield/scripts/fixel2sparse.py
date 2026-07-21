import nibabel as nib
import numpy as np
import argparse
from pathlib import Path
from fiberfield.utils import fixel_to_sparse_format


def main():
    parser = argparse.ArgumentParser(
        description="Convert MRtrix3 fixel format to 4D sparse peaks and associated metrics",
        epilog="Authored by Adam Saunders.",
    )
    parser.add_argument(
        "fixel_dir",
        type=str,
        help="Path to directory where fixel files are located.",
    )
    parser.add_argument(
        "peaks",
        type=str,
        help="Output path to peaks as a NIFTI file (X x Y x Z x N x 3).",
    )
    parser.add_argument(
        "sparse_data_dir",
        type=str,
        help="Output directory for associated sparse data with metrics NIFTI file (X x Y x Z x N) or (X x Y x Z x N x P) where P is the number of parameters.",
    )
    args = parser.parse_args()

    sparse_data_files = []
    sparse_data_names = []
    index_file, directions_file = None, None
    for file in args.fixel_dir.glob("*.nii.gz"):
        if file.name == "index.nii.gz":
            index_file = file
        elif file.name == "directions.nii.gz":
            directions_file = file
        else:
            sparse_data_files.append(file)
            sparse_data_names.append(file.name)

    if not index_file or not directions_file:
        raise ValueError(
            "Fixel directory must contain 'index.nii.gz' and 'directions.nii.gz' files."
        )

    if not Path(args.sparse_data_dir).exists():
        Path(args.sparse_data_dir).mkdir(parents=True, exist_ok=True)

    peaks, sparse_data = fixel_to_sparse_format(
        index_file, directions_file, sparse_data_files
    )

    nib.save(
        nib.Nifti1Image(peaks.astype(np.float32), nib.load(directions_file).affine),
        args.peaks,
    )
    for i, d in enumerate(sparse_data):
        nib.save(
            nib.Nifti1Image(d.astype(np.float32), peaks.affine),
            args.sparse_data_dir / f"{sparse_data_names[i]}",
        )


if __name__ == "__main__":
    main()

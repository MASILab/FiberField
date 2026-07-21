import nibabel as nib
import numpy as np
import argparse
from pathlib import Path
from diffusion_neural_field.utils import sparse_to_fixel_format


def main():
    parser = argparse.ArgumentParser(
        description="Convert 4D sparse peaks and associated metrics to MRtrix3 fixel format",
        epilog="Authored by Adam Saunders.",
    )
    parser.add_argument(
        "peaks",
        type=str,
        help="Path to peaks as a NIFTI file (X x Y x Z x N x 3) or (X x Y x Z x N*3).",
    )
    parser.add_argument(
        "sparse_data",
        nargs="*",
        type=str,
        help="Path to assocated sparse data with metrics NIFTI file (X x Y x Z x N) or (X x Y x Z x N x P) where P is the number of parameters.",
    )
    parser.add_argument(
        "fixel_dir",
        type=str,
        help="Path to directory where fixel files will be saved.",
    )
    args = parser.parse_args()

    peaks = nib.load(args.peaks)
    data = peaks.get_fdata()

    fixel_dir = Path(args.fixel_dir)
    fixel_dir.mkdir(parents=True, exist_ok=True)

    if not (data.ndim == 5 and data.shape[4] == 3) and not (
        data.ndim == 4 and data.shape[3] % 3 == 0
    ):
        raise ValueError(
            f"Input NIFTI file must be 5D with shape (X, Y, Z, N, 3) or 4D with shape (X, Y, Z, N*3), but got shape {data.shape}."
        )
    if data.ndim == 4:
        data = data.reshape(
            data.shape[0], data.shape[1], data.shape[2], data.shape[3] // 3, 3
        )

    sparse_data = None
    sparse_data_names = None
    if args.sparse_data:
        sparse_data = [nib.load(path).get_fdata() for path in args.sparse_data]
        sparse_data_names = [Path(path).name for path in args.sparse_data]
        for i, d in enumerate(sparse_data):
            if d.ndim not in [4, 5]:
                raise ValueError(
                    f"Sparse data NIFTI file {args.sparse_data[i]} must be 4D or 5D, but got shape {d.shape}."
                )
            if d.shape[0:3] != data.shape[0:3]:
                raise ValueError(
                    f"Sparse data NIFTI file {args.sparse_data[i]} must have the same spatial dimensions as peaks, but got shape {d.shape}."
                )
            if d.shape[3] != data.shape[3]:
                raise ValueError(
                    f"Sparse data NIFTI file {args.sparse_data[i]} must have the same number of peaks as peaks, but got shape {d.shape}."
                )

    index, directions, fixel_data = sparse_to_fixel_format(data, sparse_data, eps=1e-8)
    nib.save(
        nib.Nifti2Image(index.astype(np.int32), peaks.affine),
        fixel_dir / "index.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(directions.astype(np.float32), peaks.affine),
        fixel_dir / "directions.nii.gz",
    )
    for i, d in enumerate(fixel_data):
        nib.save(
            nib.Nifti1Image(d.astype(np.float32), peaks.affine),
            fixel_dir / f"{sparse_data_names[i]}",
        )


if __name__ == "__main__":
    main()

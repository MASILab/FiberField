import argparse
import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Convert fixel AFD to per-fixel volume fractions."
    )
    parser.add_argument("fixel_dir", help="MRtrix fixel directory")
    parser.add_argument(
        "--input",
        "-i",
        default="afd.nii.gz",
        help="Input fixel metric (default: afd.nii.gz)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="volume_fraction.nii.gz",
        help="Output fixel metric",
    )
    args = parser.parse_args()

    index_img = nib.load(f"{args.fixel_dir}/index.nii.gz")
    afd_img = nib.load(f"{args.fixel_dir}/{args.input}")

    index = np.asarray(index_img.dataobj)
    afd = np.asarray(afd_img.dataobj)

    counts = index[..., 0].astype(int)
    starts = index[..., 1].astype(int)
    volume_fraction = np.zeros_like(afd)

    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            for k in range(counts.shape[2]):
                n = counts[i, j, k]
                if n == 0:
                    continue

                start = starts[i, j, k]
                stop = start + n

                voxel_afd = afd[start:stop, 0, 0]
                total = voxel_afd.sum()

                if total > 0:
                    volume_fraction[start:stop, 0, 0] = voxel_afd / total

    nib.save(
        nib.Nifti2Image(
            volume_fraction,
            affine=afd_img.affine,
            header=afd_img.header,
        ),
        f"{args.fixel_dir}/{args.output}",
    )


if __name__ == "__main__":
    main()

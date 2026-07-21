from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.optimize import linear_sum_assignment
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Match fixels to a reference set based on angular similarity. "
        "Also outputs voxel-wise true positives, false positives, and false negatives "
        "and fixel-wise angular differences.",
        epilog="Authored by Adam Saunders.",
    )
    parser.add_argument("fixel_dir", help="MRtrix fixel directory")
    parser.add_argument("reference_dir", help="Reference fixel directory")
    parser.add_argument("output_dir", help="Output directory for matched fixels")
    parser.add_argument(
        "--fill_value",
        "-f",
        default=0.0,
        help="Fill value for unmatched fixels (default: 0.0). Can be set to NaN.",
    )
    parser.add_argument(
        "--max_angle",
        "-a",
        type=float,
        default=45.0,
        help="Maximum angular difference (in degrees) for matching fixels (default: 45.0)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.fill_value.lower() == "nan":
        args.fill_value = np.nan
    else:
        args.fill_value = float(args.fill_value)

    # Load fixel data
    fixel_dir = Path(args.fixel_dir)
    reference_dir = Path(args.reference_dir)
    index_img = nib.load(f"{fixel_dir}/index.nii.gz")
    directions_img = nib.load(f"{fixel_dir}/directions.nii.gz")

    index_img = nib.as_closest_canonical(index_img)
    directions_img = nib.as_closest_canonical(directions_img)

    index = index_img.get_fdata()
    directions = directions_img.get_fdata()

    # Also load any other *.nii.gz and see if they are fixel metrics or voxel metrics
    fixel_data = []
    for file in fixel_dir.glob("*.nii.gz"):
        if file.name in ["index.nii.gz", "directions.nii.gz"]:
            continue
        metric_img = nib.load(file)
        metric_img = nib.as_closest_canonical(metric_img)
        metric_data = metric_img.get_fdata()
        # n x p x 1 is fixel data, i x j x k or i x j x k x p is voxel data
        is_voxel_data = (
            metric_data.ndim in [3, 4] and metric_data.shape[0:3] == index.shape[0:3]
        )
        # if is_voxel_data:
        #     nib.save(
        #         nib.Nifti2Image(metric_data, metric_img.affine, metric_img.header),
        #         output_dir / file.name,
        #     )
        is_fixel_data = (
            metric_data.ndim in [3] and metric_data.shape[0] == directions.shape[0]
        )
        if is_fixel_data:
            fixel_data.append((file.name, metric_data))

    # Load reference fixel data
    ref_index_img = nib.load(f"{reference_dir}/index.nii.gz")
    ref_directions_img = nib.load(f"{reference_dir}/directions.nii.gz")
    ref_index_img = nib.as_closest_canonical(ref_index_img)
    ref_directions_img = nib.as_closest_canonical(ref_directions_img)
    ref_index = ref_index_img.get_fdata()
    ref_directions = ref_directions_img.get_fdata()

    count = index[..., 0].astype(int)
    start = index[..., 1].astype(int)
    ref_count = ref_index[..., 0].astype(int)
    ref_start = ref_index[..., 1].astype(int)

    occupied_voxels = np.argwhere((count > 0) | (ref_count > 0))
    mapping = np.full(ref_directions.shape[0], -1, dtype=int)
    true_positives = np.zeros(count.shape, dtype=int)
    false_positives = np.zeros(count.shape, dtype=int)
    false_negatives = np.zeros(count.shape, dtype=int)
    angular_difference = np.full((ref_directions.shape[0], 1, 1), np.nan, dtype=float)
    for voxel in occupied_voxels:
        x, y, z = voxel
        if count[x, y, z] == 0:
            false_negatives[x, y, z] = ref_count[x, y, z]
            continue
        if ref_count[x, y, z] == 0:
            false_positives[x, y, z] = count[x, y, z]
            continue

        # Load peak directions for the current voxel
        voxel_directions = directions[
            start[x, y, z] : start[x, y, z] + count[x, y, z],
            :,
            0,
        ]
        voxel_ref_directions = ref_directions[
            ref_start[x, y, z] : ref_start[x, y, z] + ref_count[x, y, z],
            :,
            0,
        ]

        # Compute the cost matrix based on angular similarity
        cos_angles = np.abs(voxel_directions @ voxel_ref_directions.T)
        cost_matrix = np.arccos(np.clip(cos_angles, -1.0, 1.0))  # Angular distance

        # Set costs above the max_angle threshold to a high value to prevent matching
        max_angle_rad = np.deg2rad(args.max_angle)

        # Perform matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Fill the output directions based on the matching
        n_matches = 0
        for r, c in zip(row_ind, col_ind):
            angle = cost_matrix[r, c]
            if angle > max_angle_rad:
                continue
            mapping[ref_start[x, y, z] + c] = start[x, y, z] + r

            angular_difference[ref_start[x, y, z] + c, 0, 0] = np.rad2deg(angle)
            n_matches += 1

        true_positives[x, y, z] = n_matches
        false_positives[x, y, z] = count[x, y, z] - n_matches
        false_negatives[x, y, z] = ref_count[x, y, z] - n_matches

    print(f"Matched: {(mapping >= 0).sum()} / {len(mapping)}")

    # Re-map the fixel data to the reference fixel space
    for name, data in fixel_data:
        remapped_data = np.full(
            (ref_directions.shape[0],) + data.shape[1:],
            args.fill_value,
            dtype=data.dtype,
        )
        for voxel in occupied_voxels:
            x, y, z = voxel
            voxel_mapping = mapping[
                ref_start[x, y, z] : ref_start[x, y, z] + ref_count[x, y, z]
            ]
            out = np.full(
                (ref_count[x, y, z],) + data.shape[1:],
                args.fill_value,
                dtype=data.dtype,
            )
            valid = voxel_mapping >= 0
            out[valid] = data[voxel_mapping[valid]]
            remapped_data[
                ref_start[x, y, z] : ref_start[x, y, z] + ref_count[x, y, z]
            ] = out

        remapped_data = remapped_data.reshape(ref_directions.shape[0], 1, -1)
        nib.save(
            nib.Nifti2Image(
                remapped_data, index_img.affine, index_img.header, dtype=np.float32
            ),
            output_dir / name,
        )

    # Copy reference index/directions to output directory
    nib.save(
        nib.Nifti2Image(ref_index.astype(np.int32), index_img.affine),
        output_dir / "index.nii.gz",
    )
    nib.save(
        nib.Nifti2Image(ref_directions.astype(np.float32), directions_img.affine),
        output_dir / "directions.nii.gz",
    )

    nib.save(
        nib.Nifti2Image(true_positives, ref_index_img.affine, ref_index_img.header),
        output_dir / "true_positives.nii.gz",
    )

    nib.save(
        nib.Nifti2Image(false_positives, ref_index_img.affine, ref_index_img.header),
        output_dir / "false_positives.nii.gz",
    )

    nib.save(
        nib.Nifti2Image(false_negatives, ref_index_img.affine, ref_index_img.header),
        output_dir / "false_negatives.nii.gz",
    )

    nib.save(
        nib.Nifti2Image(
            angular_difference, ref_directions_img.affine, ref_directions_img.header
        ),
        output_dir / "angular_difference.nii.gz",
    )


if __name__ == "__main__":
    main()

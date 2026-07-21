import nibabel as nib
import numpy as np
import argparse
from pathlib import Path
from fiberfield.utils import visualize_fixels


def main():
    parser = argparse.ArgumentParser(
        description="Visualize fixels from MRtrix3 fixel format files.",
        epilog="Authored by Adam Saunders.",
    )
    parser.add_argument(
        "fixel_dir",
        type=str,
        help="Path to directory containing fixel files. Must have index.nii.gz and directions.nii.gz, and optionally other fixel data files.",
    )
    parser.add_argument(
        "bg_img",
        type=str,
        help="Path to background image NIFTI file.",
    )
    parser.add_argument(
        "--fixel_data",
        type=str,
        required=False,
        default=None,
        help="Optional name of fixel data to visualize, relative to fixel_dir.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=False,
        default=None,
        help="Optional path to save the visualization. Otherwise, the visualization will be displayed interactively.",
    )
    parser.add_argument(
        "--plot_slice",
        nargs="+",
        type=int,
        required=False,
        default=None,
        help="Optional slice number to plot or tuple of 3 coordinates for center.",
    )
    parser.add_argument(
        "--axis",
        type=int,
        required=False,
        default=2,
        help="Optional axis along which to plot slices, where 0 is sagittal, 1 is coronal, and 2 is axial. By default, 2.",
    )
    parser.add_argument(
        "--fixel_cmap",
        type=str,
        required=False,
        default="viridis",
        help="Colormap to use for visualizing fixel data.",
    )
    parser.add_argument(
        "--fixel_value_range",
        nargs=2,
        type=float,
        required=False,
        default=None,
        help="Optional value range for visualizing fixel data.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        required=False,
        default=0.9,
        help="Optional scale for visualizing fixel data, by default 0.9.",
    )
    parser.add_argument(
        "--linewidth",
        type=int,
        required=False,
        default=3,
        help="Optional line width for visualizing fixel data, by default 3.",
    )
    parser.add_argument(
        "--size",
        nargs=2,
        type=int,
        required=False,
        default=(600, 400),
        help="Optional size for window, by default (600, 400).",
    )
    parser.add_argument(
        "--no-colorbar",
        action="store_true",
        help="Optional flag to disable the colorbar.",
    )
    parser.add_argument(
        "--value_range",
        nargs=2,
        type=float,
        required=False,
        default=None,
        help="Optional value range for visualizing background image.",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        required=False,
        default=1,
        help="Optional zoom level.",
    )
    args = parser.parse_args()

    fixel_dir = Path(args.fixel_dir)
    if (
        not (fixel_dir / "index.nii.gz").exists()
        or not (fixel_dir / "directions.nii.gz").exists()
    ):
        raise ValueError(
            f"Fixel directory {fixel_dir} must contain index.nii.gz and directions.nii.gz."
        )
    args = parser.parse_args()

    fixel_dir = Path(args.fixel_dir)
    if (
        not (fixel_dir / "index.nii.gz").exists()
        or not (fixel_dir / "directions.nii.gz").exists()
    ):
        raise ValueError(
            f"Fixel directory {fixel_dir} must contain index.nii.gz and directions.nii.gz."
        )

    index_nifti = nib.load(fixel_dir / "index.nii.gz")
    directions_nifti = nib.load(fixel_dir / "directions.nii.gz")

    bg_img_nifti = nib.load(args.bg_img)

    index_nifti = nib.as_closest_canonical(index_nifti)
    directions_nifti = nib.as_closest_canonical(directions_nifti)
    bg_img_nifti = nib.as_closest_canonical(bg_img_nifti)

    if args.fixel_data is not None:
        fixel_data_nifti = nib.load(fixel_dir / args.fixel_data)
        fixel_data_nifti = nib.as_closest_canonical(fixel_data_nifti)
    else:
        fixel_data_nifti = None

    visualize_fixels(
        index_nifti=index_nifti,
        directions_nifti=directions_nifti,
        bg_img_nifti=bg_img_nifti,
        fixel_data_nifti=fixel_data_nifti,
        save_path=args.save_path,
        plot_slice=args.plot_slice,
        axis=args.axis,
        fixel_cmap=args.fixel_cmap,
        fixel_value_range=args.fixel_value_range,
        scale=args.scale,
        linewidth=args.linewidth,
        size=args.size,
        colorbar=not args.no_colorbar,
        bg_img_value_range=args.value_range,
        zoom=args.zoom,
    )


if __name__ == "__main__":
    main()

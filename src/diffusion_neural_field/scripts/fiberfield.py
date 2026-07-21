import argparse
from diffusion_neural_field.run.stick import run_stick
from diffusion_neural_field.run.standard_model import run_sm


def main():
    parser = argparse.ArgumentParser(
        description="Run FiberField for modeling fiber properties in diffusion MRI.",
        epilog="Authored by Adam Saunders.",
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to input diffusion MRI NIFTI file."
    )
    parser.add_argument(
        "--bval", required=True, help="Path to b-values file in FSL format."
    )
    parser.add_argument(
        "--bvec", required=True, help="Path to b-vectors file in FSL format."
    )
    parser.add_argument("--mask", required=True, help="Path to brain mask NIFTI file.")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save FiberField outputs. Fixel output will be in output_dir/fixel.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="stick",
        choices=["stick", "sm"],
        help="Fiber model to use, choices are 'stick' or 'sm' for Standard Model (default: stick).",
    )
    parser.add_argument(
        "--max_fibers",
        type=int,
        default=3,
        help="Maximum number of fibers per voxel (default: 3).",
    )
    parser.add_argument(
        "--layer_size",
        type=int,
        default=2048,
        help="Size of each layer in the neural network (default: 2048).",
    )
    parser.add_argument(
        "--num_enc",
        type=int,
        default=5000,
        help="Number of positional encoding frequencies (default: 5000).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=4.0,
        help="Standard deviation for positional encoding (default: 4.0).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.002,
        help="Weight for fiber penalty per number of fibers (default: 0.002).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=500,
        help="Batch size for training (default: 500).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for training (default: 1e-4).",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1000,
        help="Number of epochs for training (default: 1000).",
    )
    parser.add_argument(
        "--plot_slice",
        type=int,
        default=None,
        help="Slice number to plot for validation (default: None for middle slice).",
    )
    parser.add_argument(
        "--plot_axis",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Axis along which to plot slices for validation, where 0 is sagittal, 1 is coronal, and 2 is axial (default: 2).",
    )
    parser.add_argument(
        "--validation_frequency",
        type=int,
        default=10,
        help="Number of epochs between validation (default: 10).",
    )
    parser.add_argument(
        "--validation_patience",
        type=int,
        default=25,
        help="Stop training if validation loss has not improved in this many validation checks (default: 25).",
    )
    parser.add_argument(
        "--min_diffusivity",
        type=float,
        default=1e-4,
        help="At inference, filter out any voxels with diffusivity below this threshold (default: 1e-4).",
    )
    parser.add_argument(
        "--min_f_intra",
        type=float,
        default=1e-4,
        help="At inference, filter out any voxels with intra-axonal volume fraction below this threshold (default: 1e-4).",
    )
    parser.add_argument(
        "--include_freewater",
        action="store_true",
        help="If --model is 'sm' and this flag is set, freewater is included in the model.",
    )
    parser.add_argument(
        "--d_fw",
        type=float,
        default=3.0e-3,
        help="Diffusivity of freewater in mm^2/s (default: 3.0e-3). Only used if --include_freewater is set.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for training and inference (default: None for auto-detection).",
    )

    args = parser.parse_args()

    if args.model == "stick":
        run_stick(
            input_path=args.input,
            bval_path=args.bval,
            bvec_path=args.bvec,
            mask_path=args.mask,
            output_dir=args.output_dir,
            max_fibers=args.max_fibers,
            layer_size=args.layer_size,
            num_enc=args.num_enc,
            sigma=args.sigma,
            alpha=args.alpha,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            plot_slice=args.plot_slice,
            plot_axis=args.plot_axis,
            validation_frequency=args.validation_frequency,
            validation_patience=args.validation_patience,
            min_diffusivity=args.min_diffusivity,
            min_f_intra=args.min_f_intra,
            device=args.device,
        )
    elif args.model == "sm":
        run_sm(
            input_path=args.input,
            bval_path=args.bval,
            bvec_path=args.bvec,
            mask_path=args.mask,
            output_dir=args.output_dir,
            max_fibers=args.max_fibers,
            layer_size=args.layer_size,
            num_enc=args.num_enc,
            sigma=args.sigma,
            alpha=args.alpha,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            plot_slice=args.plot_slice,
            plot_axis=args.plot_axis,
            validation_frequency=args.validation_frequency,
            validation_patience=args.validation_patience,
            min_diffusivity=args.min_diffusivity,
            min_f_intra=args.min_f_intra,
            include_freewater=args.include_freewater,
            d_fw=args.d_fw,
            device=args.device,
        )
    else:
        raise NotImplementedError(f"Model {args.model} not implemented.")


if __name__ == "__main__":
    main()

# FiberField

A neural field for adaptive white matter fiber modeling from diffusion MRI.

## Usage

After cloning this repository, you can install the package in a virtual environment. 
Alternatively, you can run commands directly using `uv run`.

The main command is `fiberfield`, which can be used to train and run inference on diffusion MRI.
See `fiberfield --help` for an explanation of the command line arguments.

For example, to train and run inference using the default settings for the multi-stick 
model used in [1]:

```bash
uv run fiberfield \
    --input "dwi.nii.gz" \
    --bval "dwi.bval" \
    --bvec "dwi.bvec" \
    --mask "mask.nii.gz" \
    --output_dir "fiberfield-stick" \
    --model stick \
    --device cuda:0
```

To estimate the full Standard Model parameters, use:

```bash
uv run fiberfield \
    --input "dwi.nii.gz" \
    --bval "dwi.bval" \
    --bvec "dwi.bvec" \
    --mask "mask.nii.gz" \
    --output_dir "fiberfield-sm" \
    --model sm \
    --device cuda:0
```

The outputs will be in a sparse format in the output directory (e.g., D_intra.nii.gz will
be i x j x k x n where n is the maximum number of fibers). They will also be stored in
MRtrix3 [fixel format](https://mrtrix.readthedocs.io/en/latest/fixel_based_analysis/fixel_directory_format.html).

To visualize the outputs, you can use the fixel visualization tool provided in 
[MRtrix3's mrview](https://www.mrtrix.org). Alternatively, we provide a fixel 
visualization tool called `fixelview`. For example, to visualize the intra-axonal volume
fraction:

```bash
uv run fixelview \  
    "fiberfield-stick/fixel" \
    mask.nii.gz \
    --fixel_data D_intra.nii.gz \
    --fixel_cmap "cmc.batlow" \
    --fixel_value_range 0.0 3e-3
```

To visualize just the directions, do not provide a `--fixel_data` argument. You can also
 save the visualization to a file using the `--save_path` argument.

If you want to match a set of fixels to a reference set, you can use the `fixelmatch` 
command.

## Container

For reproducibility, we also provide a containerized version. You can build the 
container as a Docker image using the provided Dockerfile. Or, you can download the 
pre-built Apptainer image. The usage is similar to above:

```bash
apptainer run \
    -B /path/to/data:/path/to/data \
    --nv \
    fiberfield_v0.2.0.sif \
    fiberfield \
    --input "/path/to/data/dwi.nii.gz" \
    ...
```

Note that when using the containerized version, the interactive version of `fixelview`
will not work. You must use `--save_path` and save the visualization to a file.

## References

If you use this code in your research, please cite the following paper:

> [1] Adam M. Saunders, Gaurav Rudravaram, Elyssa M. McMaster, Michael E. Kim, Trent
Schwartz, Yimeng Dou, Yihao Liu, Lianrui Zuo, Adam W. Anderson, and Bennett A. Landman.
"Neural fields for adaptive white matter microstructural modeling." Submitted to SPIE 
Medical Imaging: Image Processing, 2026.
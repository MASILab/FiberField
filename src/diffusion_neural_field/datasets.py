import torch
from torch.utils.data.dataset import Dataset
import nibabel as nib
import numpy as np
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import (
    extract_dwi_shell,
    extract_b0,
    unique_bvals_tolerance,
    get_bval_indices,
)
from dipy.core.gradients import (
    gradient_table,
    reorient_vectors,
    orientation_from_string,
)


class DiffusionDataset(Dataset):
    def __init__(
        self,
        dmri_path,
        bval_path,
        bvec_path,
        bdelta_path=None,
        mask_path=None,
        shells=None,
        bval_tol=20,
        return_b0=False,  # return b0 as separate volume and do not include in main data
        scale=True,  # scale everything to [0, 1]
        norm_b0=False,  # normalize data by b0 in each voxel
    ):
        # Load diffusion MRI and bval/bvecs
        self.dmri = nib.load(dmri_path)
        current_orientation = "".join(nib.orientations.aff2axcodes(self.dmri.affine))
        self.dmri = nib.funcs.as_closest_canonical(self.dmri)

        self.bvals, self.bvecs = read_bvals_bvecs(bval_path, bvec_path)
        if bdelta_path is not None:
            self.bdelta = np.loadtxt(bdelta_path)
        else:
            self.bdelta = np.ones_like(
                self.bvals
            )  # Default to all bdelta = 1 (i.e. linear encoding)

        # Reorient bvals if needed
        current_orientation = orientation_from_string(current_orientation)
        target_orientation = orientation_from_string("RAS")
        self.bvecs = reorient_vectors(
            self.bvecs, current_orientation, target_orientation, axis=1
        )

        self.dmri_data = self.dmri.get_fdata()

        self.return_b0 = return_b0

        unique_bvals = unique_bvals_tolerance(self.bvals, tol=bval_tol)
        if shells is None:
            self.shells = unique_bvals.tolist()
        else:
            self.shells = shells
            if 0 not in self.shells:
                self.shells = [0] + self.shells

        gtab = gradient_table(bvals=self.bvals, bvecs=self.bvecs)
        indices, dmri_list, bvals_list, bvecs_list = extract_dwi_shell(
            dwi=self.dmri_data,
            gtab=gtab,
            bvals_to_extract=self.shells,
            tol=bval_tol,
            group_shells=True,
        )
        self.dmri_data = dmri_list[0]
        self.bvals = bvals_list[0]
        self.bvecs = bvecs_list[0]
        self.bdelta = self.bdelta[indices[0]]

        self.gtab = gradient_table(
            bvals=self.bvals, bvecs=self.bvecs, b0_threshold=bval_tol
        )

        self.mean_b0 = extract_b0(self.dmri_data, self.gtab.b0s_mask, strategy="mean")
        self.dmri_tensor = torch.from_numpy(self.dmri_data).float()
        self.mean_b0_tensor = torch.from_numpy(self.mean_b0).float()

        # Get list of coordinates
        dmri_shape = self.dmri_tensor.shape[:-1]
        x, y, z = torch.meshgrid(
            torch.arange(dmri_shape[0]),
            torch.arange(dmri_shape[1]),
            torch.arange(dmri_shape[2]),
            indexing="xy",
        )
        self.coords = torch.stack((x, y, z), dim=-1).view(-1, 3).float()

        # If mask is provided, filter coordinates and dmri_tensor
        if mask_path is not None:
            self.mask = nib.load(mask_path)
            self.mask = nib.funcs.as_closest_canonical(self.mask)
            self.mask_data = self.mask.get_fdata().copy()
            mask_tensor = torch.from_numpy(self.mask_data).bool()
            self.mask_flat = mask_tensor.reshape(-1)
            self.coords = self.coords[self.mask_flat]
            self.dmri_tensor = self.dmri_tensor.reshape(-1, self.dmri_tensor.shape[-1])[
                self.mask_flat
            ]
            self.mean_b0_tensor = self.mean_b0_tensor.reshape(-1)[self.mask_flat]
        else:
            self.dmri_tensor = self.dmri_tensor.reshape(-1, self.dmri_tensor.shape[-1])
            self.mean_b0_tensor = self.mean_b0_tensor.reshape(-1)

        # Normalize dmri_tensor by mean b0
        if norm_b0:
            self.norm_dmri_tensor = (self.dmri_tensor) / (
                self.mean_b0_tensor.unsqueeze(-1) + 1e-8
            )
        else:
            self.norm_dmri_tensor = self.dmri_tensor

        if return_b0:
            self.norm_dmri_tensor = self.norm_dmri_tensor[..., ~self.gtab.b0s_mask]

        if scale:
            self.scale_value = np.percentile(self.norm_dmri_tensor, 99.0)
            self.norm_dmri_tensor = (
                self.norm_dmri_tensor / torch.tensor(self.scale_value + 1e-8).float()
            )
        else:
            self.scale_value = 1.0

        # Normalize coords from -1 to 1 across length of each dimension
        self.norm_coords = torch.zeros_like(self.coords)
        for i in range(3):
            self.norm_coords[:, i] = (
                2.0 * (self.coords[:, i] / (dmri_shape[i] - 1)) - 1.0
            )

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        if self.return_b0:
            return (
                self.norm_coords[idx],
                self.norm_dmri_tensor[idx],
                self.mean_b0_tensor[idx],
            )
        else:
            return self.norm_coords[idx], self.norm_dmri_tensor[idx]

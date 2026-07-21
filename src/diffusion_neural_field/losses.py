import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data.dataset import Dataset
import nibabel as nib
from dipy.reconst.shm import sh_to_sf_matrix, sph_harm_ind_list
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import (
    extract_dwi_shell,
    extract_b0,
    unique_bvals_tolerance,
    get_bval_indices,
)
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere


class AmplitudeConstraintLoss(nn.Module):
    def __init__(self, threshold=0.1, sh_order_max=6):
        super(AmplitudeConstraintLoss, self).__init__()
        self.threshold = threshold
        self.sh_order_max = sh_order_max

        # Create conversion matrix
        self.sphere = get_sphere(name="symmetric724")
        B = sh_to_sf_matrix(
            sphere=self.sphere,
            sh_order_max=self.sh_order_max,
            basis_type="tournier07",
            return_inv=False,
        )
        self.B = torch.from_numpy(B).float()

    def forward(self, sh_coeffs):
        B = self.B.to(sh_coeffs.device)
        amplitudes = sh_coeffs @ B  # (num_batch, num_dirs)

        max_amplitude = (
            amplitudes.mean(dim=1, keepdim=True) * self.threshold
        )  # (num_batch, 1)
        negative_amplitudes = torch.clamp(amplitudes - max_amplitude, max=0)
        loss = (negative_amplitudes**2).sum(dim=1).mean()
        return loss


class MAPFiberLoss(nn.Module):
    """Calculates a weighted loss equivalent to MAP estimation with a prior encouraging fewer fibers."""

    def __init__(self, lambda_prior=0.1):
        super(MAPFiberLoss, self).__init__()
        self.lambda_prior = lambda_prior

    def forward(self, predicted_signal, target_signal, reduction="mean"):
        """
        Calculate weighted loss equivalent to MAP estimation with fewer fiber prior.
        L = -log sum_k exp(-MSE(predicted_signal_k, target_signal) + lambda_prior * (k - 1))

        Parameters
        ----------
        predicted_signal : torch.Tensor
            Signal predicted by model for each fiber, shape (n_fibers, batch_size, n_dirs)
        target_signal : torch.Tensor
            Target signal, shape (batch_size, n_dirs)
        """
        mse_loss = F.mse_loss(
            predicted_signal,
            target_signal.unsqueeze(0).repeat(predicted_signal.shape[0], 1, 1),
            reduction="none",
        )
        loss = torch.logsumexp(
            mse_loss
            + self.lambda_prior
            * (
                torch.arange(
                    -1, predicted_signal.shape[0] - 1, device=predicted_signal.device
                ).float()
            )[:, None, None],
            dim=0,
        )
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "none":
            return loss
        else:
            raise ValueError(f"Invalid reduction type: {reduction}")

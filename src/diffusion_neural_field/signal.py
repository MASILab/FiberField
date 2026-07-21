import numpy as np
import torch

from dipy.reconst.shm import sph_harm_ind_list, sh_to_sf_matrix, forward_sdeconv_mat
from dipy.core.sphere import Sphere
from dipy.core.gradients import gradient_table
from .utils import (
    legendre_exp_integral,
    legendre_exp_integral_vectorized,
    legendre_exp_integral_vectorized_num,
)
import warnings


class CSDSignal:
    def __init__(self, response_path, gtab, sh_order_max=6, scale_value=1.0):
        # Load response function as numpy array
        self.sh_order_max = sh_order_max
        response_data = np.loadtxt(response_path, comments="#")

        # If there are multiple rows, this is not a single shell response
        if response_data.ndim > 1:
            raise ValueError(
                "Response function file contains multiple rows. Expected single shell response."
            )

        # Should have 1 + sh_order_max/2 coefficients
        expected_num_coeffs = 1 + (sh_order_max // 2)
        if response_data.shape[0] != expected_num_coeffs:
            raise ValueError(
                f"Response function file has {response_data.shape[0]} coefficients, expected {expected_num_coeffs} for sh_order_max={sh_order_max}."
            )

        m_values, l_values = sph_harm_ind_list(
            sh_order_max=sh_order_max, full_basis=False
        )
        response_data *= torch.sqrt(
            4 * np.pi / (2 * torch.arange(0, max(l_values) + 1, 2) + 1)
        ).numpy()
        response_data /= scale_value
        self.R = forward_sdeconv_mat(
            r_rh=response_data,
            l_values=l_values,
        )
        self.R = torch.from_numpy(self.R).float()

        self.gtab = gtab
        self.sphere = Sphere(xyz=gtab.bvecs[~gtab.b0s_mask])
        B = sh_to_sf_matrix(
            sphere=self.sphere,
            sh_order_max=sh_order_max,
            basis_type="tournier07",
            return_inv=False,
        )
        self.B = torch.from_numpy(B).float()

    def compute_signal(self, sh_coeffs, b0=None):
        """
        Compute diffusion signal from SH coefficients using the response function.

        Parameters
        ----------
        sh_coeffs : torch.Tensor
            Tensor of shape (num_batch, num_sh_coeffs) containing SH coefficients.
        b0 : torch.Tensor
            Tensor of shape (num_batch,) containing S0 values.

        Returns
        -------
        signal : torch.Tensor
            Tensor of shape (num_batch, num_dirs) containing the computed diffusion signals.
        """
        B = self.B.to(sh_coeffs.device)
        R = self.R.to(sh_coeffs.device)

        signal = (sh_coeffs @ R) @ B
        if b0 is not None:
            signal = signal * b0[:, None]

        return signal


class StandardModelSignal:
    def __init__(
        self, gtab, sh_order_max=2, bdelta=None, numerical=False, include_b0=True
    ):
        # Load response function as numpy array
        self.sh_order_max = sh_order_max

        self.m_values, self.l_values = sph_harm_ind_list(
            sh_order_max=sh_order_max, full_basis=False
        )

        self.gtab = gtab
        self.include_b0 = include_b0
        self.bvals = self.gtab.bvals
        self.bvecs = self.gtab.bvecs
        if not self.include_b0:
            self.bvals = self.gtab.bvals[~self.gtab.b0s_mask]
            self.bvecs = self.gtab.bvecs[~self.gtab.b0s_mask]
            self.gtab = gradient_table(bvals=self.bvals, bvecs=self.bvecs)
        self.sphere = Sphere(xyz=self.gtab.bvecs)

        B, invB = sh_to_sf_matrix(
            sphere=self.sphere,
            sh_order_max=sh_order_max,
            basis_type="tournier07",
            return_inv=True,
        )
        self.B = torch.from_numpy(B).float()
        self.invB = torch.from_numpy(invB).float()
        if bdelta is not None:
            self.bdelta = bdelta
        else:
            self.bdelta = np.ones_like(
                self.gtab.bvals
            )  # Default to all bdelta = 1 (i.e. linear encoding)

        # If numerical is False and any bdelta are < 0, raise a warning
        if not numerical and np.any(self.bdelta < 0):
            warnings.warn(
                "Negative bdelta values detected."
                "Analytical approximation is only valid for positive bdelta values. Consider setting numerical=True."
            )

        self.legendre_exp_integral_fn = (
            legendre_exp_integral_vectorized_num
            if numerical
            else legendre_exp_integral_vectorized
        )

    def compute_signal(
        self,
        d_intra,
        d_extra_par,
        d_extra_perp,
        f_intra,
        fod,
        b0=None,
        **kwargs,
    ):
        """
        Compute diffusion signal from Standard Model parameters.

        Parameters
        ----------
        d_intra : torch.Tensor
            Tensor of shape (num_batch,) containing intra-cellular diffusivity.
        d_extra_par : torch.Tensor
            Tensor of shape (num_batch,) containing extra-cellular parallel diffusivity.
        d_extra_perp : torch.Tensor
            Tensor of shape (num_batch,) containing extra-cellular perpendicular diffusivity.
        f_intra : torch.Tensor
            Tensor of shape (num_batch,) containing intra-cellular volume fraction.
        fod : torch.Tensor
            Tensor of shape (num_batch, num_sh_coeffs) containing FOD SH coefficients.
        b0 : torch.Tensor, optional
            Tensor of shape (num_batch,) containing S0 values.

        Returns
        -------
        signal : torch.Tensor
            Tensor of shape (num_batch, num_dirs) containing the computed diffusion signals.
        """
        device = d_intra.device
        dtype = d_intra.dtype

        # Measurement axes: shape (1, N)
        b = torch.as_tensor(self.gtab.bvals, dtype=dtype, device=device)[None, :]
        bd = torch.as_tensor(self.bdelta, dtype=dtype, device=device)[None, :]

        # Voxel/parameter axes: shape (B, 1)
        Di = d_intra[:, None]
        De = d_extra_par[:, None]
        Dp = d_extra_perp[:, None]

        # Harmonic orders: shape (L,)
        # lvals = torch.tensor(self.l_values, device=device)
        lvals = torch.arange(0, self.sh_order_max + 1, 2, device=device)

        # Arguments to integral: shape (B, N)
        a_intra = b * bd * Di
        a_extra = b * bd * (De - Dp)

        # Integral terms: shape (B, N, L)
        Ki = self.legendre_exp_integral_fn(lvals, a_intra)
        Ke = self.legendre_exp_integral_fn(lvals, a_extra)

        # Exponential prefactors: shape (B, N)
        intra_exp = torch.exp((b * bd * Di) / 3.0 - (b * Di) / 3.0)

        extra_exp = torch.exp((b * bd * (De - Dp)) / 3.0 - (b * (De + 2.0 * Dp)) / 3.0)

        # Mixture: shape (B, N, L)
        f = f_intra[:, None, None]

        K = f * intra_exp[:, :, None] * Ki + (1.0 - f) * extra_exp[:, :, None] * Ke

        if b0 is not None:
            K = K * b0[:, None, None]

        # Method-1-style scaling: shape (1, 1, L)
        l_float = lvals.to(dtype=dtype)
        scale = (
            2.0
            * torch.sqrt(
                torch.tensor(torch.pi, dtype=dtype, device=device)
                * (2.0 * l_float + 1.0)
            )
        )[None, None, :]

        K = K * scale

        # Put into conv vector
        conv_vec = torch.zeros(
            (K.shape[0], K.shape[1], self.B.shape[0]), device=device, dtype=dtype
        )
        for i, l in enumerate(range(0, self.sh_order_max + 1, 2)):
            for m in range(-l, l + 1):
                idx = (l**2 + l) // 2 + m
                conv_vec[:, :, idx] = K[:, :, i] * torch.sqrt(
                    torch.tensor(
                        4.0 * torch.pi / (2 * l + 1.0), dtype=dtype, device=device
                    )
                )

        # Perform convolution (multiplication in spherical harmonic domain)
        signal = torch.einsum(
            "bk, bdk, dk -> bd", fod, conv_vec, self.B.T.to(fod.device)
        )

        return signal


class StandardModelFWSignal:
    def __init__(self, gtab, sh_order_max=2, include_b0=True):
        # Load response function as numpy array
        self.sh_order_max = sh_order_max

        self.m_values, self.l_values = sph_harm_ind_list(
            sh_order_max=sh_order_max, full_basis=False
        )

        self.gtab = gtab
        self.include_b0 = include_b0
        self.bvals = self.gtab.bvals
        self.bvecs = self.gtab.bvecs
        if not self.include_b0:
            self.bvals = self.gtab.bvals[~self.gtab.b0s_mask]
            self.bvecs = self.gtab.bvecs[~self.gtab.b0s_mask]
            self.gtab = gradient_table(bvals=self.bvals, bvecs=self.bvecs)
        self.sphere = Sphere(xyz=self.gtab.bvecs)

        B = sh_to_sf_matrix(
            sphere=self.sphere,
            sh_order_max=sh_order_max,
            basis_type="tournier07",
            return_inv=False,
        )
        self.B = torch.from_numpy(B).float()

    def compute_signal(
        self, d_intra, d_extra_par, d_extra_perp, d_fw, f_intra, f_fw, fod, b0=None
    ):
        """
        Compute diffusion signal from Standard Model parameters.

        Parameters
        ----------
        d_intra : torch.Tensor
            Tensor of shape (num_batch,) containing intra-cellular diffusivity.
        d_extra_par : torch.Tensor
            Tensor of shape (num_batch,) containing extra-cellular parallel diffusivity.
        d_extra_perp : torch.Tensor
            Tensor of shape (num_batch,) containing extra-cellular perpendicular diffusivity.
        d_fw : torch.Tensor
            Tensor of shape (num_batch,) containing free water diffusivity.
        f_intra : torch.Tensor
            Tensor of shape (num_batch,) containing intra-cellular volume fraction.
        f_fw : torch.Tensor
            Tensor of shape (num_batch,) containing free water volume fraction.
        fod : torch.Tensor
            Tensor of shape (num_batch, num_sh_coeffs) containing FOD SH coefficients.
        b0 : torch.Tensor, optional
            Tensor of shape (num_batch,) containing S0 values.

        Returns
        -------
        signal : torch.Tensor
            Tensor of shape (num_batch, num_dirs) containing the computed diffusion signals.
        """
        bvals = torch.from_numpy(self.bvals).float().to(d_intra.device)  # shape (nb,)
        # broadcast bvals to shape (B, nb)
        bvals_exp = bvals[None, :]

        a_intra = bvals_exp * d_intra[:, None]  # (B, nb)
        a_extra = bvals_exp * (d_extra_par - d_extra_perp)[:, None]  # (B, nb)

        lvals = torch.tensor(self.l_values, device=d_intra.device)  # (n_l,)

        # Ki and Ke shapes: (B, nb, n_l)
        Ki = legendre_exp_integral_vectorized(lvals, a_intra)
        Ke = legendre_exp_integral_vectorized(lvals, a_extra)
        Ke = Ke * torch.exp(
            -bvals_exp.unsqueeze(-1) * d_extra_perp[:, None].unsqueeze(-1)
        )
        Kfw = torch.exp(-bvals_exp.unsqueeze(-1) * d_fw[:, None].unsqueeze(-1))

        # Combine with f_intra: broadcast to (B, nb, n_l)
        f_intra_exp = f_intra[:, None, None]
        f_fw_exp = f_fw[:, None, None]
        K = f_intra_exp * Ki + (1 - f_intra_exp - f_fw_exp) * Ke + f_fw_exp * Kfw

        if b0 is not None:
            K = K * b0[:, None, None]

        # Rescale by sqrt(4*pi)*(2l+1) - see Coelho et al. 2022
        l_float = lvals.to(d_intra.device).float()
        scale = (
            4
            * torch.sqrt(torch.tensor(torch.pi, device=d_intra.device))
            * (2 * l_float + 1)
        ).view(1, 1, -1)
        K *= scale

        # Perform convolution (multiplication in spherical harmonic domain)
        signal = torch.einsum("bk, bdk, dk -> bd", fod, K, self.B.T.to(fod.device))
        return signal


class StickSignal:
    """
    Model multple axons as sticks
    """

    def __init__(self, gtab, include_b0=True):
        self.gtab = gtab
        self.include_b0 = include_b0
        if not self.include_b0:
            self.bvals = self.gtab.bvals[~self.gtab.b0s_mask]
            self.bvecs = self.gtab.bvecs[~self.gtab.b0s_mask]
            self.gtab = gradient_table(bvals=self.bvals, bvecs=self.bvecs)
        self.sphere = Sphere(xyz=self.gtab.bvecs)
        self.bvals = torch.from_numpy(self.gtab.bvals).float()
        self.bvecs = torch.from_numpy(self.gtab.bvecs).float()

    def compute_signal(self, d_intra, f_intra, dirs, b0=None):
        """
        Compute diffusion signal from Standard Model parameters.

        Parameters
        ----------
        d_intra : torch.Tensor
            Tensor of shape (num_batch, num_fibers) containing intra-cellular diffusivities.
        f_intra : torch.Tensor
            Tensor of shape (num_batch, num_fibers) containing intra-cellular volume fraction.
        dirs : torch.Tensor
            Tensor of shape (num_batch, num_fibers, 3) containing fiber directions.
        b0 : torch.Tensor, optional
            Tensor of shape (num_batch,) containing S0 values.

        Returns
        -------
        signal : torch.Tensor
            Tensor of shape (num_batch, num_dirs) containing the computed diffusion signals.
        """
        bvecs = self.bvecs.to(d_intra.device)  # shape (nb, 3)
        bvals = self.bvals.to(d_intra.device)  # shape (nb,)
        num_fibers = d_intra.shape[1]
        mu = torch.einsum("dc,bkc->bdk", bvecs, dirs)  # (B, nb, num_fibers)
        K = torch.exp(
            -bvals[None, :, None] * d_intra[:, None, :] * mu.pow(2)
        )  # (B, nb, num_fibers)
        signal = torch.sum(f_intra[:, None, :] * K, dim=-1)  # (B, nb)
        if b0 is not None:
            signal = b0[:, None] * signal
        return signal


class DiscreteStandardModelSignal:
    """
    Standard Model with FOD represented as discrete fiber orientations instead of SH coefficients.
    """

    def __init__(self, gtab, include_b0=True):
        self.gtab = gtab
        self.include_b0 = include_b0
        if not self.include_b0:
            self.bvals = self.gtab.bvals[~self.gtab.b0s_mask]
            self.bvecs = self.gtab.bvecs[~self.gtab.b0s_mask]
            self.gtab = gradient_table(bvals=self.bvals, bvecs=self.bvecs)
        self.sphere = Sphere(xyz=self.gtab.bvecs)
        self.bvals = torch.from_numpy(self.gtab.bvals).float()
        self.bvecs = torch.from_numpy(self.gtab.bvecs).float()

    def compute_signal(
        self,
        d_intra,
        d_extra_par,
        d_extra_perp,
        f_intra,
        f_extra,
        dirs,
        b0=None,
        d_fw=None,
        f_fw=None,
    ):
        """
        Compute diffusion signal from Standard Model parameters.

        Parameters
        ----------
        d_intra : torch.Tensor
            Tensor of shape (num_batch, num_fibers) containing intra-cellular diffusivities.
        d_extra_par : torch.Tensor
            Tensor of shape (num_batch, num_fibers) containing extra-cellular parallel diffusivities.
        d_extra_perp : torch.Tensor
            Tensor of shape (num_batch, num_fibers) containing extra-cellular perpendicular diffusivities.
        f_intra : torch.Tensor
            Tensor of shape (num_batch, num_fibers) containing intra-cellular volume fraction.
        f_extra : torch.Tensor
            Tensor of shape (num_batch, num_fibers) containing extra-cellular volume fraction.
        dirs : torch.Tensor
            Tensor of shape (num_batch, num_fibers, 3) containing fiber directions.
        b0 : torch.Tensor, optional
            Tensor of shape (num_batch,) containing S0 values.
        d_fw : torch.Tensor, optional
            Tensor of shape (num_batch,) containing free water diffusivity (if freewater=True).
        f_fw : torch.Tensor, optional
            Tensor of shape (num_batch,) containing free water volume fraction (if freewater=True).

        Returns
        -------
        signal : torch.Tensor
            Tensor of shape (num_batch, num_dirs) containing the computed diffusion signals.
        """
        bvecs = self.bvecs.to(d_intra.device)  # shape (nb, 3)
        bvals = self.bvals.to(d_intra.device)  # shape (nb,)
        num_fibers = d_intra.shape[1]
        mu = torch.einsum("dc,bkc->bdk", bvecs, dirs)  # (B, nb, num_fibers)
        K_intra = torch.exp(
            -bvals[None, :, None] * d_intra[:, None, :] * mu.pow(2)
        )  # (B, nb, num_fibers)
        K_extra = torch.exp(
            -bvals[None, :, None] * d_extra_perp[:, None, :]
            - bvals[None, :, None]
            * (d_extra_par[:, None, :] - d_extra_perp[:, None, :])
            * mu.pow(2)
        )  # (B, nb, num_fibers)
        signal = torch.sum(
            f_intra[:, None, :] * K_intra + f_extra[:, None, :] * K_extra, dim=-1
        )  # (B, nb)
        if d_fw is not None:
            if f_fw is None:
                f_fw = 1 - f_intra.sum(dim=1) - f_extra.sum(dim=1)  # (B,)
                f_fw = f_fw[:, None]  # (B, 1)
            K_fw = torch.exp(-bvals * d_fw)  # (B, nb)
            signal += f_fw * K_fw
        if b0 is not None:
            signal = b0[:, None] * signal
        return signal

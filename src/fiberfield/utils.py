from dipy.viz import actor, window
from scipy.special import eval_legendre, gamma
import matplotlib.pyplot as plt
import matplotlib
import torch
from torchquad import Simpson
import numpy as np
from functools import partial
import vtk
from fury.utils import apply_affine
from fury.colormap import orient2rgb
import cmcrameri
from tqdm import tqdm

PI = torch.tensor(torch.pi)
SQRT_PI = torch.sqrt(PI)
EPS = 1e-6

# Set matplotlib font to 8
matplotlib.rcParams.update({"font.size": 6})


def plot_csd_slice(
    save_path, sh_coeffs, bg_img, signal_calculator, plot_slice, plot_vol=0
):
    scene = window.Scene()
    img_actor = actor.slicer(
        bg_img.get_fdata()[..., plot_vol],
        affine=bg_img.affine,
        interpolation="nearest",
    )

    odf_actor = actor.odf_slicer(
        sh_coeffs,
        affine=bg_img.affine,
        sphere=signal_calculator.sphere,
        B_matrix=signal_calculator.B.numpy(),
    )
    # Push out from actor
    curr_position = odf_actor.GetPosition()
    odf_actor.SetPosition(curr_position[0], curr_position[1], curr_position[2] + 1)
    scene.add(img_actor)
    scene.add(odf_actor)

    img_actor.display_extent(
        0,
        bg_img.shape[0],
        0,
        bg_img.shape[1],
        plot_slice,
        plot_slice,
    )
    odf_actor.display_extent(
        0,
        bg_img.shape[0],
        0,
        bg_img.shape[1],
        plot_slice,
        plot_slice,
    )

    scene.reset_camera()
    scene.zoom(2)
    window.snapshot(scene, fname=str(save_path), size=(1600, 1600))


def plot_sm_fw_slice(
    save_path,
    odf_save_path,
    D_intra,
    D_extra,
    D_perp,
    D_fw,
    f_intra,
    f_fw,
    S0,
    sh_coeffs,
    bg_img,
    signal_calculator,
    plot_slice,
    plot_vol=0,
    axis=2,
    cmap="gray",
    value_range={
        "D_intra": (0, 4e-3),
        "D_extra": (0, 4e-3),
        "D_perp": (0, 1.5e-3),
        "D_fw": (0, 4e-3),
        "f_intra": (0, 1.0),
        "f_extra": (0, 1.0),
        "f_fw": (0, 1.0),
        "S0": (None, None),
    },
):
    scene = window.Scene()
    bg_img_slice = bg_img.get_fdata()
    if len(bg_img_slice.shape) == 4:
        bg_img_slice = bg_img_slice[..., plot_vol]
    img_actor = actor.slicer(
        bg_img_slice,
        interpolation="nearest",
    )

    odf_actor = actor.odf_slicer(
        sh_coeffs,
        affine=bg_img.affine,
        sphere=signal_calculator.sphere,
        B_matrix=signal_calculator.B.numpy(),
    )

    if axis == 0:
        extent = (plot_slice, plot_slice, 0, bg_img.shape[1], 0, bg_img.shape[2])
    elif axis == 1:
        extent = (0, bg_img.shape[0], plot_slice, plot_slice, 0, bg_img.shape[2])
    else:
        extent = (0, bg_img.shape[0], 0, bg_img.shape[1], plot_slice, plot_slice)

    # Push out from actor
    curr_position = odf_actor.GetPosition()
    position = list(curr_position)
    position[axis] += 1
    odf_actor.SetPosition(position)

    scene.add(img_actor)
    scene.add(odf_actor)

    img_actor.display_extent(*extent)
    odf_actor.display_extent(*extent)

    scene.reset_camera()
    scene.zoom(2)
    if axis == 0:
        scene.camera().Azimuth(90)
    elif axis == 1:
        scene.camera().Elevation(90)
    if odf_save_path is not None:
        window.snapshot(scene, fname=str(odf_save_path), size=(1600, 1600))
    else:
        window.show(scene, size=(1600, 1600), reset_camera=False)

    f_extra = 1 - (f_intra + f_fw)
    f_extra[S0 == 0] = 0

    fig, ax = plt.subplots(1, 8, figsize=(15, 5))

    D_intra_plot = ax[0].imshow(
        np.take(D_intra, plot_slice, axis=axis).T,
        cmap=cmap,
        origin="lower",
        vmin=value_range["D_intra"][0],
        vmax=value_range["D_intra"][1],
    )
    D_extra_plot = ax[1].imshow(
        np.take(D_extra, plot_slice, axis=axis).T,
        cmap=cmap,
        origin="lower",
        vmin=value_range["D_extra"][0],
        vmax=value_range["D_extra"][1],
    )
    D_perp_plot = ax[2].imshow(
        np.take(D_perp, plot_slice, axis=axis).T,
        cmap=cmap,
        origin="lower",
        vmin=value_range["D_perp"][0],
        vmax=value_range["D_perp"][1],
    )
    f_intra_plot = ax[4].imshow(
        np.take(f_intra, plot_slice, axis=axis).T,
        cmap=cmap,
        origin="lower",
        vmin=value_range["f_intra"][0],
        vmax=value_range["f_intra"][1],
    )
    f_extra_plot = ax[5].imshow(
        np.take(f_extra, plot_slice, axis=axis).T,
        cmap=cmap,
        origin="lower",
        vmin=value_range["f_extra"][0],
        vmax=value_range["f_extra"][1],
    )
    fig.colorbar(
        D_intra_plot,
        ax=ax[0],
        location="bottom",
        format="%.1e",
        label="$mm^2/s$",
    )
    fig.colorbar(
        D_extra_plot,
        ax=ax[1],
        location="bottom",
        format="%.1e",
        label="$mm^2/s$",
    )
    fig.colorbar(
        D_perp_plot,
        ax=ax[2],
        location="bottom",
        format="%.1e",
        label="$mm^2/s$",
    )
    fig.colorbar(
        f_intra_plot,
        ax=ax[4],
        location="bottom",
    )
    fig.colorbar(
        f_extra_plot,
        ax=ax[5],
        location="bottom",
    )

    if D_fw is None:
        D_fw = np.zeros_like(D_intra)
    if f_intra is None:
        f_fw = np.zeros_like(f_intra)

    D_fw_plot = ax[3].imshow(
        np.take(D_fw, plot_slice, axis=axis).T,
        cmap=cmap,
        origin="lower",
        vmin=value_range["D_fw"][0],
        vmax=value_range["D_fw"][1],
    )
    f_fw_plot = ax[6].imshow(
        np.take(f_fw, plot_slice, axis=axis).T,
        cmap=cmap,
        origin="lower",
        vmin=value_range["f_fw"][0],
        vmax=value_range["f_fw"][1],
    )
    S0_plot = ax[7].imshow(
        np.take(S0, plot_slice, axis=axis).T,
        cmap=cmap,
        origin="lower",
        vmin=value_range["S0"][0],
        vmax=value_range["S0"][1],
    )
    fig.colorbar(f_fw_plot, ax=ax[6], location="bottom")
    fig.colorbar(S0_plot, ax=ax[7], location="bottom")
    fig.colorbar(
        D_fw_plot,
        ax=ax[3],
        location="bottom",
        format="%.1e",
        label="$mm^2/s$",
    )

    ax[0].set_title("D_intra")
    ax[1].set_title("D_extra")
    ax[2].set_title("D_perp")
    ax[3].set_title("D_fw")
    ax[4].set_title("f_intra")
    ax[5].set_title("f_extra")
    ax[6].set_title("f_fw")
    ax[7].set_title("S0")

    for ax_i in ax.flatten():
        ax_i.axis("off")

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


plot_sm_slice = partial(plot_sm_fw_slice, D_fw=None, f_fw=None)


def plot_stick_slice(
    save_path,
    dirs_save_path,
    D_intra,
    f_intra,
    S0,
    dirs,
    bg_img,
    signal_calculator,
    plot_slice,
    plot_vol=0,
    axis=2,
    cmap="gray",
    value_range={"D_intra": (0, 4e-3), "f_intra": (0, 1.0), "S0": (None, None)},
):
    scene = window.Scene()
    bg_img_slice = bg_img.get_fdata()
    if len(bg_img_slice.shape) == 4:
        bg_img_slice = bg_img_slice[..., plot_vol]
    img_actor = actor.slicer(
        bg_img_slice,
        interpolation="nearest",
    )

    n_fibers = dirs.shape[-1]
    peaks_actor = actor.peak_slicer(dirs, peaks_values=f_intra * n_fibers, colors=None)

    if axis == 0:
        extent = (plot_slice, plot_slice, 0, bg_img.shape[1], 0, bg_img.shape[2])
    elif axis == 1:
        extent = (0, bg_img.shape[0], plot_slice, plot_slice, 0, bg_img.shape[2])
    else:
        extent = (0, bg_img.shape[0], 0, bg_img.shape[1], plot_slice, plot_slice)

    # Push out from actor
    curr_position = peaks_actor.GetPosition()
    position = list(curr_position)
    position[axis] += 1
    peaks_actor.SetPosition(*position)

    scene.add(img_actor)
    scene.add(peaks_actor)

    img_actor.display_extent(*extent)
    peaks_actor.display_extent(*extent)

    scene.reset_camera()
    scene.zoom(2)
    if axis == 0:
        scene.camera().Azimuth(90)
    elif axis == 1:
        scene.camera().Elevation(90)
    if dirs_save_path is not None:
        window.snapshot(scene, fname=str(dirs_save_path), size=(1600, 1600))
    else:
        window.show(scene, size=(1600, 1600), reset_camera=False)

    num_fibers = dirs.shape[3]
    fig, ax = plt.subplots(num_fibers, 3, figsize=(8, 6), squeeze=False)

    for i in range(num_fibers):
        D_intra_plot = ax[i, 0].imshow(
            np.take(D_intra[:, :, :, i], plot_slice, axis=axis).T,
            cmap=cmap,
            origin="lower",
            vmin=value_range["D_intra"][0],
            vmax=value_range["D_intra"][1],
        )
        f_intra_plot = ax[i, 1].imshow(
            np.take(f_intra[:, :, :, i], plot_slice, axis=axis).T,
            cmap=cmap,
            origin="lower",
            vmin=value_range["f_intra"][0],
            vmax=value_range["f_intra"][1],
        )
        fig.colorbar(
            D_intra_plot,
            ax=ax[i, 0],
            location="bottom",
            format="%.1e",
            label="$mm^2/s$",
        )
        fig.colorbar(
            f_intra_plot,
            ax=ax[i, 1],
            location="bottom",
        )
    S0_plot = ax[0, 2].imshow(
        np.take(S0, plot_slice, axis=axis).T,
        cmap=cmap,
        origin="lower",
        vmin=value_range["S0"][0],
        vmax=value_range["S0"][1],
    )
    fig.colorbar(S0_plot, ax=ax[0, 2], location="bottom")

    ax[0, 0].set_title("D_intra")
    ax[0, 1].set_title("f_intra")
    ax[0, 2].set_title("S0")

    for ax_i in ax.flatten():
        ax_i.axis("off")

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_discrete_sm_slice(
    save_path,
    dirs_save_path,
    D_intra,
    D_extra_par,
    D_extra_perp,
    D_fw,
    f_intra,
    f_extra,
    f_fw,
    S0,
    dirs,
    bg_img,
    signal_calculator,
    plot_slice,
    plot_vol=0,
    axis=2,
    cmap="gray",
    value_range={
        "D_intra": (0, 4e-3),
        "D_extra_par": (0, 4e-3),
        "D_extra_perp": (0, 1.5e-3),
        "D_fw": (0, 4e-3),
        "f_intra": (0, 1.0),
        "f_extra": (0, 1.0),
        "f_fw": (0, 1.0),
        "S0": (None, None),
    },
):
    scene = window.Scene()
    bg_img_slice = bg_img.get_fdata()
    if len(bg_img_slice.shape) == 4:
        bg_img_slice = bg_img_slice[..., plot_vol]
    img_actor = actor.slicer(
        bg_img_slice,
        affine=bg_img.affine,
        interpolation="nearest",
    )

    f_total = f_intra + f_extra
    n_fibers = f_intra.shape[-1]
    peaks_actor = actor.peak_slicer(
        dirs, peaks_values=f_total * n_fibers, affine=bg_img.affine, colors=None
    )

    if axis == 0:
        extent = (plot_slice, plot_slice, 0, bg_img.shape[1], 0, bg_img.shape[2])
    elif axis == 1:
        extent = (0, bg_img.shape[0], plot_slice, plot_slice, 0, bg_img.shape[2])
    else:
        extent = (0, bg_img.shape[0], 0, bg_img.shape[1], plot_slice, plot_slice)

    # Push out from actor
    curr_position = peaks_actor.GetPosition()
    position = list(curr_position)
    position[axis] += 1
    peaks_actor.SetPosition(*position)

    scene.add(img_actor)
    scene.add(peaks_actor)

    img_actor.display_extent(*extent)
    peaks_actor.display_extent(*extent)

    scene.reset_camera()
    scene.zoom(2)
    if axis == 0:
        scene.camera().Azimuth(90)
    elif axis == 1:
        scene.camera().Elevation(90)
    if dirs_save_path is not None:
        window.snapshot(scene, fname=str(dirs_save_path), size=(1600, 1600))
    else:
        window.show(scene, size=(1600, 1600), reset_camera=False)

    num_fibers = dirs.shape[3]
    fig, ax = plt.subplots(num_fibers, 8, figsize=(15, 5), squeeze=False)

    for i in range(num_fibers):
        D_intra_plot = ax[i, 0].imshow(
            np.take(D_intra[:, :, :, i], plot_slice, axis=axis).T,
            cmap=cmap,
            origin="lower",
            vmin=value_range["D_intra"][0],
            vmax=value_range["D_intra"][1],
        )
        D_extra_par_plot = ax[i, 1].imshow(
            np.take(D_extra_par[:, :, :, i], plot_slice, axis=axis).T,
            cmap=cmap,
            origin="lower",
            vmin=value_range["D_extra_par"][0],
            vmax=value_range["D_extra_par"][1],
        )
        D_extra_perp_plot = ax[i, 2].imshow(
            np.take(D_extra_perp[:, :, :, i], plot_slice, axis=axis).T,
            cmap=cmap,
            origin="lower",
            vmin=value_range["D_extra_perp"][0],
            vmax=value_range["D_extra_perp"][1],
        )
        f_intra_plot = ax[i, 4].imshow(
            np.take(f_intra[:, :, :, i], plot_slice, axis=axis).T,
            cmap=cmap,
            origin="lower",
            vmin=value_range["f_intra"][0],
            vmax=value_range["f_intra"][1],
        )
        f_extra_plot = ax[i, 5].imshow(
            np.take(f_extra[:, :, :, i], plot_slice, axis=axis).T,
            cmap=cmap,
            origin="lower",
            vmin=value_range["f_extra"][0],
            vmax=value_range["f_extra"][1],
        )
        fig.colorbar(
            D_intra_plot,
            ax=ax[i, 0],
            location="bottom",
            format="%.1e",
            label="$mm^2/s$",
        )
        fig.colorbar(
            D_extra_par_plot,
            ax=ax[i, 1],
            location="bottom",
            format="%.1e",
            label="$mm^2/s$",
        )
        fig.colorbar(
            D_extra_perp_plot,
            ax=ax[i, 2],
            location="bottom",
            format="%.1e",
            label="$mm^2/s$",
        )
        fig.colorbar(
            f_intra_plot,
            ax=ax[i, 4],
            location="bottom",
        )
        fig.colorbar(
            f_extra_plot,
            ax=ax[i, 5],
            location="bottom",
        )

    D_fw_plot = ax[0, 3].imshow(
        np.take(D_fw, plot_slice, axis=axis).T,
        cmap=cmap,
        origin="lower",
        vmin=value_range["D_fw"][0],
        vmax=value_range["D_fw"][1],
    )
    f_fw_plot = ax[0, 6].imshow(
        np.take(f_fw, plot_slice, axis=axis).T,
        cmap=cmap,
        origin="lower",
        vmin=value_range["f_fw"][0],
        vmax=value_range["f_fw"][1],
    )
    S0_plot = ax[0, 7].imshow(
        np.take(S0, plot_slice, axis=axis).T,
        cmap=cmap,
        origin="lower",
        vmin=value_range["S0"][0],
        vmax=value_range["S0"][1],
    )
    fig.colorbar(f_fw_plot, ax=ax[0, 6], location="bottom")
    fig.colorbar(S0_plot, ax=ax[0, 7], location="bottom")
    fig.colorbar(
        D_fw_plot,
        ax=ax[0, 3],
        location="bottom",
        format="%.1e",
        label="$mm^2/s$",
    )

    ax[0, 0].set_title("D_intra")
    ax[0, 1].set_title("D_extra_par")
    ax[0, 2].set_title("D_extra_perp")
    ax[0, 3].set_title("D_fw")
    ax[0, 4].set_title("f_intra")
    ax[0, 5].set_title("f_extra")
    ax[0, 6].set_title("f_fw")
    ax[0, 7].set_title("S0")

    for ax_i in ax.flatten():
        ax_i.axis("off")

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def legendre(x, l):
    """Evaluate legendre polynomial at x for degree l."""
    x_np = x.detach().cpu().numpy()
    return torch.tensor(eval_legendre(int(l), x_np), device=x.device)


def legendre_exp(x, l, a):
    """Evaluate Legendre polynomial multiplied by exp(-a x^2)."""
    return legendre(x, l) * torch.exp(-a * x**2)


def legendre_exp_integral_num(a, l, lower=0.0, upper=1.0):
    """Compute integral of Legendre polynomial of order l multiplied by exp(-a x^2) from lower to upper."""
    integrand = lambda x: legendre_exp(x, l, a)
    sampler = Simpson()
    integral = sampler.integrate(
        integrand, dim=1, integration_domain=torch.tensor([[lower, upper]]), N=201
    )
    return integral


def legendre_integral_num(l, lower=0.0, upper=1.0):
    """Compute integral of Legendre polynomial of order l from lower to upper."""
    integrand = lambda x: legendre(x, l)
    sampler = Simpson()
    integral = sampler.integrate(
        integrand, dim=1, integration_domain=torch.tensor([[lower, upper]]), N=201
    )
    return integral


def legendre_exp_integral(l, a):
    orig_a = a.clone()
    a = torch.clamp(a, min=EPS)

    if l == 0:
        output = SQRT_PI * torch.erf(torch.sqrt(a)) / (2 * torch.sqrt(a))

    elif l == 2:
        num = -6 * torch.sqrt(a) * torch.exp(-a) + (3 - 2 * a) * SQRT_PI * torch.erf(
            torch.sqrt(a)
        )
        denom = 8 * a ** (3 / 2)
        output = num / denom

    elif l == 4:
        term1_num = -5 * (21 + 2 * a) * torch.exp(-a)
        term1_denom = 32 * a**2
        term1 = term1_num / term1_denom

        term2_num = 3 * (35 + 4 * (-5 + a) * a) * SQRT_PI * torch.erf(torch.sqrt(a))
        term2_denom = 64 * a ** (5 / 2)
        term2 = term2_num / term2_denom
        output = term1 + term2

    elif l == 6:
        num_term1 = -42 * torch.sqrt(a) * (165 + 4 * a * (5 + a)) * torch.exp(-a)
        num_term2 = (
            -5
            * (-693 + 378 * a - 84 * a**2 + 8 * a**3)
            * SQRT_PI
            * torch.erf(torch.sqrt(a))
        )
        denom = 256 * a ** (7 / 2)
        output = (num_term1 + num_term2) / denom

    elif l == 8:
        num_term1 = (
            -6
            * torch.sqrt(a)
            * (225225 + 2 * a * (15015 + 2 * a * (1925 + 62 * a)))
            * torch.exp(-a)
        )
        num_term2 = (
            35
            * (19305 + 8 * a * (-1287 + a * (297 + 2 * (-18 + a) * a)))
            * SQRT_PI
            * torch.erf(torch.sqrt(a))
        )
        denom = 4096 * a ** (9 / 2)
        output = (num_term1 + num_term2) / denom

    else:
        raise ValueError("Legendre polynomial order must be 0,2,4,6, or 8.")

    # Approximation from Hendriks et al. for small a
    output = torch.where(
        orig_a < EPS, 0.5 * gamma(l + 0.5) / gamma(2 * l + 1.5) * (-orig_a) ** l, output
    )
    return output


def legendre_integral(l):
    return 0.5 * gamma(l + 0.5) / gamma(2 * l + 1.5)


def legendre_exp_integral_vectorized(lvals, a):
    """
    Vectorized computation of integral_{0..1} exp(-a x^2) P_l(x) dx
    for l in {0,2,4,6,8}.

    Parameters
    ----------
    lvals : 1D torch.LongTensor or torch.Tensor of ints, shape (n_l,)
        The Legendre orders (must be subset of {0,2,4,6,8}).
    a : torch.Tensor, shape (..., nb)
        Values of 'a' (e.g., b * diffusivity). Must be >= 0.

    Returns
    -------
    out : torch.Tensor, shape (..., nb, n_l)
        Integral values for each l in lvals.
    """
    # Validate lvals are supported
    allowed = {0, 2, 4, 6, 8}
    lvals = torch.as_tensor(lvals, dtype=torch.long, device=a.device)
    if not set(lvals.cpu().tolist()).issubset(allowed):
        raise ValueError("lvals must be subset of {0,2,4,6,8}")

    # Save original a for small-a approximation, then clamp for main formula
    orig_a = a
    a = torch.clamp_min(a, EPS)

    # Bring a to shape (..., nb, 1) so we can broadcast against l dimension
    a_exp = a.unsqueeze(-1)  # shape (..., nb, 1)
    orig_a_exp = orig_a.unsqueeze(-1)  # shape (..., nb, 1)

    # l tensor broadcastable shape (1, 1, n_l)
    l = lvals.to(a.device).view(
        *([1] * (a_exp.dim() - 1)), -1
    )  # shape (..., 1, n_l) but broadcast works

    # Precompute common terms
    sqrt_a = torch.sqrt(a_exp)
    erf_sqrt_a = torch.erf(sqrt_a)
    exp_neg_a = torch.exp(-a_exp)

    sqrt_pi = torch.sqrt(torch.tensor(torch.pi, device=a.device))
    out_shape = (*a_exp.shape[:-1], l.shape[-1])
    main_output = torch.zeros(out_shape, dtype=a_exp.dtype, device=a.device)

    # Masks for each l (shape (n_l,))
    mask_l0 = lvals == 0
    mask_l2 = lvals == 2
    mask_l4 = lvals == 4
    mask_l6 = lvals == 6
    mask_l8 = lvals == 8

    # compute each formula (all broadcast to (..., nb, n_l))
    if mask_l0.any():
        val0 = (sqrt_pi * erf_sqrt_a) / (2 * sqrt_a)
        main_output[..., mask_l0] = val0

    if mask_l2.any():
        num = -6 * sqrt_a * exp_neg_a + (3 - 2 * a_exp) * sqrt_pi * erf_sqrt_a
        denom = 8 * a_exp ** (3 / 2)
        main_output[..., mask_l2] = num / denom

    if mask_l4.any():
        term1_num = -5 * (21 + 2 * a_exp) * exp_neg_a
        term1_denom = 32 * a_exp**2
        term1 = term1_num / term1_denom

        term2_num = 3 * (35 + 4 * (-5 + a_exp) * a_exp) * sqrt_pi * erf_sqrt_a
        term2_denom = 64 * a_exp ** (5 / 2)
        term2 = term2_num / term2_denom

        main_output[..., mask_l4] = term1 + term2

    if mask_l6.any():
        num_term1 = -42 * sqrt_a * (165 + 4 * a_exp * (5 + a_exp)) * exp_neg_a
        num_term2 = (
            -5
            * (-693 + 378 * a_exp - 84 * a_exp**2 + 8 * a_exp**3)
            * sqrt_pi
            * erf_sqrt_a
        )
        denom = 256 * a_exp ** (7 / 2)
        main_output[..., mask_l6] = (num_term1 + num_term2) / denom

    if mask_l8.any():
        num_term1 = (
            -6
            * sqrt_a
            * (225225 + 2 * a_exp * (15015 + 2 * a_exp * (1925 + 62 * a_exp)))
            * exp_neg_a
        )
        num_term2 = (
            35
            * (19305 + 8 * a_exp * (-1287 + a_exp * (297 + 2 * (-18 + a_exp) * a_exp)))
            * sqrt_pi
            * erf_sqrt_a
        )
        denom = 4096 * a_exp ** (9 / 2)
        main_output[..., mask_l8] = (num_term1 + num_term2) / denom

    # Approximation from Hendriks et al. for small a
    l_float = lvals.to(a.device).view(*([1] * (a_exp.dim() - 1)), -1).to(a_exp.dtype)
    small_pref = (
        0.5
        * torch.exp(torch.lgamma(l_float + 0.5))
        / torch.exp(torch.lgamma(2 * l_float + 1.5))
    )

    # Compute (-orig_a) ** l -- l are integers 0,2,4,... so negative base is safe
    small_power = torch.pow(-orig_a_exp, l_float)
    small_output = small_pref * small_power

    # Apply threshold mask and choose values
    small_mask = orig_a_exp < EPS
    out = torch.where(small_mask, small_output, main_output)

    return out  # shape (..., nb, n_l)


def legendre_exp_integral_vectorized_num(lvals, a):
    integral = torch.zeros((*a.shape, len(lvals)), device=a.device)
    for j in range(a.shape[0]):
        for i, l in enumerate(lvals):
            integral[j, ..., i] = legendre_exp_integral_num(a[j], l)
    return integral


def sparse_to_fixel_format(peaks, sparse_data=None, eps=1e-8):
    """
    Convert sparse voxel format to MRtrix3 fixel format.

    Parameters
    -------
    peaks : ndarray
        (X, Y, Z, M, 3) array of peak directions.
    sparse_data : list of ndarray
        List of sparse arrays with shape
            (X, Y, Z, M, P) or (X, Y, Z, M)
        where P is the number of parameters per fixel.
    eps : float
        Threshold for considering a peak as valid (non-zero).

    Returns
    ----------
    index : ndarray
        (X, Y, Z, 2) MRtrix index image.
        index[...,0] = number of fixels in voxel
        index[...,1] = start index of first fixel
    directions : ndarray
        (Nfixels, 3, 1) fixel directions.
    fixel_data : list of ndarray, optional
        List of fixel-wise arrays with shape
            (Nfixels, P, 1) or (Nfixels, 1, 1)
    """

    if sparse_data is None:
        sparse_data = []

    peak_mag = np.linalg.norm(peaks, axis=-1)
    valid = peak_mag > eps

    # Sort so valid peaks come first
    order = np.argsort(~valid, axis=-1)

    # Reorder peaks
    peaks = np.take_along_axis(
        peaks,
        order[..., None],
        axis=-2,
    )

    # Reorder all sparse data
    reordered_sparse_data = []
    for data in sparse_data:
        if data.ndim == 5:
            reordered_sparse_data.append(
                np.take_along_axis(
                    data,
                    order[..., None],
                    axis=-2,
                )
            )
        else:
            reordered_sparse_data.append(
                np.take_along_axis(
                    data,
                    order,
                    axis=-1,
                )
            )

    sparse_data = reordered_sparse_data

    # peaks: [X, Y, Z, M, 3]
    peak_mag = np.linalg.norm(peaks, axis=-1)  # [X, Y, Z, M]
    valid = peak_mag > eps  # [X, Y, Z, M]
    counts = valid.sum(axis=-1).astype(np.int32)  # [X, Y, Z]

    index = np.zeros(peaks.shape[:3] + (2,), dtype=np.int32)
    index[..., 0] = counts

    # 1-based start index of first fixel in each voxel
    flat_counts = counts.ravel()
    flat_starts = np.zeros_like(flat_counts, dtype=np.int32)
    nonzero = flat_counts > 0
    flat_starts[nonzero] = np.cumsum(flat_counts)[nonzero] - flat_counts[nonzero]
    index[..., 1] = flat_starts.reshape(counts.shape)

    n_fixels = int(counts.sum())

    directions = np.zeros((n_fixels, 3, 1), dtype=peaks.dtype)
    fixel_data = []

    for data in sparse_data:
        n_params = data.shape[-1] if data.ndim == 5 else 1
        fixel_data.append(np.zeros((n_fixels, n_params, 1), dtype=data.dtype))

    for i in range(peaks.shape[0]):
        for j in range(peaks.shape[1]):
            for k in range(peaks.shape[2]):
                n_peaks = index[i, j, k, 0]
                if n_peaks == 0:
                    continue

                start = index[i, j, k, 1]  # convert to 0-based array index

                for l in range(n_peaks):
                    fixel_idx = start + l
                    directions[fixel_idx, :, 0] = peaks[i, j, k, l]

                    for m, out in enumerate(fixel_data):
                        if sparse_data[m].ndim == 5:
                            out[fixel_idx, :, 0] = sparse_data[m][i, j, k, l]
                        else:
                            out[fixel_idx, 0, 0] = sparse_data[m][i, j, k, l]

    return index, directions, fixel_data


def fixel_to_sparse_format(index, directions, fixel_data=None):
    """
    Convert MRtrix3 fixel format back to sparse voxel format.

    Parameters
    ----------
    index : ndarray
        (X, Y, Z, 2) MRtrix index image.
        index[...,0] = number of fixels in voxel
        index[...,1] = start index of first fixel
    directions : ndarray
        (Nfixels, 3, 1) fixel directions.
    fixel_data : list of ndarray, optional
        List of fixel-wise arrays with shape
            (Nfixels, P, 1) or (Nfixels, 1, 1)

    Returns
    -------
    peaks : ndarray
        (X, Y, Z, M, 3)
    sparse_data : list of ndarray
        One sparse array for each entry in fixel_data.
    """

    if fixel_data is None:
        fixel_data = []

    counts = index[..., 0].astype(int)
    starts = index[..., 1].astype(int)

    max_fibers = counts.max()

    peaks = np.zeros(index.shape[:3] + (max_fibers, 3), dtype=directions.dtype)

    sparse_data = []
    for data in fixel_data:
        n_params = data.shape[1]

        if n_params == 1:
            sparse = np.zeros(index.shape[:3] + (max_fibers,), dtype=data.dtype)
        else:
            sparse = np.zeros(
                index.shape[:3] + (max_fibers, n_params),
                dtype=data.dtype,
            )

        sparse_data.append(sparse)

    for i in range(index.shape[0]):
        for j in range(index.shape[1]):
            for k in range(index.shape[2]):
                n = counts[i, j, k]
                if n == 0:
                    continue

                start = starts[i, j, k]

                for l in range(n):
                    fixel_idx = start + l

                    peaks[i, j, k, l] = directions[fixel_idx, :, 0]

                    for m, sparse in enumerate(sparse_data):
                        if sparse.ndim == 4:
                            sparse[i, j, k, l] = fixel_data[m][fixel_idx, 0, 0]
                        else:
                            sparse[i, j, k, l] = fixel_data[m][fixel_idx, :, 0]

    return peaks, sparse_data


def visualize_fixels(
    index_nifti,
    directions_nifti,
    bg_img_nifti,
    bg_img_value_range=None,
    fixel_data_nifti=None,
    save_path=None,
    plot_slice=None,
    axis=2,
    fixel_cmap="viridis",
    fixel_value_range=None,
    scale=0.9,
    linewidth=3,
    size=(400, 400),
    colorbar=True,
    zoom=1,
    center=None,
):
    """Visualize MRtrix format fixels

    Parameters
    ----------
    index_nifti : Nifti2Image
        Index NIfTI in MRtrix3 fixel format
    directions_nifti : Nifti2Image
        Directions NIfTI in MRtrix3 fixel format
    bg_img_nifti : Nifti2Image
        Background NIfTI image
    bg_img_value_range : tuple, optional
        Value range for visualizing background image, by default None
    fixel_data_nifti : Nifti2Image, optional
        Fixel-wise data NIfTI image, by default None
    save_path : str, optional
        Path to save the visualization, by default None
    plot_slice : int or tuple, optional
        Slice index to visualize or center of coordinates, by default None
    axis : int, optional
        Axis along which to visualize the slice, by default 2
    fixel_cmap : str, optional
        Colormap for visualizing fixel data, by default "viridis"
    fixel_value_range : tuple, optional
        Value range for visualizing fixel data, by default None
    scale : float, optional
        Scale factor for visualizing fixel data, by default 0.9
    linewidth : int, optional
        Line width for visualizing fixel data, by default 5
    size : tuple, optional
        Size of the visualization, by default (400, 400)
    colorbar : bool, optional
        Whether to display a colorbar, by default True
    zoom : float, optional
        Zoom factor for the visualization, by default 1
    """
    scene = window.Scene()
    bg_img_data = bg_img_nifti.get_fdata()
    bg_img_data = np.nan_to_num(bg_img_data, nan=0.0)
    if bg_img_value_range is None:
        bg_img_value_range = (np.min(bg_img_data), np.max(bg_img_data))
    img_actor = actor.slicer(
        bg_img_data, interpolation="nearest", value_range=bg_img_value_range
    )

    index_data = index_nifti.get_fdata().astype(np.int32)
    directions_data = directions_nifti.get_fdata()
    if fixel_data_nifti is not None:
        fixel_data = [fixel_data_nifti.get_fdata()]
    else:
        fixel_data = []

    peaks, sparse_data = fixel_to_sparse_format(index_data, directions_data, fixel_data)

    if plot_slice is None:
        plot_slice = bg_img_nifti.shape[axis] // 2
    if isinstance(plot_slice, (list, tuple)):
        if len(plot_slice) != 3:
            plot_slice = plot_slice[0]
            center = None
        else:
            center = np.array(plot_slice)
            plot_slice = int(plot_slice[axis])
    else:
        center = None

    if axis == 0:
        extent = (
            plot_slice,
            plot_slice,
            0,
            bg_img_nifti.shape[1],
            0,
            bg_img_nifti.shape[2],
        )
        mask = np.zeros(peaks.shape[:3], dtype=bool)
        mask[plot_slice, :, :] = True
    elif axis == 1:
        extent = (
            0,
            bg_img_nifti.shape[0],
            plot_slice,
            plot_slice,
            0,
            bg_img_nifti.shape[2],
        )
        mask = np.zeros(peaks.shape[:3], dtype=bool)
        mask[:, plot_slice, :] = True
    else:
        extent = (
            0,
            bg_img_nifti.shape[0],
            0,
            bg_img_nifti.shape[1],
            plot_slice,
            plot_slice,
        )
        mask = np.zeros(peaks.shape[:3], dtype=bool)
        mask[:, :, plot_slice] = True

    colors = None
    colorbar_actor = None
    if len(sparse_data) > 0:
        cmap = plt.get_cmap(fixel_cmap)
        if fixel_value_range is not None:
            vmin, vmax = fixel_value_range
        else:
            vmin, vmax = np.min(sparse_data[0]), np.max(sparse_data[0])
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        colors = cmap(norm(sparse_data[0]))

        lut = vtk.vtkLookupTable()
        lut.SetTableRange(vmin, vmax)
        lut.SetNumberOfTableValues(256)
        lut.Build()
        for i in range(256):
            rgba = cmap(i / 255.0)
            lut.SetTableValue(i, *rgba)
        colorbar_actor = actor.scalar_bar(lookup_table=lut)
        label_prop = colorbar_actor.GetLabelTextProperty()
        label_prop.ItalicOff()
        label_prop.SetFontFamilyToArial()

    list_dirs = []
    list_colors = []

    # Get voxel centers in the slice
    ijk = np.argwhere(mask)

    if bg_img_nifti.affine is None:
        centers = ijk.astype(np.float32)
    else:
        # centers = apply_affine(bg_img_nifti.affine, ijk)
        centers = ijk.astype(np.float32)

    # Peak directions for all voxels in the slice
    dirs = peaks[
        ijk[:, 0],
        ijk[:, 1],
        ijk[:, 2],
    ]  # (Nvox, Npeaks, 3)

    # Find valid peaks
    valid = np.any(dirs != 0, axis=-1)  # (Nvox, Npeaks)
    voxel_idx, peak_idx = np.nonzero(valid)

    # Keep only valid peaks
    centers = centers[voxel_idx]  # (Nlines, 3)
    dirs = dirs[voxel_idx, peak_idx]  # (Nlines, 3)

    # Scale directions
    dirs = dirs * (scale / 2)

    # Build line endpoints
    starts = centers - dirs
    ends = centers + dirs

    # Shape: (Nlines, 2, 3)
    list_dirs = np.stack((starts, ends), axis=1)

    # Colors
    if colors is not None:
        list_colors = colors[
            ijk[voxel_idx, 0],
            ijk[voxel_idx, 1],
            ijk[voxel_idx, 2],
            peak_idx,
            :3,
        ]
    else:
        list_colors = orient2rgb(dirs)

    peaks_actor = actor.line(
        list_dirs,
        colors=np.asarray(list_colors),
        linewidth=linewidth,
        lod=False,
    )

    # Push out from actor
    curr_position = peaks_actor.GetPosition()
    position = list(curr_position)
    position[axis] += 1
    peaks_actor.SetPosition(*position)

    scene.add(img_actor)
    scene.add(peaks_actor)

    img_actor.display_extent(*extent)

    # Change projection
    scene.projection(proj_type="parallel")
    scene.reset_camera()

    # Get current distance from camera to focal point
    camera = scene.camera()
    focal_point = np.array(camera.GetFocalPoint())
    position = np.array(camera.GetPosition())
    offset = position - focal_point

    # Get center of the slice in world coordinates
    if center is None:
        center = np.mean(centers, axis=0)

    scene.zoom(zoom)
    if axis == 0:
        scene.camera().Azimuth(90)
    elif axis == 1:
        scene.camera().Elevation(90)
    if colorbar and colorbar_actor is not None:
        scene.add(colorbar_actor)

    # Set camera center and focal point to the center of the slice
    scene.camera().SetFocalPoint(center)
    scene.camera().SetPosition(center + offset)

    if save_path is not None:
        window.snapshot(scene, fname=str(save_path), size=size)
    else:
        window.show(scene, size=size, reset_camera=False)

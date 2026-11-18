from fiberfield.datasets import DiffusionDataset
from fiberfield.models import (
    StickMultiheadModel,
    DiscreteStandardModel,
)
from fiberfield.signal import StickSignal, DiscreteStandardModelSignal
from fiberfield.utils import (
    plot_stick_slice,
    sparse_to_fixel_format,
    plot_discrete_sm_slice,
)

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import nibabel as nib
from torch.special import i0e
from dipy.reconst.shm import sph_harm_ind_list

import numpy as np
from tqdm import tqdm, trange
from pathlib import Path
from datetime import datetime
import matplotlib
import yaml

matplotlib.use("Agg")


def log_likelihood(y_true, y_pred, sigma=1, model="gaussian"):
    if model == "gaussian":
        return -0.5 * torch.log(2 * torch.pi * sigma**2) - ((y_true - y_pred) ** 2) / (
            2 * sigma**2
        )
    elif model == "rician":
        term1 = torch.log(y_true / (sigma**2))
        term2 = (y_true**2 + y_pred**2) / (2 * sigma**2)
        x = (y_true * y_pred) / (sigma**2)
        term3 = torch.log(i0e(x)) + torch.abs(x)
        return term1 - term2 + term3
    else:
        raise ValueError("Invalid model type. Choose 'gaussian' or 'rician'.")


def likelihood(y_true, y_pred, sigma=1, model="gaussian"):
    if model == "gaussian":
        return (
            1
            / (np.sqrt(2 * torch.pi) * sigma)
            * torch.exp(-((y_true - y_pred) ** 2) / (2 * sigma**2))
        )
    elif model == "rician":
        term1 = y_true / (sigma**2)
        term2 = torch.exp(-(y_true**2 + y_pred**2) / (2 * sigma**2))
        x = (y_true * y_pred) / (sigma**2)
        term3 = i0e(x) * torch.exp(torch.abs(x))
        return term1 * term2 * term3
    else:
        raise ValueError("Invalid model type. Choose 'gaussian' or 'rician'.")


def combine_close_fibers(
    D_intra_batch, f_intra_batch, dirs_batch, min_angular_separation, type="greedy"
):
    N, k = f_intra_batch.shape
    if k < 2:
        return D_intra_batch, f_intra_batch, dirs_batch

    if type == "greedy":
        return _combine_close_fibers_greedy(
            D_intra_batch, f_intra_batch, dirs_batch, min_angular_separation
        )
    elif type == "single":
        return _combine_close_fibers_single_pass(
            D_intra_batch, f_intra_batch, dirs_batch, min_angular_separation
        )
    else:
        raise ValueError(f"Invalid type '{type}'. Choose 'greedy' or 'single'.")


def _combine_close_fibers_greedy(
    D_intra_batch, f_intra_batch, dirs_batch, min_angular_separation
):
    N, k = f_intra_batch.shape
    pair_i, pair_j = torch.triu_indices(k, k, offset=1, device=f_intra_batch.device)

    for _ in range(k - 1):
        active = f_intra_batch > 0

        di = dirs_batch[:, pair_i, :]
        dj = dirs_batch[:, pair_j, :]
        dot = (di * dj).sum(dim=-1).abs().clamp(-1.0, 1.0)
        angle_deg = torch.rad2deg(torch.acos(dot))

        pair_active = active[:, pair_i] & active[:, pair_j]
        angle_masked = torch.where(
            pair_active, angle_deg, torch.full_like(angle_deg, float("inf"))
        )

        min_angle, min_pair_idx = angle_masked.min(dim=-1)
        found_pair = min_angle < min_angular_separation

        if not bool(found_pair.any()):
            break

        sel_i = pair_i[min_pair_idx].unsqueeze(-1)
        sel_j = pair_j[min_pair_idx].unsqueeze(-1)

        f_i = torch.gather(f_intra_batch, 1, sel_i).squeeze(-1)
        f_j = torch.gather(f_intra_batch, 1, sel_j).squeeze(-1)
        D_i = torch.gather(D_intra_batch, 1, sel_i).squeeze(-1)
        D_j = torch.gather(D_intra_batch, 1, sel_j).squeeze(-1)
        dir_i = torch.gather(
            dirs_batch, 1, sel_i.unsqueeze(-1).expand(-1, -1, 3)
        ).squeeze(1)
        dir_j = torch.gather(
            dirs_batch, 1, sel_j.unsqueeze(-1).expand(-1, -1, 3)
        ).squeeze(1)

        f_sum = f_i + f_j

        # Directions are axes (sign-ambiguous), so align dir_j onto dir_i before averaging.
        sign = torch.sign((dir_i * dir_j).sum(dim=-1, keepdim=True))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        dir_j_aligned = dir_j * sign

        w_i = (f_i / (f_sum + 1e-8)).unsqueeze(-1)
        w_j = (f_j / (f_sum + 1e-8)).unsqueeze(-1)
        merged_dir = F.normalize(w_i * dir_i + w_j * dir_j_aligned, dim=-1, eps=1e-8)
        merged_D = w_i.squeeze(-1) * D_i + w_j.squeeze(-1) * D_j

        new_f_i = torch.where(found_pair, f_sum, f_i)
        new_D_i = torch.where(found_pair, merged_D, D_i)
        new_dir_i = torch.where(found_pair.unsqueeze(-1), merged_dir, dir_i)

        zeros_like_f = torch.zeros_like(f_j)
        new_f_j = torch.where(found_pair, zeros_like_f, f_j)
        new_D_j = torch.where(found_pair, zeros_like_f, D_j)
        new_dir_j = torch.where(
            found_pair.unsqueeze(-1), torch.zeros_like(dir_j), dir_j
        )

        f_intra_batch = f_intra_batch.scatter(1, sel_i, new_f_i.unsqueeze(-1))
        f_intra_batch = f_intra_batch.scatter(1, sel_j, new_f_j.unsqueeze(-1))
        D_intra_batch = D_intra_batch.scatter(1, sel_i, new_D_i.unsqueeze(-1))
        D_intra_batch = D_intra_batch.scatter(1, sel_j, new_D_j.unsqueeze(-1))
        dirs_batch = dirs_batch.scatter(
            1, sel_i.unsqueeze(-1).expand(-1, -1, 3), new_dir_i.unsqueeze(1)
        )
        dirs_batch = dirs_batch.scatter(
            1, sel_j.unsqueeze(-1).expand(-1, -1, 3), new_dir_j.unsqueeze(1)
        )

    return D_intra_batch, f_intra_batch, dirs_batch


def _combine_close_fibers_single_pass(
    D_intra_batch, f_intra_batch, dirs_batch, min_angular_separation
):
    N, k = f_intra_batch.shape
    device = f_intra_batch.device

    active = f_intra_batch > 0

    dot = torch.einsum("nid,njd->nij", dirs_batch, dirs_batch).abs().clamp(-1.0, 1.0)
    angle_deg = torch.rad2deg(torch.acos(dot))

    pair_active = active.unsqueeze(-1) & active.unsqueeze(-2)
    close = (angle_deg < min_angular_separation) & pair_active

    self_loop = torch.eye(k, dtype=torch.bool, device=device).unsqueeze(
        0
    ) & active.unsqueeze(-1)
    reach = (close | self_loop).float()

    # Transitive closure via repeated squaring: each squaring doubles the
    # path length covered, so ceil(log2(k - 1)) rounds fully connect any
    # component (diameter at most k - 1).
    num_rounds = max(1, (k - 1).bit_length())
    for _ in range(num_rounds):
        reach = (torch.bmm(reach, reach) > 0).float()

    idx = torch.arange(k, device=device).view(1, 1, k).expand(N, k, k)
    masked_idx = torch.where(reach > 0, idx, torch.full_like(idx, k))
    rep = masked_idx.min(dim=-1).values.clamp(max=k - 1)

    group_f_total = torch.einsum("nij,nj->ni", reach, f_intra_batch)

    ref_dir = torch.gather(dirs_batch, 1, rep.unsqueeze(-1).expand(-1, -1, 3))
    sign = torch.sign((dirs_batch * ref_dir).sum(dim=-1, keepdim=True))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    aligned_dirs = dirs_batch * sign

    weighted_dir_sum = torch.einsum(
        "nij,nj,njd->nid", reach, f_intra_batch, aligned_dirs
    )
    weighted_D_sum = torch.einsum("nij,nj,nj->ni", reach, f_intra_batch, D_intra_batch)

    safe_total = (group_f_total + 1e-8).unsqueeze(-1)
    merged_dir = F.normalize(weighted_dir_sum / safe_total, dim=-1, eps=1e-8)
    merged_D = weighted_D_sum / safe_total.squeeze(-1)

    is_rep_active = (
        torch.arange(k, device=device).view(1, k).expand(N, k) == rep
    ) & active

    new_f = torch.where(is_rep_active, group_f_total, torch.zeros_like(f_intra_batch))
    new_D = torch.where(is_rep_active, merged_D, torch.zeros_like(D_intra_batch))
    new_dirs = torch.where(
        is_rep_active.unsqueeze(-1), merged_dir, torch.zeros_like(dirs_batch)
    )

    return new_D, new_f, new_dirs


def run_sm_two_stage(
    input_path,
    bval_path,
    bvec_path,
    mask_path,
    output_dir,
    shells=None,
    max_fibers=3,
    layer_size=2048,
    num_enc=5000,
    pos_sigma=4,
    sigma=4,
    loss_criteria="BIC",
    likelihood="gaussian",
    batch_size=500,
    learning_rate=1e-4,
    num_epochs=1000,
    plot_slice=None,
    plot_axis=2,
    validation_frequency=10,
    validation_patience=25,
    min_diffusivity=1e-4,
    min_f_intra=1e-4,
    min_angular_separation=5,
    include_freewater=False,
    d_fw=3e-3,
    sh_order_max=4,
    device=None,
    stage="both",
    seed=None,
):
    if str(stage) not in ["both", "1", "2"]:
        raise ValueError("Invalid stage. Choose 'both', '1', or '2'.")
    config = {
        "loss": likelihood,
        "loss_criteria": loss_criteria,
        "p_k": [5, 9, 11],
        "max_fibers": max_fibers,
        "layer_size": layer_size,
        "num_enc": num_enc,
        "sigma": sigma,
        "pos_sigma": pos_sigma,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "plot_slice": plot_slice,
        "validate_every": validation_frequency,
        "validation_patience": validation_patience,
        "min_diffusivity": min_diffusivity,
        "min_f_intra": min_f_intra,
        "min_angular_separation": min_angular_separation,
        "device": device or ("cuda" if torch.cuda.is_available() else "cpu"),
        "seed": seed,
    }
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    output_dir_stick = Path(output_dir) / "stick"
    output_dir_stick.mkdir(parents=True, exist_ok=True)
    fixel_dir = output_dir_stick / "fixel"
    fixel_dir.mkdir(exist_ok=True)

    if str(stage) in ["both", "1"]:
        dataset = DiffusionDataset(
            dmri_path=input_path,
            bval_path=bval_path,
            bvec_path=bvec_path,
            mask_path=mask_path,
            shells=shells,
            scale=True,
        )
        N = dataset.dmri_data.shape[-1]
        if loss_criteria == "BIC":
            config["alpha"] = 0.5 * np.log(N)
        elif loss_criteria == "AIC":
            config["alpha"] = 1
        else:
            raise ValueError("Invalid loss criteria. Choose 'BIC' or 'AIC'.")
        alpha = config["alpha"]
        p_k = torch.tensor(config["p_k"]).to(config["device"])

        if config["plot_slice"] is None:
            config["plot_slice"] = dataset.dmri_data.shape[plot_axis] // 2

        # Write config to stick/config.yaml for reproducibility
        with open(output_dir_stick / "config.yaml", "w") as f:
            yaml.dump(config, f)

        train_dataloader = DataLoader(
            dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0
        )
        valid_dataloader = DataLoader(
            dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0
        )
        signal_calculator = StickSignal(
            gtab=dataset.gtab,
        )

        sigma_tensor = torch.tensor(config["sigma"]).to(config["device"])

        model = StickMultiheadModel(
            max_fibers=config["max_fibers"],
            layer_size=config["layer_size"],
            num_enc=config["num_enc"],
            sigma=config["pos_sigma"],
        )

        model = model.to(config["device"])

        coords, diffusion_signal = next(iter(train_dataloader))
        coords, diffusion_signal = (
            coords.to(config["device"]),
            diffusion_signal.to(config["device"]),
        )
        outputs = model(coords)
        diffusion_signal_reconst_list = []
        for k in range(len(outputs)):
            D_intra, f_intra, S0, dirs = outputs[k + 1]

            diffusion_signal_reconst_list.append(
                signal_calculator.compute_signal(
                    d_intra=D_intra,
                    f_intra=f_intra,
                    dirs=dirs,
                    b0=S0.squeeze(-1),
                )
            )
        diffusion_signal_reconst = torch.stack(diffusion_signal_reconst_list)

        # log_L = log_likelihood(
        #     diffusion_signal,
        #     diffusion_signal_reconst,
        #     sigma=sigma_tensor,
        #     model=config["loss"],
        # )
        # loss_value = -torch.logsumexp(
        #     torch.sum(log_L, dim=-1) - (alpha * p_k)[:, None], dim=0
        # ).mean()
        loss_value = F.mse_loss(
            diffusion_signal_reconst,
            diffusion_signal.repeat(diffusion_signal_reconst.shape[0], 1, 1),
        )
        tqdm.write(f"Loss value: {loss_value.item()}")

        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        best_loss = np.inf
        last_step = 0
        n_since_last_improvement = 0

        # run = wandb.init(
        #     project="fiberfield",
        #     config=config,
        #     mode="online",
        #     name=config["name"],
        # )
        model.to(config["device"])
        tqdm.write(f"Training on device: {config['device']}")

        epoch_pbar = trange(
            config["num_epochs"], desc="Stage 1: Stick Model", leave=False
        )
        for epoch in epoch_pbar:
            epoch_pbar.set_postfix(loss=f"{loss_value.item():.4f}")
            model.train()
            for batch_num, (coords, diffusion_signal) in enumerate(train_dataloader):
                coords = coords.to(config["device"])
                diffusion_signal = diffusion_signal.to(config["device"])

                output_list = model(coords)
                diffusion_signal_reconst_list = []
                for k in range(1, len(output_list) + 1):
                    D_intra, f_intra, S0, dirs = output_list[k]

                    diffusion_signal_reconst_list.append(
                        signal_calculator.compute_signal(
                            d_intra=D_intra,
                            f_intra=f_intra,
                            dirs=dirs,
                            b0=S0.squeeze(-1),
                        )
                    )

                # loss_map = loss(torch.stack(diffusion_signal_reconst_list), diffusion_signal)
                diffusion_signal_reconst = torch.stack(diffusion_signal_reconst_list)

                p_k = torch.tensor(config["p_k"]).to(config["device"])
                alpha = torch.tensor(config["alpha"]).to(config["device"])

                # log_L = log_likelihood(
                #     diffusion_signal,
                #     diffusion_signal_reconst,
                #     sigma=sigma_tensor,
                #     model=config["loss"],
                # )
                # loss_value = -torch.logsumexp(
                #     torch.sum(log_L, dim=-1) - (alpha * p_k)[:, None], dim=0
                # ).mean()
                loss_value = F.mse_loss(
                    diffusion_signal_reconst,
                    diffusion_signal.repeat(diffusion_signal_reconst.shape[0], 1, 1),
                )

                total_loss = loss_value.mean()

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # wandb.log(
                #     {
                #         "train/loss_map": total_loss.item(),
                #         "train/total_loss": total_loss.item(),
                #     },
                #     step=epoch * len(train_dataloader) + batch_num,
                # )
                last_step = epoch * len(train_dataloader) + batch_num

            if epoch % config["validate_every"] == 0:
                model.eval()

                with torch.no_grad():
                    D_intra_all = []
                    f_intra_all = []
                    diffusion_signal_all = []
                    diffusion_signal_reconst_all = []
                    S0_all = []
                    dirs_all = []

                    valid_pbar = tqdm(valid_dataloader, desc="Validation", leave=False)
                    for coords, diffusion_signal in valid_pbar:
                        coords = coords.to(config["device"])
                        diffusion_signal = diffusion_signal.to(config["device"])

                        outputs_list = model(coords)
                        (
                            D_intra_batch_list,
                            f_intra_batch_list,
                            S0_batch_list,
                            dirs_batch_list,
                            diffusion_signal_batch_list,
                        ) = [], [], [], [], []
                        for k in range(1, len(outputs_list) + 1):
                            D_intra_batch, f_intra_batch, S0_batch, dirs_batch = (
                                outputs_list[k]
                            )

                            # diffusivity_mask = D_intra_batch >= config["min_diffusivity"]
                            # D_intra_batch = D_intra_batch * diffusivity_mask
                            # f_intra_batch = f_intra_batch * diffusivity_mask
                            # f_intra_batch = f_intra_batch / (
                            #     f_intra_batch.sum(dim=-1, keepdim=True) + 1e-8
                            # )

                            diffusion_signal_batch_list.append(
                                signal_calculator.compute_signal(
                                    d_intra=D_intra_batch,
                                    f_intra=f_intra_batch,
                                    dirs=dirs_batch,
                                    b0=S0_batch.squeeze(-1),
                                )
                            )

                            # Pad everything to length 3
                            D_intra_batch = torch.nn.functional.pad(
                                D_intra_batch, (0, 3 - D_intra_batch.shape[-1])
                            )
                            f_intra_batch = torch.nn.functional.pad(
                                f_intra_batch, (0, 3 - f_intra_batch.shape[-1])
                            )
                            dirs_batch = torch.nn.functional.pad(
                                dirs_batch, (0, 0, 0, 3 - dirs_batch.shape[1])
                            )
                            D_intra_batch_list.append(D_intra_batch)
                            f_intra_batch_list.append(f_intra_batch)
                            S0_batch_list.append(S0_batch.detach().cpu())
                            dirs_batch_list.append(dirs_batch)

                        D_intra_batch = torch.stack(D_intra_batch_list, dim=1)
                        f_intra_batch = torch.stack(f_intra_batch_list, dim=1)
                        S0_batch = torch.stack(S0_batch_list, dim=1)
                        dirs_batch = torch.stack(dirs_batch_list, dim=1)

                        D_intra_all.append(D_intra_batch.detach().cpu())
                        f_intra_all.append(f_intra_batch.detach().cpu())
                        S0_all.append(S0_batch.detach().cpu())
                        dirs_all.append(dirs_batch.detach().cpu())
                        diffusion_signal_all.append(diffusion_signal.detach().cpu())
                        diffusion_signal_reconst_all.append(
                            torch.stack(diffusion_signal_batch_list).detach().cpu()
                        )

                    D_intra = torch.cat(D_intra_all, dim=0)
                    f_intra = torch.cat(f_intra_all, dim=0)
                    S0 = torch.cat(S0_all, dim=0)
                    dirs = torch.cat(dirs_all, dim=0)
                    diffusion_signal = torch.cat(diffusion_signal_all, dim=0)
                    diffusion_signal_reconst = torch.cat(
                        diffusion_signal_reconst_all, dim=1
                    )

                    k = torch.arange(1, config["max_fibers"] + 1)
                    p_k = torch.tensor(config["p_k"])
                    alpha = torch.tensor(config["alpha"])
                    sigma_tensor = torch.tensor(config["sigma"])

                    # log_L = log_likelihood(
                    #     diffusion_signal,
                    #     diffusion_signal_reconst,
                    #     sigma=sigma_tensor,
                    #     model=config["loss"],
                    # )
                    # loss_value = -torch.logsumexp(
                    #     torch.sum(log_L, dim=-1) - (alpha * p_k)[:, None], dim=0
                    # ).mean()
                    loss_value = F.mse_loss(
                        diffusion_signal_reconst,
                        diffusion_signal.repeat(
                            diffusion_signal_reconst.shape[0], 1, 1
                        ),
                    )
                    total_loss = loss_value.mean()

                    val_loss_map = total_loss
                    val_total_loss = val_loss_map

                # Select arg min over inner loss
                p_k = torch.tensor(config["p_k"])
                alpha = torch.tensor(config["alpha"])

                residuals = diffusion_signal_reconst - diffusion_signal.unsqueeze(0)
                sse = torch.sum(residuals**2, dim=-1)
                N = diffusion_signal.shape[-1]
                p_k = p_k.detach().cpu()
                sigma_tensor = torch.tensor(config["sigma"])

                if loss_criteria == "BIC":
                    bic = sse / sigma_tensor**2 + p_k[:, None] * np.log(N)

                    best_k = torch.argmin(bic, dim=0)
                elif loss_criteria == "AIC":
                    aic = sse / sigma_tensor**2 + 2 * p_k[:, None]
                    best_k = torch.argmin(aic, dim=0)
                # log_L = log_likelihood(
                #     diffusion_signal,
                #     diffusion_signal_reconst,
                #     sigma=sigma_tensor,
                #     model=config["loss"],
                # )
                # inner_term = torch.sum(log_L, dim=-1) - (alpha * p_k)[:, None]

                # best_k = torch.argmax(inner_term, dim=0)
                D_intra = D_intra[torch.arange(D_intra.shape[0]), best_k]
                f_intra = f_intra[torch.arange(f_intra.shape[0]), best_k]
                S0 = S0[torch.arange(S0.shape[0]), best_k]
                dirs = dirs[torch.arange(dirs.shape[0]), best_k]
                diffusion_signal_reconst = diffusion_signal_reconst[
                    best_k, torch.arange(diffusion_signal_reconst.shape[1]), :
                ]

                # Sort by f_intra in descending order
                sorted_indices = torch.argsort(f_intra, dim=-1, descending=True)
                D_intra = torch.gather(D_intra, dim=-1, index=sorted_indices)
                f_intra = torch.gather(f_intra, dim=-1, index=sorted_indices)
                dirs = torch.gather(
                    dirs, dim=-2, index=sorted_indices.unsqueeze(-1).expand(-1, -1, 3)
                )

                # Reshape
                D_intra_np = np.zeros(
                    dataset.dmri_data.shape[:3] + (config["max_fibers"],)
                )
                f_intra_np = np.zeros(
                    dataset.dmri_data.shape[:3] + (config["max_fibers"],)
                )
                S0_np = np.zeros(dataset.dmri_data.shape[:3])
                dirs_np = np.zeros(
                    dataset.dmri_data.shape[:3] + (config["max_fibers"], 3)
                )
                D_intra_np[dataset.mask_data.astype(bool), :] = (
                    D_intra.detach().cpu().numpy()
                )
                dirs_np[dataset.mask_data.astype(bool), :] = (
                    dirs.detach().cpu().numpy().reshape(-1, config["max_fibers"], 3)
                )
                f_intra_np[dataset.mask_data.astype(bool), :] = (
                    f_intra.detach().cpu().numpy()
                )
                S0_np[dataset.mask_data.astype(bool)] = (
                    S0.detach().cpu().numpy().squeeze(-1)
                )
                plot_stick_slice(
                    save_path=output_dir_stick / f"validation_epoch_{epoch}.png",
                    dirs_save_path=output_dir_stick
                    / f"peaks_validation_epoch_{epoch}.png",
                    D_intra=D_intra_np,
                    f_intra=f_intra_np,
                    S0=S0_np,
                    dirs=dirs_np,
                    bg_img=dataset.dmri,
                    signal_calculator=signal_calculator,
                    plot_slice=config["plot_slice"],
                    plot_vol=0,
                    axis=plot_axis,
                )

                # wandb.log(
                #     {
                #         "valid/image": wandb.Image(
                #             str(output_dir_stick / f"validation_epoch_{epoch}.png")
                #         ),
                #         "valid/odf_image": wandb.Image(
                #             str(output_dir_stick / f"peaks_validation_epoch_{epoch}.png")
                #         ),
                #         "valid/loss_map": val_loss_map.item(),
                #         "valid/total_loss": val_total_loss.item(),
                #     },
                #     step=last_step + 1,
                # )

                if val_total_loss.item() < best_loss:
                    best_loss = val_total_loss.item()
                    torch.save(model.state_dict(), output_dir_stick / "best_model.pth")
                    n_since_last_improvement = 0
                else:
                    n_since_last_improvement += 1

                if n_since_last_improvement >= config["validation_patience"]:
                    tqdm.write("Early stopping triggered.")
                    break

            if n_since_last_improvement >= config["validation_patience"]:
                break

        # Load best model
        model.load_state_dict(torch.load(output_dir_stick / "best_model.pth"))
        model.eval()

        with torch.no_grad():
            D_intra_all = []
            f_intra_all = []
            diffusion_signal_all = []
            diffusion_signal_reconst_all = []
            S0_all = []
            dirs_all = []

            for coords, diffusion_signal in valid_pbar:
                coords = coords.to(config["device"])
                diffusion_signal = diffusion_signal.to(config["device"])

                outputs_list = model(coords)
                (
                    D_intra_batch_list,
                    f_intra_batch_list,
                    S0_batch_list,
                    dirs_batch_list,
                    diffusion_signal_batch_list,
                ) = [], [], [], [], []
                for k in range(1, len(outputs_list) + 1):
                    D_intra_batch, f_intra_batch, S0_batch, dirs_batch = outputs_list[k]

                    diffusivity_mask = D_intra_batch >= config["min_diffusivity"]
                    f_intra_mask = f_intra_batch >= config["min_f_intra"]
                    output_mask = diffusivity_mask * f_intra_mask
                    D_intra_batch = D_intra_batch * output_mask
                    f_intra_batch = f_intra_batch * output_mask
                    dirs_batch = dirs_batch * output_mask[..., None]
                    f_intra_batch = f_intra_batch / (
                        f_intra_batch.sum(dim=-1, keepdim=True) + 1e-8
                    )

                    # Combine fibers that have less than min_angular_separation degrees between them
                    # Add the volume fractions and average the diffusivities/directions
                    D_intra_batch, f_intra_batch, dirs_batch = combine_close_fibers(
                        D_intra_batch,
                        f_intra_batch,
                        dirs_batch,
                        config["min_angular_separation"],
                        type="single",
                    )

                    diffusion_signal_batch_list.append(
                        signal_calculator.compute_signal(
                            d_intra=D_intra_batch,
                            f_intra=f_intra_batch,
                            dirs=dirs_batch,
                            b0=S0_batch.squeeze(-1),
                        )
                    )

                    # Pad everything to length 3
                    D_intra_batch = torch.nn.functional.pad(
                        D_intra_batch, (0, 3 - D_intra_batch.shape[-1])
                    )
                    f_intra_batch = torch.nn.functional.pad(
                        f_intra_batch, (0, 3 - f_intra_batch.shape[-1])
                    )
                    dirs_batch = torch.nn.functional.pad(
                        dirs_batch, (0, 0, 0, 3 - dirs_batch.shape[1])
                    )
                    D_intra_batch_list.append(D_intra_batch)
                    f_intra_batch_list.append(f_intra_batch)
                    S0_batch_list.append(S0_batch.detach().cpu())
                    dirs_batch_list.append(dirs_batch)

                D_intra_batch = torch.stack(D_intra_batch_list, dim=1)
                f_intra_batch = torch.stack(f_intra_batch_list, dim=1)
                S0_batch = torch.stack(S0_batch_list, dim=1)
                dirs_batch = torch.stack(dirs_batch_list, dim=1)

                D_intra_all.append(D_intra_batch.detach().cpu())
                f_intra_all.append(f_intra_batch.detach().cpu())
                S0_all.append(S0_batch.detach().cpu())
                dirs_all.append(dirs_batch.detach().cpu())
                diffusion_signal_all.append(diffusion_signal.detach().cpu())
                diffusion_signal_reconst_all.append(
                    torch.stack(diffusion_signal_batch_list).detach().cpu()
                )

            D_intra = torch.cat(D_intra_all, dim=0)
            f_intra = torch.cat(f_intra_all, dim=0)
            S0 = torch.cat(S0_all, dim=0)
            dirs = torch.cat(dirs_all, dim=0)
            diffusion_signal = torch.cat(diffusion_signal_all, dim=0)
            diffusion_signal_reconst = torch.cat(diffusion_signal_reconst_all, dim=1)

        # Select arg min over inner loss
        p_k = torch.tensor(config["p_k"])
        alpha = torch.tensor(config["alpha"])
        sigma_tensor = torch.tensor(config["sigma"])
        residuals = diffusion_signal_reconst - diffusion_signal.unsqueeze(0)
        sse = torch.sum(residuals**2, dim=-1)
        N = diffusion_signal.shape[-1]
        p_k = p_k.detach().cpu()
        sigma_tensor = torch.tensor(config["sigma"])

        if loss_criteria == "BIC":
            bic = sse / sigma_tensor**2 + p_k[:, None] * np.log(N)

            best_k = torch.argmin(bic, dim=0)
        elif loss_criteria == "AIC":
            aic = sse / sigma_tensor**2 + 2 * p_k[:, None]
            best_k = torch.argmin(aic, dim=0)

        # log_L = log_likelihood(
        #     diffusion_signal,
        #     diffusion_signal_reconst,
        #     sigma=sigma_tensor,
        #     model=config["loss"],
        # )
        # inner_term = torch.sum(log_L, dim=-1) - (alpha * p_k)[:, None]

        # best_k = torch.argmax(inner_term, dim=0)
        D_intra = D_intra[torch.arange(D_intra.shape[0]), best_k]
        f_intra = f_intra[torch.arange(f_intra.shape[0]), best_k]
        S0 = S0[torch.arange(S0.shape[0]), best_k]
        dirs = dirs[torch.arange(dirs.shape[0]), best_k]
        diffusion_signal_reconst = diffusion_signal_reconst[
            best_k, torch.arange(diffusion_signal_reconst.shape[1]), :
        ]

        # Sort everything by volume fraction within each voxel
        sorted_indices = torch.argsort(f_intra, dim=-1, descending=True)
        D_intra = torch.gather(D_intra, dim=-1, index=sorted_indices)
        f_intra = torch.gather(f_intra, dim=-1, index=sorted_indices)
        dirs = torch.gather(
            dirs, dim=-2, index=sorted_indices.unsqueeze(-1).expand(-1, -1, 3)
        )

        # Save coefficients
        D_intra_np = np.zeros(dataset.dmri_data.shape[:3] + (config["max_fibers"],))
        f_intra_np = np.zeros(dataset.dmri_data.shape[:3] + (config["max_fibers"],))
        S0_np = np.zeros(dataset.dmri_data.shape[:3])

        D_intra_np[dataset.mask_data.astype(bool), :] = D_intra.detach().cpu().numpy()
        f_intra_np[dataset.mask_data.astype(bool), :] = f_intra.detach().cpu().numpy()
        S0_np[dataset.mask_data.astype(bool)] = S0.detach().cpu().numpy().squeeze(-1)

        D_intra_img = nib.Nifti1Image(
            D_intra_np, affine=dataset.dmri.affine, header=dataset.dmri.header
        )
        f_intra_img = nib.Nifti1Image(
            f_intra_np, affine=dataset.dmri.affine, header=dataset.dmri.header
        )
        S0_img = nib.Nifti1Image(
            S0_np, affine=dataset.dmri.affine, header=dataset.dmri.header
        )

        nib.save(D_intra_img, str(output_dir_stick / "D_intra.nii.gz"))
        nib.save(f_intra_img, str(output_dir_stick / "f_intra.nii.gz"))
        nib.save(S0_img, str(output_dir_stick / "S0.nii.gz"))

        dirs_np = np.zeros(dataset.dmri_data.shape[:3] + (config["max_fibers"], 3))
        dirs_np[dataset.mask_data.astype(bool), :] = dirs.detach().cpu().numpy()
        dirs_img = nib.Nifti1Image(
            dirs_np, affine=dataset.dmri.affine, header=dataset.dmri.header
        )
        nib.save(dirs_img, str(output_dir_stick / "peaks.nii.gz"))

        diffusion_signal_reconst_np = np.zeros(dataset.dmri_data.shape)
        diffusion_signal_reconst_np[dataset.mask_data.astype(bool)] = (
            diffusion_signal_reconst.detach().cpu().numpy()
        )
        diffusion_signal_reconst_img = nib.Nifti1Image(
            diffusion_signal_reconst_np,
            affine=dataset.dmri.affine,
            header=dataset.dmri.header,
        )
        nib.save(
            diffusion_signal_reconst_img,
            str(output_dir_stick / "diffusion_signal_reconst.nii.gz"),
        )

        # Also save in MRtrix3 fixel format
        index, directions, fixel_data = sparse_to_fixel_format(
            peaks=dirs_np,
            sparse_data=[D_intra_np, f_intra_np],
        )

        index_nifti = nib.Nifti2Image(
            index, affine=dataset.dmri.affine, header=dataset.dmri.header
        )
        nib.save(index_nifti, str(fixel_dir / "index.nii.gz"))

        directions_nifti = nib.Nifti2Image(
            directions, affine=dataset.dmri.affine, header=dataset.dmri.header
        )
        nib.save(directions_nifti, str(fixel_dir / "directions.nii.gz"))

        D_intra_nifti = nib.Nifti2Image(
            fixel_data[0], affine=dataset.dmri.affine, header=dataset.dmri.header
        )
        nib.save(D_intra_nifti, str(fixel_dir / "D_intra.nii.gz"))

        f_intra_nifti = nib.Nifti2Image(
            fixel_data[1], affine=dataset.dmri.affine, header=dataset.dmri.header
        )
        nib.save(f_intra_nifti, str(fixel_dir / "f_intra.nii.gz"))

        # run.finish()
        del model, optimizer, dataset, train_dataloader, valid_dataloader, valid_pbar
        torch.cuda.empty_cache()

    if str(stage) in ["both", "2"]:
        config = {
            "loss": "gaussian",
            "sh_order_max": sh_order_max,
            "n_sh_coeffs": len(sph_harm_ind_list(sh_order_max)[0]),
            "max_fibers": max_fibers,
            "layer_size": layer_size,
            "num_enc": num_enc,
            "sigma": sigma,
            "pos_sigma": pos_sigma,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "plot_slice": plot_slice,
            "validate_every": validation_frequency,
            "validation_patience": validation_patience,
            "min_diffusivity": min_diffusivity,
            "min_f_intra": min_f_intra,
            "include_freewater": include_freewater,
            "D_fw": d_fw,
            "voxel_diffusivity": False,
            "device": device or ("cuda" if torch.cuda.is_available() else "cpu"),
            "peaks": str(Path(output_dir_stick) / "peaks.nii.gz"),
        }
        output_dir_sm = Path(output_dir) / "sm"
        output_dir_sm.mkdir(parents=True, exist_ok=True)
        fixel_dir = output_dir_sm / "fixel"
        fixel_dir.mkdir(exist_ok=True)

        dataset = DiffusionDataset(
            dmri_path=input_path,
            bval_path=bval_path,
            bvec_path=bvec_path,
            mask_path=mask_path,
            shells=shells,
            scale=True,
            extra_data={"peaks": config["peaks"]},
        )

        N = dataset.dmri_data.shape[-1]
        config["alpha"] = 0.5 * np.log(N)
        alpha = config["alpha"]

        if config["plot_slice"] is None:
            config["plot_slice"] = dataset.dmri_data.shape[plot_axis] // 2

        # Write config to sm/config.yaml for reproducibility
        with open(output_dir_sm / "config.yaml", "w") as f:
            yaml.dump(config, f)

        train_dataloader = DataLoader(
            dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0
        )
        valid_dataloader = DataLoader(
            dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0
        )
        signal_calculator = DiscreteStandardModelSignal(
            gtab=dataset.gtab,
        )

        sigma_tensor = torch.tensor(config["sigma"]).to(config["device"])

        model = DiscreteStandardModel(
            n_fibers=config["max_fibers"],
            layer_size=config["layer_size"],
            num_enc=config["num_enc"],
            sigma=config["pos_sigma"],
            freewater=config["include_freewater"],
        )

        model = model.to(config["device"])

        coords, diffusion_signal, extra_data = next(iter(train_dataloader))
        coords, diffusion_signal, dirs = (
            coords.to(config["device"]),
            diffusion_signal.to(config["device"]),
            extra_data["peaks"].to(config["device"]),
        )

        output = model(coords, peak_dir=dirs)
        (
            D_intra,
            D_extra_par,
            D_extra_perp,
            _,
            f_intra,
            f_extra,
            f_fw,
            S0,
            dirs_batch,
        ) = output

        # diffusivity_mask = D_intra >= config["min_diffusivity"]
        # D_intra = D_intra * diffusivity_mask
        # f_intra = f_intra * diffusivity_mask
        # f_intra = f_intra / (f_intra.sum(dim=-1, keepdim=True) + 1e-8)

        D_fw = torch.ones_like(D_intra[..., :1]) * config["D_fw"]
        D_fw = D_fw.to(D_intra.device)
        diffusion_signal_reconst = signal_calculator.compute_signal(
            d_intra=D_intra,
            d_extra_par=D_extra_par,
            d_extra_perp=D_extra_perp,
            f_intra=f_intra,
            f_extra=f_extra,
            f_fw=f_fw,
            dirs=dirs_batch,
            b0=S0.squeeze(-1),
            d_fw=D_fw,
        )

        loss_value = F.mse_loss(diffusion_signal_reconst, diffusion_signal)
        tqdm.write(f"Loss value: {loss_value.item()}")

        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        best_loss = np.inf
        last_step = 0
        n_since_last_improvement = 0

        # wandb.init(
        #     project="fiberfield",
        #     config=config,
        #     mode="online",
        #     name=config["name"],
        # )
        model.to(config["device"])
        tqdm.write(f"Training on device: {config['device']}")

        epoch_pbar = trange(
            config["num_epochs"], desc="Stage 2: Standard Model", leave=False
        )
        for epoch in epoch_pbar:
            epoch_pbar.set_postfix(loss=f"{loss_value.item():.4f}")
            model.train()
            for batch_num, (coords, diffusion_signal, extra_data) in enumerate(
                train_dataloader
            ):
                coords = coords.to(config["device"])
                diffusion_signal = diffusion_signal.to(config["device"])
                dirs = extra_data["peaks"].to(config["device"])

                output = model(coords, peak_dir=dirs)
                (
                    D_intra,
                    D_extra_par,
                    D_extra_perp,
                    _,
                    f_intra,
                    f_extra,
                    f_fw,
                    S0,
                    dirs,
                    *_,
                ) = output

                D_fw = torch.ones_like(D_intra[..., :1]) * config["D_fw"]
                D_fw = D_fw.to(D_intra.device)
                diffusion_signal_reconst = signal_calculator.compute_signal(
                    d_intra=D_intra,
                    d_extra_par=D_extra_par,
                    d_extra_perp=D_extra_perp,
                    f_intra=f_intra,
                    f_extra=f_extra,
                    f_fw=f_fw,
                    dirs=dirs,
                    b0=S0.squeeze(-1),
                    d_fw=D_fw,
                )

                loss_value = F.mse_loss(diffusion_signal_reconst, diffusion_signal)
                total_loss = loss_value.mean()

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # wandb.log(
                #     {
                #         "train/loss_map": total_loss.item(),
                #         "train/total_loss": total_loss.item(),
                #     },
                #     step=epoch * len(train_dataloader) + batch_num,
                # )
                last_step = epoch * len(train_dataloader) + batch_num

            if epoch % config["validate_every"] == 0:
                model.eval()

                with torch.no_grad():
                    D_intra_all = []
                    D_extra_par_all = []
                    D_extra_perp_all = []
                    D_fw_all = []
                    f_intra_all = []
                    f_extra_all = []
                    f_fw_all = []
                    diffusion_signal_all = []
                    diffusion_signal_reconst_all = []
                    S0_all = []
                    dirs_all = []

                    valid_pbar = tqdm(valid_dataloader, desc="Validation", leave=False)
                    for coords, diffusion_signal, extra_data in valid_pbar:
                        coords = coords.to(config["device"])
                        diffusion_signal = diffusion_signal.to(config["device"])
                        dirs = extra_data["peaks"].to(config["device"])

                        output = model(coords, peak_dir=dirs)
                        (
                            D_intra_batch_list,
                            D_extra_par_batch_list,
                            D_extra_perp_batch_list,
                            D_fw_batch_list,
                            f_intra_batch_list,
                            f_extra_batch_list,
                            f_fw_batch_list,
                            S0_batch_list,
                            diffusion_signal_batch_list,
                            dirs_batch_list,
                        ) = (
                            [],
                            [],
                            [],
                            [],
                            [],
                            [],
                            [],
                            [],
                            [],
                            [],
                        )
                        (
                            D_intra_batch,
                            D_extra_par_batch,
                            D_extra_perp_batch,
                            _,
                            f_intra_batch,
                            f_extra_batch,
                            f_fw_batch,
                            S0_batch,
                            dirs_batch,
                        ) = output

                        D_fw_batch = (
                            torch.ones_like(D_intra_batch[..., :1]) * config["D_fw"]
                        )
                        D_fw_batch = D_fw_batch.to(D_intra_batch.device)

                        diffusion_signal_batch_list.append(
                            signal_calculator.compute_signal(
                                d_intra=D_intra_batch,
                                d_extra_par=D_extra_par_batch,
                                d_extra_perp=D_extra_perp_batch,
                                f_intra=f_intra_batch,
                                f_extra=f_extra_batch,
                                f_fw=f_fw_batch,
                                dirs=dirs_batch,
                                b0=S0_batch.squeeze(-1),
                                d_fw=D_fw_batch,
                            )
                        )

                        # Pad everything to length config["max_fibers"]
                        D_intra_batch = torch.nn.functional.pad(
                            D_intra_batch,
                            (0, config["max_fibers"] - D_intra_batch.shape[-1]),
                        )
                        D_extra_par_batch = torch.nn.functional.pad(
                            D_extra_par_batch,
                            (0, config["max_fibers"] - D_extra_par_batch.shape[-1]),
                        )
                        D_extra_perp_batch = torch.nn.functional.pad(
                            D_extra_perp_batch,
                            (0, config["max_fibers"] - D_extra_perp_batch.shape[-1]),
                        )
                        f_intra_batch = torch.nn.functional.pad(
                            f_intra_batch,
                            (0, config["max_fibers"] - f_intra_batch.shape[-1]),
                        )
                        f_extra_batch = torch.nn.functional.pad(
                            f_extra_batch,
                            (0, config["max_fibers"] - f_extra_batch.shape[-1]),
                        )
                        dirs_batch = torch.nn.functional.pad(
                            dirs_batch,
                            (0, 0, 0, 0, 0, config["max_fibers"] - dirs_batch.shape[1]),
                        )

                        D_intra_batch_list.append(D_intra_batch)
                        D_extra_par_batch_list.append(D_extra_par_batch)
                        D_extra_perp_batch_list.append(D_extra_perp_batch)
                        D_fw_batch_list.append(D_fw_batch)
                        f_intra_batch_list.append(f_intra_batch)
                        f_extra_batch_list.append(f_extra_batch)
                        f_fw_batch_list.append(f_fw_batch)
                        S0_batch_list.append(S0_batch)
                        dirs_batch_list.append(dirs_batch)

                        D_intra_batch = torch.stack(D_intra_batch_list, dim=1)
                        D_extra_par_batch = torch.stack(D_extra_par_batch_list, dim=1)
                        D_extra_perp_batch = torch.stack(D_extra_perp_batch_list, dim=1)
                        D_fw_batch = torch.stack(D_fw_batch_list, dim=1)
                        f_intra_batch = torch.stack(f_intra_batch_list, dim=1)
                        f_extra_batch = torch.stack(f_extra_batch_list, dim=1)
                        f_fw_batch = torch.stack(f_fw_batch_list, dim=1)
                        S0_batch = torch.stack(S0_batch_list, dim=1)
                        dirs_batch = torch.stack(dirs_batch_list, dim=1)

                        D_intra_all.append(D_intra_batch.detach().cpu())
                        D_extra_par_all.append(D_extra_par_batch.detach().cpu())
                        D_extra_perp_all.append(D_extra_perp_batch.detach().cpu())
                        D_fw_all.append(D_fw_batch.detach().cpu())
                        f_intra_all.append(f_intra_batch.detach().cpu())
                        f_extra_all.append(f_extra_batch.detach().cpu())
                        f_fw_all.append(f_fw_batch.detach().cpu())
                        S0_all.append(S0_batch.detach().cpu())
                        diffusion_signal_all.append(diffusion_signal.detach().cpu())
                        diffusion_signal_reconst_all.append(
                            torch.stack(diffusion_signal_batch_list).detach().cpu()
                        )
                        dirs_all.append(dirs_batch.detach().cpu())

                    D_intra = torch.cat(D_intra_all, dim=0)
                    D_extra_par = torch.cat(D_extra_par_all, dim=0)
                    D_extra_perp = torch.cat(D_extra_perp_all, dim=0)
                    D_fw = torch.cat(D_fw_all, dim=0)
                    f_intra = torch.cat(f_intra_all, dim=0)
                    f_extra = torch.cat(f_extra_all, dim=0)
                    f_fw = torch.cat(f_fw_all, dim=0)
                    S0 = torch.cat(S0_all, dim=0)
                    diffusion_signal = torch.cat(diffusion_signal_all, dim=0)
                    diffusion_signal_reconst = torch.cat(
                        diffusion_signal_reconst_all, dim=1
                    )
                    dirs = torch.cat(dirs_all, dim=0)

                    loss_value = F.mse_loss(diffusion_signal_reconst, diffusion_signal)
                    total_loss = loss_value.mean()

                    val_loss_map = total_loss
                    val_total_loss = val_loss_map

                D_intra = D_intra[torch.arange(D_intra.shape[0]), 0]
                D_extra_par = D_extra_par[torch.arange(D_extra_par.shape[0]), 0]
                D_extra_perp = D_extra_perp[torch.arange(D_extra_perp.shape[0]), 0]
                D_fw = D_fw[torch.arange(D_fw.shape[0]), 0]
                f_intra = f_intra[torch.arange(f_intra.shape[0]), 0]
                f_extra = f_extra[torch.arange(f_extra.shape[0]), 0]
                f_fw = f_fw[torch.arange(f_fw.shape[0]), 0]
                S0 = S0[torch.arange(S0.shape[0]), 0]
                diffusion_signal_reconst = diffusion_signal_reconst[
                    0, torch.arange(diffusion_signal_reconst.shape[1]), :
                ]
                dirs = dirs[torch.arange(dirs.shape[0]), 0]

                # Sort by f_intra in descending order
                sorted_indices = torch.argsort(f_intra, dim=-1, descending=True)
                D_intra = torch.gather(D_intra, dim=-1, index=sorted_indices)
                D_extra_par = torch.gather(D_extra_par, dim=-1, index=sorted_indices)
                D_extra_perp = torch.gather(D_extra_perp, dim=-1, index=sorted_indices)
                f_intra = torch.gather(f_intra, dim=-1, index=sorted_indices)
                f_extra = torch.gather(f_extra, dim=-1, index=sorted_indices)

                dirs = torch.gather(
                    dirs,
                    dim=-2,
                    index=sorted_indices[..., None].expand(-1, -1, 3),
                )

                # Reshape
                D_intra_np = np.zeros(
                    dataset.dmri_data.shape[:3] + (config["max_fibers"],)
                )
                D_extra_par_np = np.zeros(
                    dataset.dmri_data.shape[:3] + (config["max_fibers"],)
                )
                D_extra_perp_np = np.zeros(
                    dataset.dmri_data.shape[:3] + (config["max_fibers"],)
                )
                D_fw_np = np.zeros(dataset.dmri_data.shape[:3])
                f_intra_np = np.zeros(
                    dataset.dmri_data.shape[:3] + (config["max_fibers"],)
                )
                f_extra_np = np.zeros(
                    dataset.dmri_data.shape[:3] + (config["max_fibers"],)
                )
                f_fw_np = np.zeros(dataset.dmri_data.shape[:3])
                S0_np = np.zeros(dataset.dmri_data.shape[:3])

                dirs_np = np.zeros(
                    dataset.dmri_data.shape[:3] + (config["max_fibers"], 3)
                )

                D_intra_np[dataset.mask_data.astype(bool), :] = (
                    D_intra.detach().cpu().numpy()
                )
                D_extra_par_np[dataset.mask_data.astype(bool), :] = (
                    D_extra_par.detach().cpu().numpy()
                )
                D_extra_perp_np[dataset.mask_data.astype(bool), :] = (
                    D_extra_perp.detach().cpu().numpy()
                )
                D_fw_np[dataset.mask_data.astype(bool)] = (
                    D_fw.detach().cpu().numpy().squeeze(-1)
                )
                f_intra_np[dataset.mask_data.astype(bool), :] = (
                    f_intra.detach().cpu().numpy()
                )
                f_extra_np[dataset.mask_data.astype(bool), :] = (
                    f_extra.detach().cpu().numpy()
                )
                f_fw_np[dataset.mask_data.astype(bool)] = (
                    f_fw.detach().cpu().numpy().squeeze(-1)
                )
                S0_np[dataset.mask_data.astype(bool)] = (
                    S0.detach().cpu().numpy().squeeze(-1)
                )
                dirs_np[dataset.mask_data.astype(bool), :] = dirs.detach().cpu().numpy()

                plot_discrete_sm_slice(
                    save_path=output_dir_sm / f"validation_epoch_{epoch}.png",
                    dirs_save_path=output_dir_sm / f"odfs_validation_epoch_{epoch}.png",
                    D_intra=D_intra_np,
                    D_extra_par=D_extra_par_np,
                    D_extra_perp=D_extra_perp_np,
                    D_fw=D_fw_np,
                    f_intra=f_intra_np,
                    f_extra=f_extra_np,
                    f_fw=f_fw_np,
                    dirs=dirs_np,
                    S0=S0_np,
                    bg_img=dataset.dmri,
                    signal_calculator=signal_calculator,
                    plot_slice=config["plot_slice"],
                    plot_vol=0,
                    axis=plot_axis,
                )

                if val_total_loss.item() < best_loss:
                    best_loss = val_total_loss.item()
                    torch.save(model.state_dict(), output_dir_sm / "best_model.pth")
                    n_since_last_improvement = 0
                else:
                    n_since_last_improvement += 1

                if n_since_last_improvement >= config["validation_patience"]:
                    tqdm.write("Early stopping triggered.")
                    break

            if n_since_last_improvement >= config["validation_patience"]:
                break

        # Load best model
        model.load_state_dict(torch.load(output_dir_sm / "best_model.pth"))
        model.eval()

        with torch.no_grad():
            D_intra_all = []
            D_extra_par_all = []
            D_extra_perp_all = []
            D_fw_all = []
            f_intra_all = []
            f_extra_all = []
            f_fw_all = []
            diffusion_signal_all = []
            diffusion_signal_reconst_all = []
            S0_all = []
            dirs_all = []

            for coords, diffusion_signal, extra_data in valid_pbar:
                coords = coords.to(config["device"])
                diffusion_signal = diffusion_signal.to(config["device"])
                dirs = extra_data["peaks"].to(config["device"])

                output = model(coords, peak_dir=dirs)
                (
                    D_intra_batch_list,
                    D_extra_par_batch_list,
                    D_extra_perp_batch_list,
                    D_fw_batch_list,
                    f_intra_batch_list,
                    f_extra_batch_list,
                    f_fw_batch_list,
                    S0_batch_list,
                    diffusion_signal_batch_list,
                    dirs_batch_list,
                ) = [], [], [], [], [], [], [], [], [], []
                (
                    D_intra_batch,
                    D_extra_par_batch,
                    D_extra_perp_batch,
                    _,
                    f_intra_batch,
                    f_extra_batch,
                    f_fw_batch,
                    S0_batch,
                    dirs_batch,
                ) = output

                diffusivity_mask = D_intra_batch >= config["min_diffusivity"]
                fraction_mask = f_intra_batch >= config["min_f_intra"]
                output_mask = diffusivity_mask * fraction_mask
                D_intra_batch = D_intra_batch * output_mask
                D_extra_par_batch = D_extra_par_batch * output_mask
                D_extra_perp_batch = D_extra_perp_batch * output_mask
                f_intra_batch = f_intra_batch * output_mask
                f_extra_batch = f_extra_batch * output_mask
                # f_fw_batch = f_fw_batch * output_mask

                dirs_batch = dirs_batch * output_mask[..., None]

                f_all_batch = torch.cat(
                    [f_intra_batch, f_extra_batch, f_fw_batch], dim=-1
                )
                f_all_batch = f_all_batch / (
                    f_all_batch.sum(dim=-1, keepdim=True) + 1e-8
                )
                f_intra_batch, f_extra_batch, f_fw_batch = torch.split(
                    f_all_batch,
                    [
                        f_intra_batch.shape[-1],
                        f_extra_batch.shape[-1],
                        f_fw_batch.shape[-1],
                    ],
                    dim=-1,
                )

                D_fw_batch = torch.ones_like(D_intra_batch[..., :1]) * config["D_fw"]
                D_fw_batch = D_fw_batch.to(D_intra_batch.device)

                diffusion_signal_batch_list.append(
                    signal_calculator.compute_signal(
                        d_intra=D_intra_batch,
                        d_extra_par=D_extra_par_batch,
                        d_extra_perp=D_extra_perp_batch,
                        f_intra=f_intra_batch,
                        f_extra=f_extra_batch,
                        f_fw=f_fw_batch,
                        dirs=dirs_batch,
                        b0=S0_batch.squeeze(-1),
                        d_fw=D_fw_batch,
                    )
                )

                # Pad everything to length config["max_fibers"]
                D_intra_batch = torch.nn.functional.pad(
                    D_intra_batch, (0, config["max_fibers"] - D_intra_batch.shape[-1])
                )
                D_extra_par_batch = torch.nn.functional.pad(
                    D_extra_par_batch,
                    (0, config["max_fibers"] - D_extra_par_batch.shape[-1]),
                )
                D_extra_perp_batch = torch.nn.functional.pad(
                    D_extra_perp_batch,
                    (0, config["max_fibers"] - D_extra_perp_batch.shape[-1]),
                )
                f_intra_batch = torch.nn.functional.pad(
                    f_intra_batch, (0, config["max_fibers"] - f_intra_batch.shape[-1])
                )
                f_extra_batch = torch.nn.functional.pad(
                    f_extra_batch, (0, config["max_fibers"] - f_extra_batch.shape[-1])
                )

                dirs_batch = torch.nn.functional.pad(
                    dirs_batch,
                    (0, 0, 0, config["max_fibers"] - dirs_batch.shape[1]),
                )

                D_intra_batch_list.append(D_intra_batch)
                D_extra_par_batch_list.append(D_extra_par_batch)
                D_extra_perp_batch_list.append(D_extra_perp_batch)
                D_fw_batch_list.append(D_fw_batch)
                f_intra_batch_list.append(f_intra_batch)
                f_extra_batch_list.append(f_extra_batch)
                f_fw_batch_list.append(f_fw_batch)
                S0_batch_list.append(S0_batch.detach().cpu())
                dirs_batch_list.append(dirs_batch)

                D_intra_batch = torch.stack(D_intra_batch_list, dim=1)
                D_extra_par_batch = torch.stack(D_extra_par_batch_list, dim=1)
                D_extra_perp_batch = torch.stack(D_extra_perp_batch_list, dim=1)
                D_fw_batch = torch.stack(D_fw_batch_list, dim=1)
                f_intra_batch = torch.stack(f_intra_batch_list, dim=1)
                f_extra_batch = torch.stack(f_extra_batch_list, dim=1)
                f_fw_batch = torch.stack(f_fw_batch_list, dim=1)
                S0_batch = torch.stack(S0_batch_list, dim=1)
                dirs_batch = torch.stack(dirs_batch_list, dim=1)

                D_intra_all.append(D_intra_batch.detach().cpu())
                D_extra_par_all.append(D_extra_par_batch.detach().cpu())
                D_extra_perp_all.append(D_extra_perp_batch.detach().cpu())
                D_fw_all.append(D_fw_batch.detach().cpu())
                f_intra_all.append(f_intra_batch.detach().cpu())
                f_extra_all.append(f_extra_batch.detach().cpu())
                f_fw_all.append(f_fw_batch.detach().cpu())
                S0_all.append(S0_batch.detach().cpu())
                diffusion_signal_all.append(diffusion_signal.detach().cpu())
                diffusion_signal_reconst_all.append(
                    torch.stack(diffusion_signal_batch_list).detach().cpu()
                )
                dirs_all.append(dirs_batch.detach().cpu())

            D_intra = torch.cat(D_intra_all, dim=0)
            D_extra_par = torch.cat(D_extra_par_all, dim=0)
            D_extra_perp = torch.cat(D_extra_perp_all, dim=0)
            D_fw = torch.cat(D_fw_all, dim=0)
            f_intra = torch.cat(f_intra_all, dim=0)
            f_extra = torch.cat(f_extra_all, dim=0)
            f_fw = torch.cat(f_fw_all, dim=0)
            S0 = torch.cat(S0_all, dim=0)
            diffusion_signal = torch.cat(diffusion_signal_all, dim=0)
            diffusion_signal_reconst = torch.cat(diffusion_signal_reconst_all, dim=1)
            dirs = torch.cat(dirs_all, dim=0)

        D_intra = D_intra[torch.arange(D_intra.shape[0]), 0]
        D_extra_par = D_extra_par[torch.arange(D_extra_par.shape[0]), 0]
        D_extra_perp = D_extra_perp[torch.arange(D_extra_perp.shape[0]), 0]
        D_fw = D_fw[torch.arange(D_fw.shape[0]), 0]
        f_intra = f_intra[torch.arange(f_intra.shape[0]), 0]
        f_extra = f_extra[torch.arange(f_extra.shape[0]), 0]
        f_fw = f_fw[torch.arange(f_fw.shape[0]), 0]
        S0 = S0[torch.arange(S0.shape[0]), 0]
        dirs = dirs[torch.arange(dirs.shape[0]), 0]
        diffusion_signal_reconst = diffusion_signal_reconst[
            0, torch.arange(diffusion_signal_reconst.shape[1]), :
        ]

        # Sort everything by volume fraction within each voxel
        sorted_indices = torch.argsort(f_intra, dim=-1, descending=True)
        D_intra = torch.gather(D_intra, dim=-1, index=sorted_indices)
        D_extra_par = torch.gather(D_extra_par, dim=-1, index=sorted_indices)
        D_extra_perp = torch.gather(D_extra_perp, dim=-1, index=sorted_indices)
        f_intra = torch.gather(f_intra, dim=-1, index=sorted_indices)
        f_extra = torch.gather(f_extra, dim=-1, index=sorted_indices)
        dirs = torch.gather(
            dirs, dim=-2, index=sorted_indices.unsqueeze(-1).expand(-1, -1, 3)
        )

        # Save coefficients
        D_intra_np = np.zeros(dataset.dmri_data.shape[:3] + (config["max_fibers"],))
        D_extra_par_np = np.zeros(dataset.dmri_data.shape[:3] + (config["max_fibers"],))
        D_extra_perp_np = np.zeros(
            dataset.dmri_data.shape[:3] + (config["max_fibers"],)
        )
        D_fw_np = np.zeros(dataset.dmri_data.shape[:3])
        f_intra_np = np.zeros(dataset.dmri_data.shape[:3] + (config["max_fibers"],))
        f_extra_np = np.zeros(dataset.dmri_data.shape[:3] + (config["max_fibers"],))
        f_fw_np = np.zeros(dataset.dmri_data.shape[:3])
        S0_np = np.zeros(dataset.dmri_data.shape[:3])

        dirs_np = np.zeros(dataset.dmri_data.shape[:3] + (config["max_fibers"], 3))
        D_intra_np[dataset.mask_data.astype(bool), :] = D_intra.detach().cpu().numpy()
        D_extra_par_np[dataset.mask_data.astype(bool), :] = (
            D_extra_par.detach().cpu().numpy()
        )
        D_extra_perp_np[dataset.mask_data.astype(bool), :] = (
            D_extra_perp.detach().cpu().numpy()
        )
        D_fw_np[dataset.mask_data.astype(bool)] = (
            D_fw.detach().cpu().numpy().squeeze(-1)
        )
        f_intra_np[dataset.mask_data.astype(bool), :] = f_intra.detach().cpu().numpy()
        f_extra_np[dataset.mask_data.astype(bool), :] = f_extra.detach().cpu().numpy()
        f_fw_np[dataset.mask_data.astype(bool)] = (
            f_fw.detach().cpu().numpy().squeeze(-1)
        )
        S0_np[dataset.mask_data.astype(bool)] = S0.detach().cpu().numpy().squeeze(-1)
        dirs_np[dataset.mask_data.astype(bool), :] = dirs.detach().cpu().numpy()

        D_intra_img = nib.Nifti1Image(
            D_intra_np, affine=dataset.dmri.affine, header=dataset.dmri.header
        )
        D_extra_par_img = nib.Nifti1Image(
            D_extra_par_np, affine=dataset.dmri.affine, header=dataset.dmri.header
        )
        D_extra_perp_img = nib.Nifti1Image(
            D_extra_perp_np, affine=dataset.dmri.affine, header=dataset.dmri.header
        )
        D_fw_img = nib.Nifti1Image(
            D_fw_np, affine=dataset.dmri.affine, header=dataset.dmri.header
        )
        f_intra_img = nib.Nifti1Image(
            f_intra_np, affine=dataset.dmri.affine, header=dataset.dmri.header
        )
        f_extra_img = nib.Nifti1Image(
            f_extra_np, affine=dataset.dmri.affine, header=dataset.dmri.header
        )
        f_fw_img = nib.Nifti1Image(
            f_fw_np, affine=dataset.dmri.affine, header=dataset.dmri.header
        )
        S0_img = nib.Nifti1Image(
            S0_np, affine=dataset.dmri.affine, header=dataset.dmri.header
        )
        dirs_img = nib.Nifti1Image(
            dirs_np, affine=dataset.dmri.affine, header=dataset.dmri.header
        )

        nib.save(D_intra_img, str(output_dir_sm / "D_intra.nii.gz"))
        nib.save(D_extra_par_img, str(output_dir_sm / "D_extra_par.nii.gz"))
        nib.save(D_extra_perp_img, str(output_dir_sm / "D_extra_perp.nii.gz"))
        nib.save(f_intra_img, str(output_dir_sm / "f_intra.nii.gz"))
        nib.save(f_extra_img, str(output_dir_sm / "f_extra.nii.gz"))
        nib.save(S0_img, str(output_dir_sm / "S0.nii.gz"))
        nib.save(dirs_img, str(output_dir_sm / "peaks.nii.gz"))

        if config["include_freewater"]:
            nib.save(f_fw_img, str(output_dir_sm / "f_fw.nii.gz"))
            nib.save(D_fw_img, str(output_dir_sm / "D_fw.nii.gz"))

        for fiber_idx in range(config["max_fibers"]):
            fiber_dir = dirs_np[..., fiber_idx, :]
            fiber_dir_img = nib.Nifti1Image(
                fiber_dir, affine=dataset.dmri.affine, header=dataset.dmri.header
            )
            nib.save(
                fiber_dir_img, str(output_dir_sm / f"peaks_{fiber_idx + 1}.nii.gz")
            )

        diffusion_signal_reconst_np = np.zeros(dataset.dmri_data.shape)
        diffusion_signal_reconst_np[dataset.mask_data.astype(bool)] = (
            diffusion_signal_reconst.detach().cpu().numpy()
        )
        diffusion_signal_reconst_img = nib.Nifti1Image(
            diffusion_signal_reconst_np,
            affine=dataset.dmri.affine,
            header=dataset.dmri.header,
        )
        nib.save(
            diffusion_signal_reconst_img,
            str(output_dir_sm / "diffusion_signal_reconst.nii.gz"),
        )

        # Also save in MRtrix3 fixel format
        index, directions, fixel_data = sparse_to_fixel_format(
            peaks=dirs_np,
            sparse_data=[
                D_intra_np,
                D_extra_par_np,
                D_extra_perp_np,
                f_intra_np,
                f_extra_np,
            ],
        )

        index_nifti = nib.Nifti2Image(
            index, affine=dataset.dmri.affine, header=dataset.dmri.header
        )
        nib.save(index_nifti, str(fixel_dir / "index.nii.gz"))

        directions_nifti = nib.Nifti2Image(
            directions, affine=dataset.dmri.affine, header=dataset.dmri.header
        )
        nib.save(directions_nifti, str(fixel_dir / "directions.nii.gz"))

        D_intra_nifti = nib.Nifti2Image(
            fixel_data[0], affine=dataset.dmri.affine, header=dataset.dmri.header
        )
        nib.save(D_intra_nifti, str(fixel_dir / "D_intra.nii.gz"))

        D_extra_par_nifti = nib.Nifti2Image(
            fixel_data[1], affine=dataset.dmri.affine, header=dataset.dmri.header
        )
        nib.save(D_extra_par_nifti, str(fixel_dir / "D_extra_par.nii.gz"))

        D_extra_perp_nifti = nib.Nifti2Image(
            fixel_data[2], affine=dataset.dmri.affine, header=dataset.dmri.header
        )
        nib.save(D_extra_perp_nifti, str(fixel_dir / "D_extra_perp.nii.gz"))

        f_intra_nifti = nib.Nifti2Image(
            fixel_data[3], affine=dataset.dmri.affine, header=dataset.dmri.header
        )
        nib.save(f_intra_nifti, str(fixel_dir / "f_intra.nii.gz"))

        f_extra_nifti = nib.Nifti2Image(
            fixel_data[4], affine=dataset.dmri.affine, header=dataset.dmri.header
        )
        nib.save(f_extra_nifti, str(fixel_dir / "f_extra.nii.gz"))

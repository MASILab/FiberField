from fiberfield.datasets import DiffusionDataset
from fiberfield.models import StickMultiheadModel
from fiberfield.signal import StickSignal
from fiberfield.utils import plot_stick_slice, sparse_to_fixel_format
import wandb

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import nibabel as nib

import numpy as np
from tqdm import tqdm, trange
from pathlib import Path


def run_stick(
    input_path,
    bval_path,
    bvec_path,
    mask_path,
    output_dir,
    max_fibers=3,
    layer_size=2048,
    num_enc=5000,
    sigma=4,
    alpha=0.002,
    batch_size=500,
    learning_rate=1e-4,
    num_epochs=1000,
    plot_slice=None,
    plot_axis=2,
    validation_frequency=10,
    validation_patience=25,
    min_diffusivity=1e-4,
    min_f_intra=1e-4,
    device=None,
):
    config = {
        "max_fibers": max_fibers,
        "layer_size": layer_size,
        "num_enc": num_enc,
        "sigma": sigma,
        "alpha": alpha,
        "p_k": [0, 1, 2],
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "plot_slice": plot_slice,
        "validate_every": validation_frequency,
        "validation_patience": validation_patience,
        "min_diffusivity": min_diffusivity,
        "min_f_intra": min_f_intra,
        "device": device or ("cuda" if torch.cuda.is_available() else "cpu"),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixel_dir = output_dir / "fixel"
    fixel_dir.mkdir(exist_ok=True)

    dataset = DiffusionDataset(
        dmri_path=input_path,
        bval_path=bval_path,
        bvec_path=bvec_path,
        mask_path=mask_path,
        scale=True,
    )

    if config["plot_slice"] is None:
        config["plot_slice"] = dataset.dmri_data.shape[plot_axis] // 2

    train_dataloader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4
    )
    valid_dataloader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )
    signal_calculator = StickSignal(
        gtab=dataset.gtab,
    )

    p_k = torch.tensor(config["p_k"])
    alpha = torch.tensor(config["alpha"])
    sigma = torch.tensor(config["sigma"])

    model = StickMultiheadModel(
        max_fibers=config["max_fibers"],
        layer_size=config["layer_size"],
        num_enc=config["num_enc"],
        sigma=config["sigma"],
    )

    coords, diffusion_signal = next(iter(train_dataloader))
    outputs = model(coords)
    diffusion_signal_reconst_list = []
    for k in range(len(outputs)):
        D_intra, f_intra, S0, dirs = outputs[k + 1]

        # diffusivity_mask = D_intra >= config["min_diffusivity"]
        # D_intra = D_intra * diffusivity_mask
        # f_intra = f_intra * diffusivity_mask
        # f_intra = f_intra / (f_intra.sum(dim=-1, keepdim=True) + 1e-8)

        diffusion_signal_reconst_list.append(
            signal_calculator.compute_signal(
                d_intra=D_intra,
                f_intra=f_intra,
                dirs=dirs,
                b0=S0.squeeze(-1),
            )
        )
    diffusion_signal_reconst = torch.stack(diffusion_signal_reconst_list)

    mse_k = ((diffusion_signal_reconst - diffusion_signal.unsqueeze(0)) ** 2).mean(
        dim=-1
    )
    loss_value = -torch.logsumexp(
        -mse_k - alpha * p_k.to(mse_k.device)[:, None], dim=0
    ).mean()
    tqdm.write(f"Loss value: {loss_value.item()}")

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    best_loss = np.inf
    last_step = 0
    n_since_last_improvement = 0

    # wandb.init(
    #     project="diffusion_neural_field",
    #     config=config,
    #     mode="online",
    #     name=config["name"],
    # )
    model.to(config["device"])
    tqdm.write(f"Training on device: {config['device']}")

    epoch_pbar = trange(config["num_epochs"], desc="Training", leave=False)
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

                # diffusivity_mask = D_intra >= config["min_diffusivity"]
                # D_intra = D_intra * diffusivity_mask
                # f_intra = f_intra * diffusivity_mask
                # f_intra = f_intra / (f_intra.sum(dim=-1, keepdim=True) + 1e-8)

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
            mse_k = (
                (diffusion_signal_reconst - diffusion_signal.unsqueeze(0)) ** 2
            ).mean(dim=-1)
            loss_value = -torch.logsumexp(
                -mse_k - alpha * p_k.to(mse_k.device)[:, None], dim=0
            ).mean()
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

                mse_k = (
                    (diffusion_signal_reconst - diffusion_signal.unsqueeze(0)) ** 2
                ).mean(dim=-1)
                loss_value = -torch.logsumexp(
                    -mse_k - alpha * p_k.to(mse_k.device)[:, None], dim=0
                ).mean()
                total_loss = loss_value.mean()

                val_loss_map = total_loss
                val_total_loss = val_loss_map

            # Select arg min over inner loss
            mse_k = (
                (diffusion_signal_reconst - diffusion_signal.unsqueeze(0)) ** 2
            ).mean(dim=-1)
            mse_k = mse_k + alpha * p_k.to(mse_k.device)[:, None]
            best_k = torch.argmin(mse_k, dim=0)
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
            D_intra_np = np.zeros(dataset.dmri_data.shape[:3] + (config["max_fibers"],))
            f_intra_np = np.zeros(dataset.dmri_data.shape[:3] + (config["max_fibers"],))
            S0_np = np.zeros(dataset.dmri_data.shape[:3])
            dirs_np = np.zeros(dataset.dmri_data.shape[:3] + (config["max_fibers"], 3))
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
                save_path=output_dir / f"validation_epoch_{epoch}.png",
                dirs_save_path=output_dir / f"peaks_validation_epoch_{epoch}.png",
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
            #             str(output_dir / f"validation_epoch_{epoch}.png")
            #         ),
            #         "valid/odf_image": wandb.Image(
            #             str(output_dir / f"peaks_validation_epoch_{epoch}.png")
            #         ),
            #         "valid/loss_map": val_loss_map.item(),
            #         "valid/total_loss": val_total_loss.item(),
            #     },
            #     step=last_step + 1,
            # )

            if val_total_loss.item() < best_loss:
                best_loss = val_total_loss.item()
                torch.save(model.state_dict(), output_dir / "best_model.pth")
                n_since_last_improvement = 0
            else:
                n_since_last_improvement += 1

            if n_since_last_improvement >= config["validation_patience"]:
                tqdm.write("Early stopping triggered.")
                break

        if n_since_last_improvement >= config["validation_patience"]:
            break

    # Load best model
    model.load_state_dict(torch.load(output_dir / "best_model.pth"))
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
                f_intra_batch = f_intra_batch / (
                    f_intra_batch.sum(dim=-1, keepdim=True) + 1e-8
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
    mse_k = ((diffusion_signal_reconst - diffusion_signal.unsqueeze(0)) ** 2).mean(
        dim=-1
    )
    mse_k = mse_k + alpha * p_k.to(mse_k.device)[:, None]
    best_k = torch.argmin(mse_k, dim=0)
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

    nib.save(D_intra_img, str(output_dir / "D_intra.nii.gz"))
    nib.save(f_intra_img, str(output_dir / "f_intra.nii.gz"))
    nib.save(S0_img, str(output_dir / "S0.nii.gz"))

    dirs_np = np.zeros(dataset.dmri_data.shape[:3] + (config["max_fibers"], 3))
    dirs_np[dataset.mask_data.astype(bool), :] = dirs.detach().cpu().numpy()
    dirs_img = nib.Nifti1Image(
        dirs_np, affine=dataset.dmri.affine, header=dataset.dmri.header
    )
    nib.save(dirs_img, str(output_dir / "peaks.nii.gz"))

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
        str(output_dir / "diffusion_signal_reconst.nii.gz"),
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

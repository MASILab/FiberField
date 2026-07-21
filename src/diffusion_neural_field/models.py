import torch
import torch.nn as nn
import torch.nn.functional as F

from dipy.reconst.shm import sph_harm_ind_list


class PositionalEncoding(nn.Module):
    def __init__(self, num_enc=5000, dim=3, sigma=3.5):
        super(PositionalEncoding, self).__init__()
        self.num_enc = num_enc
        self.dim = dim
        self.sigma = sigma

        # Create frequency bands
        self.B = nn.Parameter(
            torch.randn((self.num_enc // 2, self.dim)) * self.sigma, requires_grad=False
        )

    def forward(self, x):
        x_proj = (2.0 * torch.pi * x) @ self.B.T
        encodings = [torch.cos(x_proj), torch.sin(x_proj)]
        return torch.cat(encodings, dim=-1)


class SingleShellCSD(nn.Module):
    def __init__(self, sh_order_max=6, layer_size=2048, num_enc=5000, sigma=3.5):
        super(SingleShellCSD, self).__init__()
        self.sh_order_max = sh_order_max
        m_list, l_list = sph_harm_ind_list(
            sh_order_max=self.sh_order_max, full_basis=False
        )
        self.num_sh_coeffs = len(m_list)

        self.pos_enc = PositionalEncoding(num_enc=num_enc, dim=3, sigma=sigma)
        self.mlp = nn.Sequential(
            nn.Linear(num_enc, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, self.num_sh_coeffs),
        )

    def forward(self, x):
        x_enc = self.pos_enc(x)
        sh_coeffs = self.mlp(x_enc)
        return sh_coeffs


class StandardModel(nn.Module):
    def __init__(
        self,
        sh_order_max=2,
        layer_size=2048,
        num_enc=5000,
        sigma=3.5,
        include_freewater=False,
    ):
        super(StandardModel, self).__init__()
        self.sh_order_max = sh_order_max
        m_list, l_list = sph_harm_ind_list(
            sh_order_max=self.sh_order_max, full_basis=False
        )
        self.num_sh_coeffs = len(m_list)
        self.include_freewater = include_freewater
        self.pos_enc = PositionalEncoding(num_enc=num_enc, dim=3, sigma=sigma)
        self.mlp = nn.Sequential(
            nn.Linear(num_enc, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
        )
        self.D_intra_head = nn.Sequential(nn.Linear(layer_size, 1), nn.Sigmoid())
        self.D_extra_head = nn.Sequential(nn.Linear(layer_size, 1), nn.Sigmoid())
        self.D_perp_head = nn.Sequential(nn.Linear(layer_size, 1), nn.Sigmoid())
        self.f_intra_head = nn.Sequential(nn.Linear(layer_size, 1), nn.Sigmoid())
        self.S0_head = nn.Sequential(nn.Linear(layer_size, 1), nn.Softplus())
        self.fod_head = nn.Linear(layer_size, self.num_sh_coeffs - 1)

        if self.include_freewater:
            self.D_fw_head = nn.Sequential(nn.Linear(layer_size, 1), nn.Sigmoid())
            self.f_fw_head = nn.Sequential(nn.Linear(layer_size, 1), nn.Sigmoid())

    def forward(self, x):
        x_enc = self.pos_enc(x)
        z = self.mlp(x_enc)
        D_intra = self.D_intra_head(z) * 4e-3
        D_extra = self.D_extra_head(z) * 4e-3
        D_perp = self.D_perp_head(z) * 1.5e-3
        f_intra = self.f_intra_head(z)
        S0 = self.S0_head(z)
        fod_sh_coeffs = self.fod_head(z)
        # Ensure FOD scaling by setting first coefficient
        p00 = torch.ones(x.shape[0], 1) / torch.sqrt(torch.tensor(4.0 * torch.pi))
        p00 = p00.to(x.device)
        sh_coeffs = torch.cat((p00, fod_sh_coeffs), dim=-1)
        if self.include_freewater:
            f_fw = self.f_fw_head(z)
            D_fw = self.D_fw_head(z) * 4.0e-3
            return sh_coeffs, D_intra, D_extra, D_perp, D_fw, f_intra, f_fw, S0
        else:
            return sh_coeffs, D_intra, D_extra, D_perp, f_intra, S0


class StickModel(nn.Module):
    def __init__(self, n_fibers=1, layer_size=2048, num_enc=5000, sigma=3.5):
        super(StickModel, self).__init__()
        self.n_fibers = n_fibers
        self.pos_enc = PositionalEncoding(num_enc=num_enc, dim=3, sigma=sigma)
        self.mlp = nn.Sequential(
            nn.Linear(num_enc, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
        )
        self.D_intra_head = nn.Sequential(
            nn.Linear(layer_size, self.n_fibers), nn.Sigmoid()
        )
        self.f_intra_head = nn.Sequential(
            nn.Linear(layer_size, self.n_fibers), nn.Softmax(dim=-1)
        )
        self.S0_head = nn.Sequential(nn.Linear(layer_size, 1), nn.Softplus())
        self.dir_head = nn.Linear(layer_size, self.n_fibers * 3)

    def forward(self, x):
        x_enc = self.pos_enc(x)
        z = self.mlp(x_enc)
        D_intra = self.D_intra_head(z) * 4e-3
        f_intra = self.f_intra_head(z)
        S0 = self.S0_head(z)
        dirs = self.dir_head(z).view(-1, self.n_fibers, 3)
        dirs = F.normalize(dirs, dim=-1)
        return D_intra, f_intra, S0, dirs


class StickHead(nn.Module):
    def __init__(
        self,
        n_fibers=1,
        layer_size=2048,
        include_b0=True,
        min_diffusivity=0,
        voxel_diffusivity=False,
    ):
        super(StickHead, self).__init__()
        self.include_b0 = include_b0
        self.n_fibers = n_fibers
        self.voxel_diffusivity = voxel_diffusivity
        if self.voxel_diffusivity:
            self.D_intra_head = nn.Sequential(nn.Linear(layer_size, 1), nn.Sigmoid())
        else:
            self.D_intra_head = nn.Sequential(
                nn.Linear(layer_size, self.n_fibers), nn.Sigmoid()
            )
        self.f_intra_head = nn.Sequential(
            nn.Linear(layer_size, self.n_fibers), nn.Softmax(dim=-1)
        )
        self.S0_head = nn.Sequential(nn.Linear(layer_size, 1), nn.Softplus())
        self.dir_head = nn.Linear(layer_size, self.n_fibers * 3)
        self.min_diffusivity = min_diffusivity

    def forward(self, z):
        D_intra = (
            self.D_intra_head(z) * (4e-3 - self.min_diffusivity) + self.min_diffusivity
        )
        if self.voxel_diffusivity:
            D_intra = D_intra.repeat(1, self.n_fibers)
        f_intra = self.f_intra_head(z)
        S0 = self.S0_head(z)
        dirs = self.dir_head(z).view(-1, self.n_fibers, 3)
        dirs = F.normalize(dirs, dim=-1)
        if self.include_b0:
            return D_intra, f_intra, S0, dirs
        else:
            return D_intra, f_intra, dirs


class StickMultiheadModel(nn.Module):
    def __init__(
        self,
        max_fibers=3,
        layer_size=2048,
        num_enc=5000,
        sigma=3.5,
        include_b0=True,
        min_diffusivity=0,
        voxel_diffusivity=False,
    ):
        super(StickMultiheadModel, self).__init__()
        self.max_fibers = max_fibers
        self.pos_enc = PositionalEncoding(num_enc=num_enc, dim=3, sigma=sigma)
        self.mlp = nn.Sequential(
            nn.Linear(num_enc, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList(
            [
                StickHead(
                    n_fibers=k + 1,
                    layer_size=layer_size,
                    include_b0=include_b0,
                    min_diffusivity=min_diffusivity,
                    voxel_diffusivity=voxel_diffusivity,
                )
                for k in range(max_fibers)
            ]
        )

    def forward(self, x):
        x_enc = self.pos_enc(x)
        z = self.mlp(x_enc)
        return {k + 1: head(z) for k, head in enumerate(self.heads)}


class DiscreteStandardModel(nn.Module):
    def __init__(
        self, n_fibers=1, layer_size=2048, num_enc=5000, sigma=3.5, freewater=False
    ):
        super(DiscreteStandardModel, self).__init__()
        self.n_fibers = n_fibers
        self.freewater = freewater
        self.pos_enc = PositionalEncoding(num_enc=num_enc, dim=3, sigma=sigma)
        self.mlp = nn.Sequential(
            nn.Linear(num_enc, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
        )
        self.D_intra_head = nn.Sequential(
            nn.Linear(layer_size, self.n_fibers), nn.Sigmoid()
        )
        self.D_extra_par_head = nn.Sequential(
            nn.Linear(layer_size, self.n_fibers), nn.Sigmoid()
        )
        self.D_extra_perp_head = nn.Sequential(
            nn.Linear(layer_size, self.n_fibers), nn.Sigmoid()
        )
        self.D_fw_head = nn.Sequential(nn.Linear(layer_size, 1), nn.Sigmoid())
        n_fractions = self.n_fibers * 2 + 1 if self.freewater else self.n_fibers * 2
        self.f_head = nn.Sequential(
            nn.Linear(layer_size, n_fractions), nn.Softmax(dim=-1)
        )
        self.S0_head = nn.Sequential(nn.Linear(layer_size, 1), nn.Softplus())
        self.dir_head = nn.Linear(layer_size, self.n_fibers * 3)

    def forward(self, x):
        x_enc = self.pos_enc(x)
        z = self.mlp(x_enc)
        D_intra = self.D_intra_head(z) * 4e-3
        D_extra_par = self.D_extra_par_head(z) * 4e-3
        D_extra_perp = self.D_extra_perp_head(z) * 1.5e-3
        D_fw = (
            self.D_fw_head(z) * 4.0e-3
            if self.freewater
            else torch.zeros_like(D_intra[:, :1]).to(x.device)
        )
        f = self.f_head(z)
        f_intra = f[:, : self.n_fibers]
        f_extra = f[:, self.n_fibers : 2 * self.n_fibers]
        f_fw = (
            f[:, -1:]
            if self.freewater
            else torch.zeros_like(f_intra[:, :1]).to(x.device)
        )
        S0 = self.S0_head(z)
        dirs = self.dir_head(z).view(-1, self.n_fibers, 3)
        dirs = F.normalize(dirs, dim=-1)
        return (
            D_intra,
            D_extra_par,
            D_extra_perp,
            D_fw,
            f_intra,
            f_extra,
            f_fw,
            S0,
            dirs,
        )


class DiscreteStandardModelHead(nn.Module):
    def __init__(
        self,
        n_fibers=1,
        layer_size=2048,
        include_b0=True,
        include_freewater=False,
        min_diffusivity=0,
        voxel_diffusivity=False,
    ):
        """Predicts discrete Standard Model parameters for a given number of fibers.

        Parameters
        ----------
        n_fibers : int, optional
            Fixed number of fibers, by default 1
        layer_size : int, optional
            Size of the hidden layers, by default 2048
        include_b0 : bool, optional
            Whether to include the b0 signal in the output, by default True
        include_freewater : bool, optional
            Whether to include the freewater signal in the output, by default False
        min_diffusivity : int, optional
            Minimum diffusivity value, by default 0
        voxel_diffusivity : bool, optional
            If "extra", just extra-axonal diffusivities are voxel-wise, if True,
            all diffusivities are voxel-wise, by default False
        """
        super(DiscreteStandardModelHead, self).__init__()
        self.include_b0 = include_b0
        self.include_freewater = include_freewater
        self.n_fibers = n_fibers
        self.voxel_diffusivity = voxel_diffusivity
        if self.voxel_diffusivity == "extra":
            self.D_intra_head = nn.Sequential(
                nn.Linear(layer_size, self.n_fibers), nn.Sigmoid()
            )
            self.D_extra_par_head = nn.Sequential(
                nn.Linear(layer_size, 1), nn.Sigmoid()
            )
            self.D_extra_perp_head = nn.Sequential(
                nn.Linear(layer_size, 1), nn.Sigmoid()
            )
        elif self.voxel_diffusivity:
            self.D_intra_head = nn.Sequential(nn.Linear(layer_size, 1), nn.Sigmoid())
            self.D_extra_par_head = nn.Sequential(
                nn.Linear(layer_size, 1), nn.Sigmoid()
            )
            self.D_extra_perp_head = nn.Sequential(
                nn.Linear(layer_size, 1), nn.Sigmoid()
            )
        else:
            self.D_intra_head = nn.Sequential(
                nn.Linear(layer_size, self.n_fibers), nn.Sigmoid()
            )
            self.D_extra_par_head = nn.Sequential(
                nn.Linear(layer_size, self.n_fibers), nn.Sigmoid()
            )
            self.D_extra_perp_head = nn.Sequential(
                nn.Linear(layer_size, self.n_fibers), nn.Sigmoid()
            )
        self.min_diffusivity = min_diffusivity

        self.D_fw_head = nn.Sequential(nn.Linear(layer_size, 1), nn.Sigmoid())
        n_fractions = (
            self.n_fibers * 2 + 1 if self.include_freewater else self.n_fibers * 2
        )
        self.f_head = nn.Sequential(
            nn.Linear(layer_size, n_fractions), nn.Softmax(dim=-1)
        )
        self.S0_head = nn.Sequential(nn.Linear(layer_size, 1), nn.Softplus())
        self.dir_head = nn.Linear(layer_size, self.n_fibers * 3)

    def forward(self, z):
        D_intra = (
            self.D_intra_head(z) * (4e-3 - self.min_diffusivity) + self.min_diffusivity
        )
        D_extra_par = (
            self.D_extra_par_head(z) * (4e-3 - self.min_diffusivity)
            + self.min_diffusivity
        )
        D_extra_perp = (
            self.D_extra_perp_head(z) * (1.5e-3 - self.min_diffusivity)
            + self.min_diffusivity
        )
        if self.voxel_diffusivity == "extra":
            D_extra_par = D_extra_par.repeat(1, self.n_fibers)
            D_extra_perp = D_extra_perp.repeat(1, self.n_fibers)
        elif self.voxel_diffusivity:
            D_intra = D_intra.repeat(1, self.n_fibers)
            D_extra_par = D_extra_par.repeat(1, self.n_fibers)
            D_extra_perp = D_extra_perp.repeat(1, self.n_fibers)
        D_fw = (
            self.D_fw_head(z) * (4e-3 - self.min_diffusivity) + self.min_diffusivity
            if self.include_freewater
            else torch.zeros_like(D_intra[:, :1]).to(z.device)
        )
        f = self.f_head(z)
        f_intra = f[:, : self.n_fibers]
        f_extra = f[:, self.n_fibers : 2 * self.n_fibers]
        f_fw = (
            f[:, -1:]
            if self.include_freewater
            else torch.zeros_like(f_intra[:, :1]).to(z.device)
        )
        S0 = self.S0_head(z)
        dirs = self.dir_head(z).view(-1, self.n_fibers, 3)
        dirs = F.normalize(dirs, dim=-1)

        if self.include_b0:
            return (
                D_intra,
                D_extra_par,
                D_extra_perp,
                D_fw,
                f_intra,
                f_extra,
                f_fw,
                S0,
                dirs,
            )
        else:
            return (
                D_intra,
                D_extra_par,
                D_extra_perp,
                D_fw,
                f_intra,
                f_extra,
                f_fw,
                dirs,
            )


class DiscreteStandardModelMultihead(nn.Module):
    def __init__(
        self,
        max_fibers=3,
        layer_size=2048,
        num_enc=5000,
        sigma=3.5,
        include_b0=True,
        min_diffusivity=0,
        voxel_diffusivity=False,
        include_freewater=False,
    ):
        super(DiscreteStandardModelMultihead, self).__init__()
        self.max_fibers = max_fibers
        self.pos_enc = PositionalEncoding(num_enc=num_enc, dim=3, sigma=sigma)
        self.mlp = nn.Sequential(
            nn.Linear(num_enc, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList(
            [
                DiscreteStandardModelHead(
                    n_fibers=k + 1,
                    layer_size=layer_size,
                    include_b0=include_b0,
                    include_freewater=include_freewater,
                    min_diffusivity=min_diffusivity,
                    voxel_diffusivity=voxel_diffusivity,
                )
                for k in range(max_fibers)
            ]
        )

    def forward(self, x):
        x_enc = self.pos_enc(x)
        z = self.mlp(x_enc)
        return {k + 1: head(z) for k, head in enumerate(self.heads)}

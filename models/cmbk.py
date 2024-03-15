import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import math
from einops import repeat, rearrange
from typing import Tuple

from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint


class Embedder(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            self.freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in self.freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                out_dim += d

        self.out_dim = out_dim

    def forward(self, inputs) -> torch.Tensor:
        # print(f"input device: {inputs.device}, freq_bands device: {self.freq_bands.device}")
        self.freq_bands = self.freq_bands.type_as(inputs)
        outputs = []
        if self.kwargs['include_input']:
            outputs.append(inputs)

        for freq in self.freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                outputs.append(p_fn(inputs * freq))
        if outputs[0].ndim == 0:
            return torch.stack(outputs, -1)
        else:
            return torch.cat(outputs, -1)


def get_embedder(multires: bool, 
                 i: int = 0, 
                 input_dim: int = 3, 
                 include_input: bool = True
                 ) -> Tuple[nn.Module, int]:
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': include_input,
        'input_dims': input_dim,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    return embedder_obj, embedder_obj.out_dim


class KernelDerivative(nn.Module):
    def __init__(self,
                 num_rays: int = 9,
                 embed_dim: int = 64,
                 chrono_view: bool = True,
                 chrono_view_embed_dim: int = 16,
                 res_momentum: bool = True,
                 num_img: int = 29,
                 ) -> None:
        super(KernelDerivative, self).__init__()
        
        self.chrono_view = chrono_view
        self.res_momentum = res_momentum
        
        if self.chrono_view:
            self.register_parameter("chrono_view_embed", nn.Parameter(torch.zeros(num_rays, num_img, chrono_view_embed_dim).type(torch.float32), requires_grad=True))
        else:
            self.register_parameter("time_embed", nn.Parameter(torch.zeros(num_rays, chrono_view_embed_dim).type(torch.float32), requires_grad=True))
        
        self.linear1 = nn.Linear(embed_dim + chrono_view_embed_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim + chrono_view_embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.relu = nn.ReLU(inplace=True)
        
        
    def forward(self,
                t: int = 0,
                x: torch.Tensor = None,
                img_idx: torch.Tensor = None,
                ) -> torch.Tensor:
        '''
        x : [ Batch size, feature_dim ]
        t : 0 ~ num_rays (9)
        output : [ Batch size, proj_dim ]
        '''
        if self.chrono_view:
            cv_embed = self.chrono_view_embed[int(t)][img_idx.type(torch.long)]
        else:
            cv_embed = self.time_embed[int(t)]
            cv_embed = repeat(cv_embed, 'd -> n d', n=x.shape[0])
            
        res_momentum = x
        x = self.relu(x)
        x = torch.cat([x, cv_embed], dim=-1)
        x = self.linear1(x)
        x = self.relu(x)
        x = torch.cat([x, cv_embed], dim=-1)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.proj(x + res_momentum) if self.res_momentum else self.proj(x)
        
        return x


class DiffEqSolver(nn.Module):
    def __init__(self, 
                 odefunc: nn.Module = None,
                 method: str = 'euler',
                 odeint_rtol: float = 1e-4,
                 odeint_atol: float = 1e-5,
                 num_rays: int = 9,
                 point_float: bool = False,
                 adjoint: bool = False,
                 ) -> None:
        super(DiffEqSolver, self).__init__()
        
        self.ode_func = odefunc
        self.ode_method = method
        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol
        self.integration_time = torch.linspace(0., 1., num_rays) if point_float else torch.arange(0, num_rays, dtype=torch.long)
        self.solver = odeint_adjoint if adjoint else odeint
            
    def forward(self, 
                x: torch.Tensor = None,
                img_idx: torch.Tensor = None,
                ) -> torch.Tensor:
        '''
        x                     : [ Batch size, feature_dim ]
        out                   : [ Batch size, num_rays, feature_dim ]
        '''
        self.integration_time = self.integration_time.type_as(x)
        out = self.solver(self.ode_func, (x, img_idx), self.integration_time.cuda(x.get_device()),
                     rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)
        out = rearrange(out, 'n b d -> b n d')
        return out
    

class CMBK(nn.Module):
    def __init__(self,
                 num_img: int = 29,
                 poses: torch.Tensor = None,
                 num_warp: int = 9,
                 img_embed: int = 32,
                 spatial_embed_freq: int = 2,
                 embed_dim: int = 64,
                 kernel_window: int = 10,
                 method: str = 'euler',
                 adjoint: bool = False,
                 point_float: bool = False,
                 chrono_view: bool = True,
                 chrono_view_embed_dim: int = 16,
                 res_momentum: bool = True
                 ) -> None:
        super(CMBK, self).__init__()
        self.num_rays = num_warp + 1
        self.num_img = num_img
        self.img_embed_dim = img_embed
        self.embed_dim = embed_dim
        self.kernel_window = kernel_window
        
        self.register_buffer('poses', poses)
        self.register_parameter("img_embed", nn.Parameter(torch.zeros(num_img, img_embed).type(torch.float32), requires_grad=True))
        
        self.spatial_embed_fn, self.spatial_embed_dim = get_embedder(spatial_embed_freq, input_dim=2)
        self.feature_dim = self.spatial_embed_dim + self.img_embed_dim
        
        self.kernel_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.kernel_derivative = KernelDerivative(
            num_rays=self.num_rays, 
            embed_dim=self.embed_dim, 
            chrono_view=chrono_view,
            chrono_view_embed_dim=chrono_view_embed_dim,
            res_momentum=res_momentum,
            num_img=self.num_img)
        self.diffeq_solver = DiffEqSolver(
            odefunc=self.kernel_derivative,
            method=method,
            num_rays=self.num_rays,
            point_float=point_float,
            adjoint=adjoint,
        )
        self.kernel_decoder = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, 5),
        )
        
    def forward(self, 
                H: int, 
                W: int, 
                K: int, 
                frames: torch.Tensor, 
                rays_x: torch.Tensor, 
                rays_y: torch.Tensor,
                iter: int = 0,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_idx = frames.squeeze(-1)
        img_embed = self.img_embed[img_idx]
        
        spatial_x = rays_x / (W / 2 / np.pi) - np.pi
        spatial_y = rays_y / (H / 2 / np.pi) - np.pi
        spatial_pos = torch.cat([spatial_x, spatial_y],
                                  dim=-1)
        spatial_embed = self.spatial_embed_fn(spatial_pos)
        x = torch.cat([img_embed, spatial_embed], dim=-1)
        
        x = self.kernel_encoder(x)
        
        x_proj = self.diffeq_solver(x, img_idx)
        x_proj = self.kernel_decoder(x_proj)
        
        delta_origin, delta_pixel, weight = torch.split(x_proj, [2, 2, 1], dim=-1)
        delta_origin = delta_origin * 0.01
        delta_rays_xy = delta_pixel * torch.arange(1, self.num_rays + 1).cuda(x.get_device()).view(1, -1, 1)
        delta_rays_xy = delta_rays_xy * self.kernel_window / self.num_rays
            
        weight = torch.softmax(weight[..., -1], dim=-1)
        
        output_supp_loss = delta_rays_xy[:, 0, :].abs().mean() + delta_origin[:, 0, :].abs().mean() * 10
        
        ''' From this point, we use the same code as Deblur-NeRF '''
        poses = self.poses[img_idx]
        
        rays_x = (rays_x - K[0, 2] + delta_rays_xy[..., 0]) / K[0, 0]
        rays_y = - (rays_y - K[1, 2] + delta_rays_xy[..., 1]) / K[1, 1]
        dirs = torch.stack([rays_x - delta_origin[..., 0],
                            rays_y - delta_origin[..., 1],
                            - torch.ones_like(rays_x)], -1)
        
        rays_d = torch.sum(dirs[..., None, :] * poses[..., None, :3, :3], dim=-1)
        translation = torch.stack([
            delta_origin[..., 0],
            delta_origin[..., 1],
            torch.zeros_like(rays_x),
            torch.ones_like(rays_x)
        ], dim=-1)
        rays_o = torch.sum(translation[..., None, :] * poses[:, None], dim=-1)
        
        return torch.cat([rays_o, rays_d], dim=-1), weight, output_supp_loss
        
        

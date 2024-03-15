import torch, os, imageio, sys
from tqdm.auto import tqdm
from dataLoader.ray_utils import get_rays
from models.smurf import SMURF, raw2alpha, AlphaGridMask
from utils import *
from dataLoader.ray_utils import ndc_rays_blender


def OctreeRender_trilinear_fast(rays, frames=None, rays_x_train=None, rays_y_train=None, model=None, iter=None, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda'):

    rgbs, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    output_supp_loss = []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        if frames is not None:
            frames_chunk = frames[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
            rays_x_chunk = rays_x_train[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
            rays_y_chunk = rays_y_train[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        else:
            frames_chunk = None
            rays_x_chunk = None
            rays_y_chunk = None
    
        rgb_map, depth_map, output_supp = model(rays_chunk, frames_chunk, rays_x_chunk, rays_y_chunk, iter=iter, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)

        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
        output_supp_loss.append(output_supp)
    if is_train:
        return torch.cat(rgbs), None, torch.cat(depth_maps), None, None, output_supp_loss[0]
    else:
        return torch.cat(rgbs), None, torch.cat(depth_maps), None, None
    


@torch.no_grad()
def evaluation(test_dataset, model, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims, l_alex, l_vgg = [], [], []
    ssims_nerf = []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/rgbd", exist_ok=True)

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis, 1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    for idx, samples in enumerate(test_dataset.all_rays[0::img_eval_interval]):

        W, H = test_dataset.img_wh
        rays = samples.view(-1,samples.shape[-1])
        rgb_map, _, depth_map, _, _ = renderer(rays, model=model, iter=None, chunk=4096, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim_nerf = rgb_ssim_nerf(rgb_map, gt_rgb)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', model.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', model.device)
                ssims_nerf.append(ssim_nerf)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        print_log(f'TEST PSNR     : {psnr}')
        if compute_extra_metrics:
            ssim_nerf = np.mean(np.asarray(ssims_nerf))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            print_log(f'TEST SSIM     : {np.mean(np.asarray(ssim_nerf))}')
            print_log(f'TEST LPIPS    : {np.mean(np.asarray(l_a))}')
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim_nerf, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))

    return PSNRs

@torch.no_grad()
def evaluation_path(test_dataset, model, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda', savePathImgs=False):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    near_far = test_dataset.near_far
    for idx, c2w in enumerate(tqdm(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        rgb_map, _, depth_map, _, _ = renderer(rays, model=model, iter=None, chunk=8192, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None and savePathImgs:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8, macro_block_size=1)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8, macro_block_size=1)

    return PSNRs


import torch
import numpy as np
import gc
import os
from torchvision.utils import make_grid, save_image
from datasets import inverse_data_transform

def get_sigmas(config):
    if config.model.sigma_dist == 'geometric':
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(config.model.sigma_begin), np.log(config.model.sigma_end),
                               config.model.num_classes))).float().to(config.device)
    elif config.model.sigma_dist == 'uniform':
        sigmas = torch.tensor(
            np.linspace(config.model.sigma_begin, config.model.sigma_end, config.model.num_classes)
        ).float().to(config.device)

    else:
        raise NotImplementedError('sigma distribution not supported')

    return sigmas

@torch.no_grad()
def anneal_Langevin_dynamics(x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False, denoise=True, start_idx=0, remainder=0, 
                               root_dir='./',config=None,bat=0):
    images = []
    
    if remainder is 0:
        remainder =n_steps_each
    
    used_sigmas = sigmas[start_idx:]
    
    last_sig_idx = used_sigmas.shape[0]-1
    internal_steps=n_steps_each
    cnts =0
    
    if config is not None:
        save_idx=used_sigmas.shape[0]*n_steps_each-config.sampling.save_steps
    else:
        save_idx=used_sigmas.shape[0]*n_steps_each-50
    

    with torch.no_grad():
        for c, sigma in enumerate(used_sigmas):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * (c+start_idx)
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            
            if c==last_sig_idx:
                internal_steps=remainder
            
            for s in range(internal_steps):
                grad = scorenet(x_mod, labels)
                noise = torch.randn_like(x_mod)
                x_mod = x_mod + step_size * grad + noise * np.sqrt(step_size * 2)

                if not final_only:
                    if cnts>=save_idx:
                        sample = x_mod.view(x_mod.shape[0], config.data.channels,config.data.image_size,config.data.image_size)
                        torch.save(sample, os.path.join(root_dir, str(cnts),'pth', 'samples_{}_{}.pth'.format(bat,cnts)))
                        sample = inverse_data_transform(config, sample)
                        image_grid = make_grid(sample, int(np.sqrt(config.sampling.batch_size)))
                        save_image(image_grid, os.path.join(root_dir, str(cnts),'images','image_grid_{}_{}.png'.format(bat,cnts)))
                    cnts+=1
                    
                if verbose:
                    grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                    noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                    image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
                    snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
                    grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2
                    print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                        c+start_idx, step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))
              
              

        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
            images.append(x_mod.to('cpu'))
            if config is not None:
                sample = x_mod.view(x_mod.shape[0], config.data.channels,config.data.image_size,config.data.image_size)
                torch.save(sample, os.path.join(root_dir, str(cnts),'pth', 'samples_{}_{}.pth'.format(bat,cnts)))
                sample = inverse_data_transform(config, sample)
                image_grid = make_grid(sample, int(np.sqrt(config.sampling.batch_size)))
                save_image(image_grid, os.path.join(root_dir, str(cnts),'images','image_grid_{}_{}.png'.format(bat,cnts)))
            
        gc.collect()
        
        if final_only:
            return [x_mod.to('cpu')]
        else:
            return images


@torch.no_grad()
def anneal_Langevin_dynamics_inpainting(x_mod, refer_image, scorenet, sigmas, image_size,
                                        n_steps_each=100, step_lr=0.000008):
    """
    Currently only good for 32x32 images. Assuming the right half is missing.
    """

    images = []

    refer_image = refer_image.unsqueeze(1).expand(-1, x_mod.shape[1], -1, -1, -1)
    refer_image = refer_image.contiguous().view(-1, 3, image_size, image_size)
    x_mod = x_mod.view(-1, 3, image_size, image_size)
    cols = image_size // 2
    half_refer_image = refer_image[..., :cols]
    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2

            for s in range(n_steps_each):
                images.append(x_mod.to('cpu'))
                corrupted_half_image = half_refer_image + torch.randn_like(half_refer_image) * sigma
                x_mod[:, :, :, :cols] = corrupted_half_image
                noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                grad = scorenet(x_mod, labels)
                x_mod = x_mod + step_size * grad + noise
                print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                                                                         grad.abs().max()))

        return images

@torch.no_grad()
def anneal_Langevin_dynamics_interpolation(x_mod, scorenet, sigmas, n_interpolations, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False):
    images = []

    n_rows = x_mod.shape[0]

    x_mod = x_mod[:, None, ...].repeat(1, n_interpolations, 1, 1, 1)
    x_mod = x_mod.reshape(-1, *x_mod.shape[2:])

    for c, sigma in enumerate(sigmas):
        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
        labels = labels.long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        for s in range(n_steps_each):
            grad = scorenet(x_mod, labels)

            noise_p = torch.randn(n_rows, x_mod.shape[1], x_mod.shape[2], x_mod.shape[3],
                                  device=x_mod.device)
            noise_q = torch.randn(n_rows, x_mod.shape[1], x_mod.shape[2], x_mod.shape[3],
                                  device=x_mod.device)
            angles = torch.linspace(0, np.pi / 2., n_interpolations, device=x_mod.device)

            noise = noise_p[:, None, ...] * torch.cos(angles)[None, :, None, None, None] + \
                        noise_q[:, None, ...] * torch.sin(angles)[None, :, None, None, None]

            noise = noise.reshape(-1, *noise.shape[2:])
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
            image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()

            x_mod = x_mod + step_size * grad + noise * np.sqrt(step_size * 2)

            snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm

            if not final_only:
                images.append(x_mod.to('cpu'))
            if verbose:
                print(
                    "level: {}, step_size: {}, image_norm: {}, grad_norm: {}, snr: {}".format(
                        c, step_size, image_norm.item(), grad_norm.item(), snr.item()))


    if final_only:
        return [x_mod.to('cpu')]
    else:
        return images
    

def forward_diffusion(scorenet, samples, sigmas, labels):
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    noise = torch.randn_like(samples) * used_sigmas
    perturbed_samples = samples + noise
    #target = - 1 / (used_sigmas ** 2) * noise
    #scores = scorenet(perturbed_samples, labels)

    return perturbed_samples
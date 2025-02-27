import numpy as np
import glob
import tqdm
from losses.dsm import anneal_dsm_score_estimation,anneal_dsm_score_fine_tune

import torch.nn.functional as F
import logging
import torch
import os
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from models.ncsnv2 import NCSNv2Deeper, NCSNv2, NCSNv2Deepest
from models.ncsn import NCSN, NCSNdeeper
from datasets import get_dataset, data_transform, inverse_data_transform
from losses import get_optimizer
from models import (anneal_Langevin_dynamics,
                    anneal_Langevin_dynamics_inpainting,
                    anneal_Langevin_dynamics_interpolation,
                    forward_diffusion)
from models import get_sigmas
from models.ema import EMAHelper
from evaluation.fid_score import get_fid, get_fid_stats_path
import pickle
import time

from ot_utils.ot_util import get_OT_solver, OT_Map

__all__ = ['NCSNRunner']


def get_model(config):
    if config.data.dataset == 'CIFAR10' or config.data.dataset == 'CELEBA':
        return NCSNv2(config).to(config.device)
    elif config.data.dataset == "FFHQ":
        return NCSNv2Deepest(config).to(config.device)
    elif config.data.dataset == 'LSUN':
        return NCSNv2Deeper(config).to(config.device)


def freeze_encoder(config,model):
    for param in model.module.res1.parameters():
        param.requires_grad = False
    for param in model.module.res2.parameters():
        param.requires_grad = False
    for param in model.module.res3.parameters():
        param.requires_grad = False
    for param in model.module.res4.parameters():
        param.requires_grad = False
    if config.data.dataset == "FFHQ":
        for param in model.module.res31.parameters():
            param.requires_grad = False
        for param in model.module.res5.parameters():
            param.requires_grad = False
    elif config.data.dataset == 'LSUN':
        for param in model.module.res5.parameters():
            param.requires_grad = False
    return model


class NCSNRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.sampling_fid = config.sampling.fid
        self.sampling_final_only = config.sampling.final_only
    
    def set_sampling_fid(self,sampling_fid):
        self.sampling_fid = sampling_fid
    def set_sampling_final_only(self,final_only):
        self.sampling_final_only = final_only
        

    def train(self):
        dataset, test_dataset = get_dataset(self.args, self.config)#获取数据，并划分为训练集和测试集
        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                num_workers=self.config.data.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=self.config.data.num_workers, drop_last=True)
        test_iter = iter(test_loader)
        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        tb_logger = self.config.tb_logger

        score = get_model(self.config)

        score = torch.nn.DataParallel(score)
        optimizer = get_optimizer(self.config, score.parameters())

        start_epoch = 0
        step = 0

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'))
            score.load_state_dict(states[0])
            ### Make sure we can resume with different eps
            states[1]['param_groups'][0]['eps'] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        sigmas = get_sigmas(self.config)
        
        self.args.log_sample_path = os.path.join(self.args.log_path, 'samples')
        if not os.path.exists(self.args.log_sample_path):
            os.makedirs(self.args.log_sample_path)

        if self.config.training.log_all_sigmas:
            ### Commented out training time logging to save time.
            test_loss_per_sigma = [None for _ in range(len(sigmas))]

            def hook(loss, labels):
                # for i in range(len(sigmas)):
                #     if torch.any(labels == i):
                #         test_loss_per_sigma[i] = torch.mean(loss[labels == i])
                pass

            def tb_hook():
                # for i in range(len(sigmas)):
                #     if test_loss_per_sigma[i] is not None:
                #         tb_logger.add_scalar('test_loss_sigma_{}'.format(i), test_loss_per_sigma[i],
                #                              global_step=step)
                pass

            def test_hook(loss, labels):
                for i in range(len(sigmas)):
                    if torch.any(labels == i):
                        test_loss_per_sigma[i] = torch.mean(loss[labels == i])

            def test_tb_hook():
                for i in range(len(sigmas)):
                    if test_loss_per_sigma[i] is not None:
                        tb_logger.add_scalar('test_loss_sigma_{}'.format(i), test_loss_per_sigma[i],
                                             global_step=step)

        else:
            hook = test_hook = None

            def tb_hook():
                pass

            def test_tb_hook():
                pass

        for epoch in range(start_epoch, self.config.training.n_epochs):
            for i, (X, y) in enumerate(dataloader):
                score.train()
                step += 1

                X = X.to(self.config.device)
                X = data_transform(self.config, X)

                loss = anneal_dsm_score_estimation(score, X, sigmas, None,
                                                   self.config.training.anneal_power,
                                                   hook)
                tb_logger.add_scalar('loss', loss, global_step=step)
                tb_hook()

                logging.info("step: {}, loss: {}".format(step, loss.item()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(score)

                if step >= self.config.training.n_iters:
                    return 0

                if step % 100 == 0:
                    if self.config.model.ema:
                        test_score = ema_helper.ema_copy(score)
                    else:
                        test_score = score

                    test_score.eval()
                    try:
                        test_X, test_y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)

                    test_X = test_X.to(self.config.device)
                    test_X = data_transform(self.config, test_X)

                    with torch.no_grad():
                        test_dsm_loss = anneal_dsm_score_estimation(test_score, test_X, sigmas, None,
                                                                    self.config.training.anneal_power,
                                                                    hook=test_hook)
                        tb_logger.add_scalar('test_loss', test_dsm_loss, global_step=step)
                        test_tb_hook()
                        logging.info("step: {}, test_loss: {}".format(step, test_dsm_loss.item()))

                        del test_score

                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(states, os.path.join(self.args.log_path, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log_path, 'checkpoint.pth'))

                    if self.config.training.snapshot_sampling:
                        if self.config.model.ema:
                            test_score = ema_helper.ema_copy(score)
                        else:
                            test_score = score

                        test_score.eval()

                        ## Different part from NeurIPS 2019.
                        ## Random state will be affected because of sampling during training time.
                        init_samples = torch.rand(36, self.config.data.channels,
                                                  self.config.data.image_size, self.config.data.image_size,
                                                  device=self.config.device)
                        init_samples = data_transform(self.config, init_samples)

                        all_samples = anneal_Langevin_dynamics(init_samples, test_score, sigmas.cpu().numpy(),
                                                               self.config.sampling.n_steps_each,
                                                               self.config.sampling.step_lr,
                                                               final_only=True, verbose=True,
                                                               denoise=self.config.sampling.denoise)

                        sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                                      self.config.data.image_size,
                                                      self.config.data.image_size)
                        torch.save(sample, os.path.join(self.args.log_sample_path, 'samples_{}.pth'.format(step)))
                        sample = inverse_data_transform(self.config, sample)

                        image_grid = make_grid(sample, 6)
                        save_image(image_grid,
                                   os.path.join(self.args.log_sample_path, 'image_grid_{}.png'.format(step)))
                        

                        del test_score
                        del all_samples

    def sample(self, batch_numb=1):
        
        if self.config.sampling.ckpt_id is None:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'), map_location=self.config.device)
        else:
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{self.config.sampling.ckpt_id}.pth'),
                                map_location=self.config.device)
        '''
        states = torch.load(os.path.join(self.args.log_path, 'fine_tune', 'checkpoint_5000.pth'), map_location=self.config.device)
        '''
        score = get_model(self.config)
        score = torch.nn.DataParallel(score)

        score.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(score)

        sigmas_th = get_sigmas(self.config)
        sigmas = sigmas_th.cpu().numpy()

        dataset, _ = get_dataset(self.args, self.config)
        dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size, shuffle=True,
                                num_workers=4)

        score.eval()

        if not self.sampling_fid:
            
            if not self.config.sampling.final_only:     
                all_steps= self.config.model.num_classes*self.config.sampling.n_steps_each + 1   
                save_idx = all_steps-self.config.sampling.save_steps
                for b in range(save_idx,all_steps):
                    image_path = os.path.join(self.args.image_folder, str(b),'images')
                    if not os.path.exists(image_path):
                        os.makedirs(image_path)
                    pth_path = os.path.join(self.args.image_folder, str(b), 'pth')
                    if not os.path.exists(pth_path):
                        os.makedirs(pth_path)
            else:
                image_path = os.path.join(self.args.image_folder, 'images')
                if not os.path.exists(image_path):
                    os.makedirs(image_path)
                pth_path = os.path.join(self.args.image_folder, 'pth')
                if not os.path.exists(pth_path):
                    os.makedirs(pth_path)
            
            if self.config.sampling.inpainting:
                data_iter = iter(dataloader)
                refer_images, _ = next(data_iter)
                refer_images = refer_images.to(self.config.device)
                width = int(np.sqrt(self.config.sampling.batch_size))
                init_samples = torch.rand(width, width, self.config.data.channels,
                                          self.config.data.image_size,
                                          self.config.data.image_size,
                                          device=self.config.device)
                init_samples = data_transform(self.config, init_samples)
                all_samples = anneal_Langevin_dynamics_inpainting(init_samples, refer_images[:width, ...], score,
                                                                  sigmas,
                                                                  self.config.data.image_size,
                                                                  self.config.sampling.n_steps_each,
                                                                  self.config.sampling.step_lr)

                torch.save(refer_images[:width, ...], os.path.join(self.args.image_folder, 'refer_image.pth'))
                refer_images = refer_images[:width, None, ...].expand(-1, width, -1, -1, -1).reshape(-1,
                                                                                                     *refer_images.shape[
                                                                                                      1:])
                save_image(refer_images, os.path.join(self.args.image_folder, 'refer_image.png'), nrow=width)

                if not self.config.sampling.final_only:
                    for i, sample in enumerate(tqdm.tqdm(all_samples)):
                        sample = sample.view(self.config.sampling.batch_size, self.config.data.channels,
                                             self.config.data.image_size,
                                             self.config.data.image_size)
                        torch.save(sample, os.path.join(self.args.image_folder, 'completion_{}.pth'.format(i)))
                        sample = inverse_data_transform(self.config, sample)

                        image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                        save_image(image_grid, os.path.join(self.args.image_folder, 'image_grid_{}.png'.format(i)))
                        
                else:
                    sample = all_samples[-1].view(self.config.sampling.batch_size, self.config.data.channels,
                                                  self.config.data.image_size,
                                                  self.config.data.image_size)
                    torch.save(sample, os.path.join(self.args.image_folder,
                                                    'completion_{}.pth'.format(self.config.sampling.ckpt_id)))
                    sample = inverse_data_transform(self.config, sample)

                    image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                    save_image(image_grid, os.path.join(self.args.image_folder,
                                                        'image_grid_{}.png'.format(self.config.sampling.ckpt_id)))
                    

            elif self.config.sampling.interpolation:
                if self.config.sampling.data_init:
                    data_iter = iter(dataloader)
                    samples, _ = next(data_iter)
                    samples = samples.to(self.config.device)
                    samples = data_transform(self.config, samples)
                    init_samples = samples + sigmas_th[0] * torch.randn_like(samples)

                else:
                    init_samples = torch.rand(self.config.sampling.batch_size, self.config.data.channels,
                                              self.config.data.image_size, self.config.data.image_size,
                                              device=self.config.device)
                    init_samples = data_transform(self.config, init_samples)

                all_samples = anneal_Langevin_dynamics_interpolation(init_samples, score, sigmas,
                                                                     self.config.sampling.n_interpolations,
                                                                     self.config.sampling.n_steps_each,
                                                                     self.config.sampling.step_lr, verbose=True,
                                                                     final_only=self.config.sampling.final_only)

                if not self.config.sampling.final_only:
                    for i, sample in tqdm.tqdm(enumerate(all_samples), total=len(all_samples),
                                               desc="saving image samples"):
                        sample = sample.view(sample.shape[0], self.config.data.channels,
                                             self.config.data.image_size,
                                             self.config.data.image_size)
                        torch.save(sample, os.path.join(self.args.image_folder, 'samples_{}.pth'.format(i)))
                        sample = inverse_data_transform(self.config, sample)

                        image_grid = make_grid(sample, nrow=self.config.sampling.n_interpolations)
                        save_image(image_grid, os.path.join(self.args.image_folder, 'image_grid_{}.png'.format(i)))
                        
                else:
                    sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                                  self.config.data.image_size,
                                                  self.config.data.image_size)
                    torch.save(sample, os.path.join(self.args.image_folder,
                                                    'samples_{}.pth'.format(self.config.sampling.ckpt_id)))
                    sample = inverse_data_transform(self.config, sample)

                    image_grid = make_grid(sample, self.config.sampling.n_interpolations)
                    save_image(image_grid, os.path.join(self.args.image_folder,
                                                        'image_grid_{}.png'.format(self.config.sampling.ckpt_id)))
                    

            else:
                #print('The max batch is : ', batch_numb)
                logging.info("The max batch is : {}".format(batch_numb))
                for b in range(33,batch_numb):
                    #print('The batch number is : ', b)
                    logging.info("The batch number is : {}".format(b))
                    old_time = time.time()
                    '''
                    del ema_helper
                    del score
                    
                    score = get_model(self.config)
                    score = torch.nn.DataParallel(score)
                    score.load_state_dict(states[0], strict=True)
                    if self.config.model.ema:
                        ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                        ema_helper.register(score)
                        ema_helper.load_state_dict(states[-1])
                        ema_helper.ema(score)
                    '''
                    
                    if self.config.sampling.data_init:
                        data_iter = iter(dataloader)
                        samples, _ = next(data_iter)
                        samples = samples.to(self.config.device)
                        samples = data_transform(self.config, samples)
                        init_samples = samples + sigmas_th[0] * torch.randn_like(samples)

                    else:
                        init_samples = torch.rand(self.config.sampling.batch_size, self.config.data.channels,
                                              self.config.data.image_size, self.config.data.image_size,
                                              device=self.config.device)
                        init_samples = data_transform(self.config, init_samples)
                    logging.info("Start sampling")
                    all_samples = anneal_Langevin_dynamics(init_samples, score, sigmas,
                                                       self.config.sampling.n_steps_each,
                                                       self.config.sampling.step_lr, verbose=False,
                                                       final_only=self.config.sampling.final_only,
                                                       denoise=self.config.sampling.denoise,
                                                       root_dir=self.args.image_folder,config=self.config,bat=b)
                    #torch.save(init_samples, os.path.join(self.args.image_folder, 'pth', 'init_samples.pth'))
                    current_time = time.time()
                    logging.info("Sample time is: {}s".format(current_time-old_time))
                    if not self.config.sampling.final_only:
                        '''
                        for i, sample in enumerate(all_samples):
                            sample = sample.view(sample.shape[0], self.config.data.channels,
                                             self.config.data.image_size,
                                             self.config.data.image_size)
                            torch.save(sample, os.path.join(self.args.image_folder, str(i),'pth', 'samples_{}_{}.pth'.format(b,i)))
                            sample = inverse_data_transform(self.config, sample)
                            image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                            save_image(image_grid, os.path.join(self.args.image_folder, str(i),'images','image_grid_{}_{}.png'.format(b,i)))
                            
                            #logging.info("The {}-th image is saved!".format(i))
                        '''
                        logging.info("A batch sample hava been saved!")
                        logging.info("The consumed time of a batch sample saving is: {}s".format(time.time()-current_time))
                        logging.info("The consumed time of a batch sampling is: {}s".format(time.time()-old_time))
                    else:
                        sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                                  self.config.data.image_size,
                                                  self.config.data.image_size)
                        torch.save(sample, os.path.join(self.args.image_folder,'pth', 'samples_{}_{}.pth'.format(b,self.config.sampling.ckpt_id)))
                        sample = inverse_data_transform(self.config, sample)
                        image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                        save_image(image_grid, os.path.join(self.args.image_folder,'images',
                                                        'image_grid_{}_{}.png'.format(b,self.config.sampling.ckpt_id)))
                        
                self.config.OT.source_dir = os.path.join(self.args.image_folder,str(self.config.model.num_classes-self.args.backSteps-1))
                self.config.OT.h_name = None
                self.sampling_fid = True
                self.sampling_final_only = True
                
        else:
            print('self.config.OT.source_dir=',self.config.OT.source_dir)
            print('self.sampling_fid=',self.sampling_fid)
            print('self.sampling_final_only=',self.sampling_final_only)
            
            total_n_samples = self.config.sampling.num_samples4fid
            n_rounds = total_n_samples // self.config.sampling.batch_size
            if self.config.sampling.data_init:
                dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size, shuffle=True,
                                        num_workers=4)
                data_iter = iter(dataloader)
                sig_n=0
                remain = 0
            else:
                remain = self.config.OT.backSteps%self.config.sampling.n_steps_each
                sig_n = sigmas.shape[0]-self.config.OT.backSteps//self.config.sampling.n_steps_each
                if remain>0:
                    sig_n-=1

            img_id = 0
            output_path = os.path.join(self.args.image_folder, 'fid')
            #'''
            ## get the OT solver
            idf_max_step = sigmas.shape[0]*self.config.sampling.n_steps_each
            #OT_solver = get_OT_solver(self.args,idf_max_step)
            OT_solver = get_OT_solver(self.config.OT,idf_max_step)
            
            for _ in tqdm.tqdm(range(n_rounds), desc='Generating image samples for FID/inception score evaluation'):
                if self.config.sampling.data_init:
                    try:
                        samples, _ = next(data_iter)
                    except StopIteration:
                        data_iter = iter(dataloader)
                        samples, _ = next(data_iter)
                    samples = samples.to(self.config.device)
                    samples = data_transform(self.config, samples)
                    samples = samples + sigmas_th[0] * torch.randn_like(samples)
                else:
                    samples = torch.rand(self.config.sampling.batch_size, self.config.data.channels,
                                         self.config.data.image_size,
                                         self.config.data.image_size, device=self.config.device)
                    #samples = data_transform(self.config, samples)
                    samples = OT_Map(self.config.OT,samples,OT_solver)

                all_samples = anneal_Langevin_dynamics(samples, score, sigmas,
                                                       self.config.sampling.n_steps_each,
                                                       self.config.sampling.step_lr, verbose=False,
                                                       final_only=self.sampling_final_only,
                                                       denoise=self.config.sampling.denoise,
                                                       start_idx=sig_n,remainder=remain)

                samples = all_samples[-1]
                for img in samples:
                    img = img.view(self.config.data.channels,
                                         self.config.data.image_size,
                                         self.config.data.image_size)
                    img = inverse_data_transform(self.config, img)

                    save_image(img, os.path.join(output_path, 'image_{}.png'.format(img_id)))
                    img_id += 1
            ########################################## FID ########################################    
            #'''    
            if self.config.data.dataset == 'CIFAR10':
                stat_path = get_fid_stats_path(self.args, self.config, download=True)
            else:
                stat_path = self.config.data.data_dir
            fid = get_fid(stat_path, output_path)
        
            print("The FID score of {} images generated by DF-OT is: {}".format(n_rounds*self.config.sampling.batch_size, fid))
            

    def test(self):
        score = get_model(self.config)
        score = torch.nn.DataParallel(score)

        sigmas = get_sigmas(self.config)
        print('sigmas[0]=',sigmas[0])
        print('sigmas[-1]=',sigmas[-1])

        dataset, test_dataset = get_dataset(self.args, self.config)
        test_dataloader = DataLoader(test_dataset, batch_size=self.config.test.batch_size, shuffle=True,
                                     num_workers=self.config.data.num_workers, drop_last=True)

        verbose = False
        for ckpt in tqdm.tqdm(range(self.config.test.begin_ckpt, self.config.test.end_ckpt + 1, 5000),
                              desc="processing ckpt:"):
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{ckpt}.pth'),
                                map_location=self.config.device)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(score)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(score)
            else:
                score.load_state_dict(states[0])

            score.eval()
            bat = self.config.test.batch_size
            '''
            remain = self.config.OT.backSteps%self.config.sampling.n_steps_each
            sig_n = sigmas.shape[0]-self.config.OT.backSteps//self.config.sampling.n_steps_each
            if remain>0:
                sig_n-=1
            labels = sig_n*torch.ones((bat,), device=self.config.device).long()
            '''
            labels = (sigmas.shape[0]-self.config.OT.backSteps)*torch.ones((bat,), device=self.config.device).long()
            
            image_path = os.path.join(self.args.image_folder, str(self.config.OT.backSteps), 'images')
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            pth_path = os.path.join(self.args.image_folder, str(self.config.OT.backSteps), 'pth')
            if not os.path.exists(pth_path):
                os.makedirs(pth_path)

            step = 0
            mean_loss = 0.
            mean_grad_norm = 0.
            average_grad_scale = 0.
            for x, y in test_dataloader:
                
                x = x.to(self.config.device)
                x = data_transform(self.config, x)

                with torch.no_grad():
                    latents = forward_diffusion(score, x, sigmas, labels)
                    sample = latents.view(x.shape[0], self.config.data.channels,self.config.data.image_size,self.config.data.image_size)
                    torch.save(sample, os.path.join(pth_path, 'samples_{}_{}.pth'.format(ckpt,step)))
                    ## is necessary or not?
                    sample = inverse_data_transform(self.config, sample)
    
                    image_grid = make_grid(sample, int(np.sqrt(bat)))
                    save_image(image_grid, os.path.join(image_path,'image_grid_{}_{}.png'.format(ckpt,step)))
                    step += 1
                    '''
                    test_loss = anneal_dsm_score_estimation(score, x, sigmas, None,
                                                            self.config.training.anneal_power)
                    if verbose:
                        logging.info("step: {}, test_loss: {}".format(step, test_loss.item()))

                    mean_loss += test_loss.item()
                    '''

            '''
            mean_loss /= step
            mean_grad_norm /= step
            average_grad_scale /= step

            logging.info("ckpt: {}, average test loss: {}".format(ckpt, mean_loss))
            '''
            

    def fast_fid(self):
        ### Test the fids of ensembled checkpoints.
        ### Shouldn't be used for models with ema
        if self.config.fast_fid.ensemble:
            if self.config.model.ema:
                raise RuntimeError("Cannot apply ensembling to models with EMA.")
            self.fast_ensemble_fid()
            return
        score = get_model(self.config)
        score = torch.nn.DataParallel(score)

        sigmas_th = get_sigmas(self.config)
        sigmas = sigmas_th.cpu().numpy()

        fids = {}
        ## get the OT solver
        idf_max_step = sigmas.shape[0]*self.config.sampling.n_steps_each
        #OT_solver = get_OT_solver(self.args,idf_max_step)
        OT_solver = get_OT_solver(self.config.OT,idf_max_step)
        remain = self.config.OT.backSteps%self.config.sampling.n_steps_each
        sig_n = sigmas.shape[0]-self.config.OT.backSteps//self.config.sampling.n_steps_each
        if remain>0:
            sig_n-=1
        
        for ckpt in tqdm.tqdm(range(self.config.fast_fid.begin_ckpt, self.config.fast_fid.end_ckpt + 1, 5000),
                              desc="processing ckpt"):
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{ckpt}.pth'),
                                map_location=self.config.device)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(score)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(score)
            else:
                score.load_state_dict(states[0])

            score.eval()

            num_iters = self.config.fast_fid.num_samples // self.config.fast_fid.batch_size
            output_path = os.path.join(self.args.image_folder, 'ckpt_{}'.format(ckpt))
            os.makedirs(output_path, exist_ok=True)
            for i in range(num_iters):
                init_samples = torch.rand(self.config.fast_fid.batch_size, self.config.data.channels,
                                          self.config.data.image_size, self.config.data.image_size,
                                          device=self.config.device)
                #init_samples = data_transform(self.config, init_samples)
                init_samples = OT_Map(self.config.OT,init_samples,OT_solver)
                
                
                all_samples = anneal_Langevin_dynamics(init_samples, score, sigmas,
                                                       self.config.fast_fid.n_steps_each,
                                                       self.config.fast_fid.step_lr,
                                                       verbose=self.config.fast_fid.verbose,
                                                       denoise=self.config.sampling.denoise,
                                                       start_idx=sig_n,remainder=remain)

                final_samples = all_samples[-1]
                for id, sample in enumerate(final_samples):
                    sample = sample.view(self.config.data.channels,
                                         self.config.data.image_size,
                                         self.config.data.image_size)

                    sample = inverse_data_transform(self.config, sample)

                    save_image(sample, os.path.join(output_path, 'sample_{}.png'.format(id)))

            stat_path = get_fid_stats_path(self.args, self.config, download=True)
            fid = get_fid(stat_path, output_path)
            fids[ckpt] = fid
            print("ckpt: {}, fid: {}".format(ckpt, fid))

        with open(os.path.join(self.args.image_folder, 'fids.pickle'), 'wb') as handle:
            pickle.dump(fids, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def fast_ensemble_fid(self):

        num_ensembles = 5
        scores = [NCSN(self.config).to(self.config.device) for _ in range(num_ensembles)]
        scores = [torch.nn.DataParallel(score) for score in scores]

        sigmas_th = get_sigmas(self.config)
        sigmas = sigmas_th.cpu().numpy()

        fids = {}
        ## get the OT solver
        idf_max_step = sigmas.shape[0]*self.config.sampling.n_steps_each
        #OT_solver = get_OT_solver(self.args,idf_max_step)
        OT_solver = get_OT_solver(self.config.OT,idf_max_step)
        remain = self.config.OT.backSteps%self.config.sampling.n_steps_each
        sig_n = sigmas.shape[0]-self.config.OT.backSteps//self.config.sampling.n_steps_each
        if remain>0:
            sig_n-=1
        for ckpt in tqdm.tqdm(range(self.config.fast_fid.begin_ckpt, self.config.fast_fid.end_ckpt + 1, 5000),
                              desc="processing ckpt"):
            begin_ckpt = max(self.config.fast_fid.begin_ckpt, ckpt - (num_ensembles - 1) * 5000)
            index = 0
            for i in range(begin_ckpt, ckpt + 5000, 5000):
                states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{i}.pth'),
                                    map_location=self.config.device)
                scores[index].load_state_dict(states[0])
                scores[index].eval()
                index += 1

            def scorenet(x, labels):
                num_ckpts = (ckpt - begin_ckpt) // 5000 + 1
                return sum([scores[i](x, labels) for i in range(num_ckpts)]) / num_ckpts

            num_iters = self.config.fast_fid.num_samples // self.config.fast_fid.batch_size
            output_path = os.path.join(self.args.image_folder, 'ckpt_{}'.format(ckpt))
            os.makedirs(output_path, exist_ok=True)
            for i in range(num_iters):
                init_samples = torch.rand(self.config.fast_fid.batch_size, self.config.data.channels,
                                          self.config.data.image_size, self.config.data.image_size,
                                          device=self.config.device)
                #init_samples = data_transform(self.config, init_samples)
                init_samples = OT_Map(self.config.OT,init_samples,OT_solver)
                all_samples = anneal_Langevin_dynamics(init_samples, score, sigmas,
                                                       self.config.fast_fid.n_steps_each,
                                                       self.config.fast_fid.step_lr,
                                                       verbose=self.config.fast_fid.verbose,
                                                       denoise=self.config.sampling.denoise,
                                                       start_idx=sig_n,remainder=remain)

                final_samples = all_samples[-1]
                for id, sample in enumerate(final_samples):
                    sample = sample.view(self.config.data.channels,
                                         self.config.data.image_size,
                                         self.config.data.image_size)

                    sample = inverse_data_transform(self.config, sample)

                    save_image(sample, os.path.join(output_path, 'sample_{}.png'.format(id)))

            stat_path = get_fid_stats_path(self.args, self.config, download=True)
            fid = get_fid(stat_path, output_path)
            fids[ckpt] = fid
            print("ckpt: {}, fid: {}".format(ckpt, fid))

        with open(os.path.join(self.args.image_folder, 'fids.pickle'), 'wb') as handle:
            pickle.dump(fids, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

    def fine_tune(self):   
        dataset, test_dataset = get_dataset(self.args, self.config)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=self.config.data.num_workers, drop_last=True)
        test_iter = iter(test_loader)
        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        score = get_model(self.config)
        score = torch.nn.DataParallel(score)
        if self.config.sampling.ckpt_id is None:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'), map_location=self.config.device)
        else:
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{self.config.sampling.ckpt_id}.pth'),
                                map_location=self.config.device)
        score.load_state_dict(states[0], strict=True)
        score = freeze_encoder(self.config,score)
        
        self.config.optim.lr *= 0.1
        optimizer = get_optimizer(self.config, score.parameters())

        start_epoch = 0
        step = 0

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)

        sigmas = get_sigmas(self.config)
        
        idf_max_step = sigmas.shape[0]*self.config.sampling.n_steps_each
        #OT_solver = get_OT_solver(self.args,idf_max_step)
        OT_solver = get_OT_solver(self.config.OT,idf_max_step)
        self.config.OT.h_name = os.path.join(self.args.log_path, 'ot','h_best.pt')
        
        remain = self.config.OT.backSteps%self.config.sampling.n_steps_each
        sig_n = sigmas.shape[0]-self.config.OT.backSteps//self.config.sampling.n_steps_each
        if remain>0:
            sig_n-=1
        used_sigmas = sigmas[sig_n:]
        step_lr = 0.000008
        
        self.args.log_sample_path = os.path.join(self.args.log_path, 'samples')
        if not os.path.exists(self.args.log_sample_path):
            os.makedirs(self.args.log_sample_path)
            
        model_path = os.path.join(self.args.log_path, 'fine_tune')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.args.log_path = model_path
        self.config.sampling.ckpt_id= None

        logging.info("Start fine tuning!")
        for epoch in range(start_epoch, self.config.training.fine_tune_epochs):
            init_samples = torch.rand(self.config.fast_fid.batch_size, self.config.data.channels,
                                          self.config.data.image_size, self.config.data.image_size,
                                          device=self.config.device)
            init_samples = OT_Map(self.config.OT,init_samples,OT_solver)
            batch_samples = init_samples.detach()
            for c, sigma in enumerate(used_sigmas):
                score.train()
                step += 1
                
                step_size = step_lr * (sigma / used_sigmas[-1]) ** 2
                labels = torch.ones(batch_samples.shape[0], device=batch_samples.device) * (c+sig_n)
                labels = labels.long()
                
                loss, reverse = anneal_dsm_score_fine_tune(score, batch_samples, sigma, 
                                        labels,step_size,self.config.sampling.n_steps_each,self.config.training.anneal_power)

                logging.info("step: {}, loss: {}".format(step, loss.item()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                batch_samples = reverse.detach()

                if self.config.model.ema:
                    ema_helper.update(score)

                if step >= self.config.training.n_iters:
                    return 0
                
                if step % 100 == 0:
                    if self.config.model.ema:
                        test_score = ema_helper.ema_copy(score)
                    else:
                        test_score = score

                    test_score.eval()
                    try:
                        test_X, test_y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)

                    test_X = test_X.to(self.config.device)
                    test_X = data_transform(self.config, test_X)

                    with torch.no_grad():
                        test_dsm_loss = anneal_dsm_score_estimation(test_score, test_X, sigmas, None,
                                                                    self.config.training.anneal_power,
                                                                    hook=None)
                        logging.info("step: {}, test_loss: {}".format(step, test_dsm_loss.item()))

                        del test_score


                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(states, os.path.join(model_path, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(model_path, 'checkpoint.pth'))

                    if self.config.training.snapshot_sampling:
                        if self.config.model.ema:
                            test_score = ema_helper.ema_copy(score)
                        else:
                            test_score = score

                        test_score.eval()

                        ## Different part from NeurIPS 2019.
                        ## Random state will be affected because of sampling during training time.
                        init_samples = torch.rand(36, self.config.data.channels,
                                                  self.config.data.image_size, self.config.data.image_size,
                                                  device=self.config.device)
                        init_samples = data_transform(self.config, init_samples)
                        samples = OT_Map(self.config.OT,init_samples,OT_solver)
                        all_samples = anneal_Langevin_dynamics(samples, test_score, sigmas.cpu().numpy(),
                                                               self.config.sampling.n_steps_each,
                                                               self.config.sampling.step_lr,
                                                               final_only=True, verbose=True,
                                                               denoise=self.config.sampling.denoise,
                                                               start_idx=sig_n,remainder=remain)

                        sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                                      self.config.data.image_size,
                                                      self.config.data.image_size)
                        torch.save(sample, os.path.join(self.args.log_sample_path, 'samples_{}.pth'.format(step)))
                        sample = inverse_data_transform(self.config, sample)

                        image_grid = make_grid(sample, 6)
                        save_image(image_grid,
                                   os.path.join(self.args.log_sample_path, 'image_grid_{}.png'.format(step)))
                        
                        del test_score
                        del all_samples


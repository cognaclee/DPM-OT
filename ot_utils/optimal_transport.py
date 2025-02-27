# -*- coding: utf-8 -*-
# @Time        : 13/11/2022 17:10 PM
# @Description :The code is modified from AEOT
# @Author      : li zezeng
# @Email       : zezeng.lee@gmail.com
import sys
import os
import torch
import numpy as np
import math

torch.set_printoptions(precision=8)
class OptimalTransport():	
    def __init__ (self, target_feature, args, device='cuda:0', out_dir='./results/ot_models/'):
        self.tg_fea = target_feature
        self.num_tg = self.tg_fea.shape[0]
        self.dim = self.tg_fea.shape[1]*self.tg_fea.shape[2]*self.tg_fea.shape[3]
        self.max_iter = args.max_iter
        self.lr = args.lr_ot
        self.bat_size_sr = args.bat_size_sr
        self.bat_size_tg = args.bat_size_tg
           
        if self.num_tg % args.bat_size_tg != 0:
        	sys.exit('Error: (num_tg) is not a multiple of (bat_size_tg)')
        
        self.epochs_per_save = 500
        self.out_dir = out_dir

        self.num_bat_sr = 10*self.num_tg // args.bat_size_sr
        self.num_bat_tg = self.num_tg // args.bat_size_tg
        #!internal variables
        '''
        self.d_h: Optimal value of h (the variable to be optimized of the variational Energy).
        self.d_g: The gradient of the energy function E(h).
        '''
        self.device = device
        self.d_h = torch.zeros(self.num_tg, dtype=torch.float, device=self.device)
        self.d_g = torch.zeros(self.num_tg, dtype=torch.float, device=self.device)
        self.d_g_sum = torch.zeros(self.num_tg, dtype=torch.float, device=self.device)
        
        print('Allocated GPU memory: {}MB'.format(torch.cuda.memory_allocated()/1e6))
        print('Cached memory: {}MB'.format(torch.cuda.memory_cached()/1e6))
   
        
    def cal_measure_one_batch(self):
        '''Calculate the pushed-forward measure of current step. 
        '''
        d_volP= torch.rand(self.bat_size_sr, self.dim, device=self.device)
        #d_volP= torch.randn(self.bat_size_sr, self.dim, device=self.device)
        d_tot_ind = torch.empty(self.bat_size_sr, dtype=torch.long, device=self.device)
        d_tot_ind_val = torch.empty(self.bat_size_sr, dtype=torch.float, device=self.device)   
        d_tot_ind_val.fill_(-1e30)
        d_tot_ind.fill_(-1)
        i = 0 
        while i < self.num_bat_tg:
            temp_tg = self.tg_fea[i*self.bat_size_tg:(i+1)*self.bat_size_tg]
            temp_tg = temp_tg.view(temp_tg.shape[0], -1)	

            '''U=PX+H'''
            d_temp_h = self.d_h[i*self.bat_size_tg:(i+1)*self.bat_size_tg]
            d_U = torch.mm(temp_tg, d_volP.t())+ d_temp_h.expand([self.bat_size_sr, -1]).t()
            '''compute max'''
            d_ind_val, d_ind = torch.max(d_U, 0)
            '''add P id offset'''
            d_ind = d_ind+(i*self.bat_size_tg)
            '''store best value'''
            d_tot_ind_val, d_ind_val_argmax = torch.max(torch.stack([d_tot_ind_val, d_ind_val],dim = 0), 0)
            d_tot_ind = torch.stack([d_tot_ind, d_ind],dim = 0)[d_ind_val_argmax, torch.arange(self.bat_size_sr)]
            '''add step'''
            i = i+1
     
        '''calculate histogram'''
        self.d_g.copy_(torch.bincount(d_tot_ind, minlength=self.num_tg))        
        #self.d_g.div_(self.bat_size_tg) 
        
    def cal_measure(self):
        self.d_g_sum.fill_(0)
        for count in range(self.num_bat_sr):       
            self.cal_measure_one_batch()
            self.d_g_sum = self.d_g_sum + self.d_g
        self.d_g = self.d_g_sum/(self.num_bat_sr*self.bat_size_sr)
    
        
    def forward(self,sr_feature):
        ind_len = sr_feature.shape[0]
        d_volP = sr_feature.view(ind_len, -1)
        d_tot_ind = torch.empty(ind_len, dtype=torch.long, device=self.device)
        d_tot_ind_val = torch.empty(ind_len, dtype=torch.float, device=self.device)   
        d_tot_ind_val.fill_(-1e30)
        d_tot_ind.fill_(-1)
        i = 0 
        for i in range(self.num_bat_tg):
            temp_tg = self.tg_fea[i*self.bat_size_tg:(i+1)*self.bat_size_tg]
            temp_tg = temp_tg.view(temp_tg.shape[0], -1)	

            '''U=PX+H'''
            d_temp_h = self.d_h[i*self.bat_size_tg:(i+1)*self.bat_size_tg]
            d_U = torch.mm(temp_tg, d_volP.t())+ d_temp_h.expand([ind_len, -1]).t()
            '''compute max'''
            d_ind_val, d_ind = torch.max(d_U, 0)
            '''add P id offset'''
            d_ind = d_ind+(i*self.bat_size_tg)
            '''store best value'''
            d_tot_ind_val, d_ind_val_argmax = torch.max(torch.stack([d_tot_ind_val, d_ind_val],dim = 0), 0)
            d_tot_ind = torch.stack([d_tot_ind, d_ind],dim = 0)[d_ind_val_argmax, torch.arange(ind_len)]
            '''add step'''
            i = i+1
        return self.tg_fea[d_tot_ind,:,:,:]

    def transfer_topk(self,sr_feature,topk):
        ind_len = sr_feature.shape[0]
        d_volP = sr_feature.view(ind_len, -1)
        d_tot_ind = torch.empty((ind_len,topk), dtype=torch.long, device=self.device)
        d_tot_ind_val = torch.empty((ind_len,topk), dtype=torch.float, device=self.device)   
        d_tot_ind_val.fill_(-1e30)
        d_tot_ind.fill_(-1)
        i = 0 
        for i in range(self.num_bat_tg):
            temp_tg = self.tg_fea[i*self.bat_size_tg:(i+1)*self.bat_size_tg]
            temp_tg = temp_tg.view(temp_tg.shape[0], -1)	

            '''U=PX+H'''
            d_temp_h = self.d_h[i*self.bat_size_tg:(i+1)*self.bat_size_tg]
            d_U = torch.mm(temp_tg, d_volP.t())+ d_temp_h.expand([ind_len, -1]).t()
            '''compute max'''
            #d_ind_val, d_ind = torch.max(d_U, 0)
            d_ind_val, d_ind, I = torch.topk(d_U, topk, dim=0)
            '''add P id offset'''
            d_ind = d_ind+(i*self.bat_size_tg)
            '''store best value'''
            d_tot_ind_val, d_ind_val_argmax = torch.max(torch.stack([d_tot_ind_val, d_ind_val],dim = 0), 0)
            d_tot_ind = torch.stack([d_tot_ind, d_ind],dim = 0)[d_ind_val_argmax, torch.arange(ind_len)]
            '''add step'''
            i = i+1
        return d_tot_ind


    def train_ot(self,target_measures,steps = 1):
        best_g_norm = 1e20
        curr_best_g_norm = 1e20
        #steps = 1
        count_bad = 0
        h_file_list = []
        
        d_adam_m = torch.zeros(self.num_tg, dtype=torch.float, device=self.device)
        d_adam_v = torch.zeros(self.num_tg, dtype=torch.float, device=self.device)
        count_bad = 0
        count_double = 0
        while(steps <= self.max_iter):
            self.cal_measure()	
            
            #if steps % 4000 == 0:
                #self.lr *=0.2
            #bias_grd = self.d_g - 1./self.num_sr
            bias_grd = self.d_g - target_measures
            d_adam_m *= 0.9
            d_adam_m += 0.1*bias_grd
            d_adam_v *= 0.999
            d_adam_v += 0.001*bias_grd*bias_grd
            d_delta_h = -self.lr*torch.div(d_adam_m, torch.add(torch.sqrt(d_adam_v),1e-8))
            self.d_h = self.d_h + d_delta_h
            #self.d_h = self.d_h - self.lr*bias_grd#It will cause the loss to decline very slowly
            '''normalize h'''
            self.d_h -= torch.mean(self.d_h)
            
            g_norm = torch.sqrt(torch.sum(torch.mul(bias_grd,bias_grd)))
              
            
            if (steps+1) % 50 == 0:    
                num_zero = torch.sum(self.d_g == 0)
                ratio_diff = torch.max(bias_grd)   
                print('[{0}/{1}] Max absolute error ratio: {2:.3f}. g norm: {3:.6f}. num zero: {4:d}'.format(
                    steps, self.max_iter, ratio_diff, g_norm, num_zero))

                 
            ''' /h: save  intercept vector of brenier_h function 
            '''
            if g_norm<curr_best_g_norm:
                filename = os.path.join(self.out_dir,'h_best.pt')
                torch.save(self.d_h,filename)
                curr_best_g_norm = g_norm
                count_bad = 0
            else:
                count_bad += 1
                
            if (steps+1) % 1000 == 0 or steps+1 == self.max_iter:
                filename = os.path.join(self.out_dir,'h_{}.pt'.format(steps+1))
                torch.save(self.d_h,filename)
                h_file_list.append(filename)
            
            if len(h_file_list)>6:
                if os.path.exists(h_file_list[0]):
                    os.remove(h_file_list[0])
                h_file_list.pop(0)
            
            if g_norm < 8e-4 and num_zero==0:
                return  
            
            if count_bad > 50 and count_double<3:
                self.num_bat_sr *= 2
                print('self.num_bat_sr has increased to {}'.format(self.bat_size_sr*self.num_bat_sr))
                count_bad = 0
                curr_best_g_norm = 1e20
                self.lr *= 0.8
                count_double +=1
            
            steps += 1


    def set_h(self, h_tensor):
        self.d_h.copy_(h_tensor)     

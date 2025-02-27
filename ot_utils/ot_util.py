import torch
import numpy as np
import scipy.io as sio
import os
from glob import glob
from ot_utils.optimal_transport import OptimalTransport

torch.set_printoptions(precision=8)
#  generate latent code P
def transfer_and_generate(OT_solve,sr_feature, args, device):
    topk = args.topk
    I_all = OT_solve.transfer_topk(sr_feature, topk)
    numX =  I_all.shape[1]
    I_all_2 = -torch.ones([2, (topk-1) * numX], dtype=torch.long, device=device)
    for ii in range(topk-1):
        I_all_2[0, ii * numX:(ii+1) * numX] = I_all[0,:]
        I_all_2[1, ii * numX:(ii+1) * numX] = I_all[ii + 1, :]
    I_all = I_all_2
    
    if torch.sum(I_all < 0) > 0:
        print('Error: numX is not a multiple of bat_size_n')

    ###compute angles
    P = OT_solve.tg_fea.view(OT_solve.num_tg, -1)   
    nm = torch.cat([P, -torch.ones([OT_solve.num_tg,1],device=device)], dim=1)
    nm /= torch.norm(nm,dim=1).view(-1,1)
    cs = torch.sum(nm[I_all[0,:],:] * nm[I_all[1,:],:], 1) #element-wise multiplication
    cs = torch.min(torch.ones([cs.shape[0]],device=device), cs)
    theta = torch.acos(cs)
    print(torch.max(theta))
    theta = (theta-torch.min(theta))/(torch.max(theta)-torch.min(theta))

    ###filter out TLerated samples with theta larger than threshold
    I_TL = I_all[:, theta <= args.angle_thresh]
    I_TL, _ = torch.sort(I_TL, dim=0)
    _, uni_TL_id = np.unique(I_TL[0,:].cpu().numpy(), return_index=True)
    np.random.shuffle(uni_TL_id)
    I_TL = I_TL[:, torch.from_numpy(uni_TL_id)]
     
    numTL = I_TL.shape[1]
    
    ###target features transfer   
    P_TL = OT_solve.tg_fea[I_TL[0,:],:,:,:]
    id_TL = I_TL[0,:].squeeze().cpu().numpy().astype(int)
    TL_feature_path = os.path.join(args.image_folder,'ot_target_features.mat')
    sio.savemat(TL_feature_path, {'features':P_TL, 'ids':id_TL})
    return P_TL

def get_OT_solver(args,idf_max_step):
    #arguments for training OT
    #args.source_dir = os.path.join(args.source_dir, str(idf_max_step-args.backSteps-1), 'pth')
    args.source_dir = os.path.join(args.source_dir, str(args.backSteps), 'pth')
    tg_names = sorted(glob(args.source_dir+'/*.'+'pth'))
    #print('tg_names.len:',len(tg_names))
    tg_list = []
    tg_step = idf_max_step-args.backSteps-1
    #tg_step = idf_max_step-args.backSteps
    for tg_name in tg_names:
        '''
        stp = int(tg_name.split('_')[-1][:-4])
        if stp==tg_step:  
            tg_list.append(torch.load(tg_name))
        '''
        tg_list.append(torch.load(tg_name))
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tg_fea = torch.cat(tg_list,0).to(device)
    num_fea = tg_fea.shape[0]
    print('tg_fea.shape:',tg_fea.shape)
    
    bat_size_tg = args.bat_size_tg
    tg_fea = tg_fea[0:num_fea//bat_size_tg*bat_size_tg,:]
    num_fea = tg_fea.shape[0]
    tg_measures = (torch.ones(num_fea)/num_fea).to(device)
    
    
    ot_dir=os.path.join(args.log_path, 'ot')
    if not os.path.exists(ot_dir):       
        os.makedirs(ot_dir)

    '''train ot'''
    OT_solve = OptimalTransport(tg_fea, args, device, ot_dir)
    if args.h_name is None:
        #OT_solve.set_h(torch.load('./exp/logs/ffhq/ot/h_best_10.pt')) #h_best
        OT_solve.set_h(torch.load('./exp/logs/ffhq/ot/h_best_30.pt')) #h_best
        #OT_solve.set_h(torch.load('./exp/logs/cifar10/ot/h_10000.pt')) #h_best
        OT_solve.train_ot(tg_measures,9000)
    else:
        OT_solve.set_h(torch.load(args.h_name)) 
    print('OT have been successfully solved')
    return OT_solve
    
def OT_Map(args,sr_feature,OT_solve):
    device = sr_feature.device
    '''source features transfer '''
    if args.topk==1:
        gen_feat = OT_solve.forward(sr_feature)
        '''
        tg2sr_idx_name = os.path.join(args.image_folder,'sr2gt_idx.pt')
        torch.save(tg2sr_idx, tg2sr_idx_name)
        tg_fea_name = os.path.join(args.image_folder,'tg_feat.pt')
        torch.save(tg_fea, tg_fea_name)
        '''
    else:
        gen_feat = transfer_and_remove_singular_points(OT_solve,sr_feature, args, device)
        
    print('OT Map successfully transfer {} source feature'.format(gen_feat.shape[0]))
    return gen_feat
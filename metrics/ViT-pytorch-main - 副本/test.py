# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta
import torch

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader, ModeMixDataset
from utils.dist_util import get_world_size
from torchvision import transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image


logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    #model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.pth" % args.name)
    #print(model_checkpoint)
    #aaa
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)



def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy


def train(args, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            loss = model(x, y)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    accuracy_test = valid(args, model, writer, test_loader, global_step)
                    #accuracy_train = valid(args, model, writer, train_loader, global_step)
                    if best_acc < accuracy_test:
                        save_model(args, model)
                        best_acc = accuracy_test
                    model.train()

                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break

    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Best Valid Accuracy: \t%f" % best_acc)
    #logger.info("Best Train Accuracy: \t%f" % accuracy_train)
    logger.info("End Training!")
    
def softmax(x, axis=1):
    row_max = x.max(axis=axis)
    row_max = row_max.reshape(-1, 1)
    hatx = x - row_max
    hatx_exp = np.exp(hatx)
    hatx_sum = np.sum(hatx_exp, axis=axis, keepdims=True)
    s = hatx_exp / hatx_sum
    return s

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--trained_model_dir", type=str, default="output/cifar10-100_500_checkpoint.pth",
                        help="Where to search for trained models.")
                        
    parser.add_argument("--figure_dir", type=str, default="figure/cifar10",
                        help="Where to search for test figures.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)
    
    # Load model
    config = CONFIGS[args.model_type]
    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=10)
    model.load_state_dict(torch.load(args.trained_model_dir))
    model.to(args.device)
    model.eval()
    

    #original_folder = '/user36/code/ViT-pytorch-main/figure/cifar10/old/'
    #original_folder = '/user36/data/DFOT_results/Sample_Images/ddim-main/cifar10/'
    #original_folder = '/user36/data/DFOT_results/Sample_Images/dpm-solver/cifar10/'
    #original_folder = '/user36/data/DFOT_results/Sample_Images/edm/cifar10/'
    #original_folder = '/user36/data/DFOT_results/Sample_Images/ncsnv2-master/cifar10/'
    #original_folder = '/user36/data/DFOT_results/Ours/cifar10/fid_50/'
    #original_folder = '/user36/data/DFOT_results/Ours/cifar10/fid_30/'
    #original_folder = '/user36/data/DFOT_results/Ours/cifar10/fid_20/'
    #original_folder = '/user36/data/DFOT_results/Ours/cifar10/fid_10/'
    #original_folder = '/user36/data/DFOT_results/Ours/cifar10/fid_5/'
    #original_folder = '/user36/code/ViT-pytorch-main/figure/cifar10/mode_mixture_test_figure/'
    original_folder = '/user36/code/ViT-pytorch-main/figure/2/'
    new_folder = '/user36/code/ViT-pytorch-main/figure/cifar10/observe/'
    
    logger.info("** The calculation of mode mixing metric Start! \n")
    mode_mix_data = ModeMixDataset(data_path=original_folder)
    batch_size = 64
    mode_mix_loader = DataLoader(mode_mix_data,batch_size=batch_size,shuffle=False,pin_memory=True,num_workers=1)
    A = np.zeros(8) # 0.1, 0.2, 0.3, 0.4, 0.8, 0.9, 0.13, 0.16
    m = 0
    for step, sample_batched in enumerate(mode_mix_loader, 0):
        imgs, labels = sample_batched[0], sample_batched[1]
        imgs = imgs.to(args.device)
        output = model(imgs)[0]
        S = softmax(output.detach().cpu().numpy())
        print(S)
        res = np.sum(S>=0.4,axis=1)
        A[3] = A[3]+np.sum(res>=2)
        #####
        #if np.sum(res>=2):
        #    IMGS_flag = np.where((res>=2)==True)[0]
        #    print(IMGS_flag)
        #    for IMG_flag in IMGS_flag:
        #        IMG = Image.open(mode_mix_data.img_names[IMG_flag+step*64])
        #        IMG.save(os.path.join(new_folder, str(torch.argmax(output, dim=-1)[IMG_flag])+'_'+str(m)+'.png'))
        #        m = m+1
        #####
        res = np.sum(S>=0.3,axis=1)
        A[2] = A[2]+np.sum(res>=2)
        
        res = np.sum(S>=0.2,axis=1)
        A[1] = A[1]+np.sum(res>=2)
        
        res = np.sum(S>=0.1,axis=1)
        A[0] = A[0]+np.sum(res>=2)
        
        res = np.sum(S>=0.13,axis=1)
        A[4] = A[4]+np.sum(res>=2)
        
        res = np.sum(S>=0.16,axis=1)
        A[5] = A[5]+np.sum(res>=2)
        
        res = np.sum(S>=0.115,axis=1)
        A[6] = A[6]+np.sum(res>=2)
        
        res = np.sum(S>=0.11,axis=1)
        A[7] = A[7]+np.sum(res>=2)
        ######
        #res = np.sum(S>=0.15,axis=1)
        #print('*'*20)
        #print((res>=2),labels)
        #print((res>=2)==np.array(labels))
        #if np.sum(res>=2):
            #IMGS_flag = np.where(((res>=2)==np.array(labels))==False)[0]
            #print(len(IMGS_flag))
            #for IMG_flag in IMGS_flag:
            #    print(mode_mix_data.img_names[IMG_flag+step*64])
        m = m+np.sum((res>=2)==np.array(labels))
        ######
        
        
    print(A)
    logging.info(A)
    logging.info(m)
    ############################

    
    # Testing
    #train(args, model)


if __name__ == "__main__":
    main()

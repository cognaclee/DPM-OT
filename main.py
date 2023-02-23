import argparse
import traceback
import time
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb
import copy
from runners import *

import os

def str2bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ['yes', 'true', 't', 'y']:
        return True
    elif val.lower() in ['no', 'false', 'f', 'n']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--config', type=str, default='cifar10.yml', help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, default='ffhq', help='A string for documentation purpose. '
                                                               'Will be the name of the log folder.')#cifar10
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--test', action='store_true', help='Whether to test the model')
    parser.add_argument('--sample', action='store_true', help='Whether to produce samples from the model')
    parser.add_argument('--fast_fid', action='store_true', help='Whether to do fast fid test')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    parser.add_argument('-i', '--image_folder', type=str, default='cifar10', help="The folder name of samples")
    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")
    
    parser.add_argument('--gene_source', default=False, type=str2bool, help='whether generate source features')#False
    parser.add_argument('--fine_tune', default=False, type=str2bool, help='whether fine tune the pretained model')#False
    '''
    parser.add_argument('--backSteps', type=int, default=10, help='The number of steps requiring reverse diffusion')
    parser.add_argument('--topk', type=int, default=1, help='The nearest k samples around current sample')
    parser.add_argument('--h_name', type=str, default=None, help='file name of OT Brenier h')#'./exp/logs/cifar10/ot/h_10000.pt'
    parser.add_argument('--source_dir', type=str, default='./exp/image_samples/cifar10/pth/pth/', help='source file directory')
    parser.add_argument('--max_iter', type=int, default=10000,help='max iters of train ot')#待调参数
    parser.add_argument('--lr_ot', type=float, default=1e-1,help='learning rate of OT')#待调参数
    parser.add_argument('--bat_size_sr', type=int, default=10000,help='Size of mini-batch of Monte-Carlo source samples on device')
    parser.add_argument('--bat_size_tg', type=int, default=1000,help='Size of mini-batch of Monte-Carlo target samples on device')
    parser.add_argument('--angle_thresh', type=float, default=0.7,help='the threshold of the angle between two samples')
    '''

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, 'logs', args.doc)

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, 'tensorboard', args.doc)

    if not args.test and not args.sample and not args.fast_fid:
        if not args.resume_training:
            if os.path.exists(args.log_path):
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input("Folder already exists. Overwrite? (Y/N)")
                    if response.upper() == 'Y':
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.log_path)
                    shutil.rmtree(tb_path)
                    os.makedirs(args.log_path)
                    if os.path.exists(tb_path):
                        shutil.rmtree(tb_path)
                else:
                    print("Folder exists. Program halted.")
                    sys.exit(0)
            else:
                os.makedirs(args.log_path)

            with open(os.path.join(args.log_path, 'config.yml'), 'w') as f:
                yaml.dump(new_config, f, default_flow_style=False)

        new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, 'stdout.txt'))
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)
        
        os.makedirs(os.path.join(args.exp, 'image_samples'), exist_ok=True)
        args.image_folder = os.path.join(args.exp, 'image_samples', args.image_folder)

        if args.sample:
            fid_path = os.path.join(args.image_folder, 'fid')
            if not os.path.exists(fid_path):
                os.makedirs(fid_path)

        elif args.fast_fid:
            os.makedirs(os.path.join(args.exp, 'fid_samples'), exist_ok=True)
            args.image_folder = os.path.join(args.exp, 'fid_samples', args.image_folder)
            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)
            else:
                overwrite = False
                if args.ni:
                    overwrite = False
                else:
                    response = input("Image folder already exists. \n "
                                     "Type Y to delete and start from an empty folder?\n"
                                     "Type N to overwrite existing folders (Y/N)")
                    if response.upper() == 'Y':
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.image_folder)
                    os.makedirs(args.image_folder)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device
    new_config.OT.log_path = args.log_path
    new_config.OT.image_folder = args.image_folder

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True
    if args.gene_source:
        new_config.sampling.final_only=False

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Config =")
    print(">" * 80)
    config_dict = copy.copy(vars(config))
    if not args.test and not args.sample and not args.fast_fid:
        del config_dict['tb_logger']
    #print(yaml.dump(config_dict, default_flow_style=False))
    print("<" * 80)

    try:
        runner = NCSNRunner(args, config)
        if args.test:
            runner.test()
        elif args.sample:
            if args.gene_source:
                runner.sample(batch_numb=205)
            else:
                runner.set_sampling_fid(True)
                runner.set_sampling_final_only(True)
            
            if args.fine_tune:
                runner.fine_tune()
            runner.sample()
        elif args.fast_fid:
            runner.fast_fid()
        else:
            runner.train()
    except:
        logging.error(traceback.format_exc())

    return 0


if __name__ == '__main__':
    sys.exit(main())

import logging
from glob import glob
import torch
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, Dataset
import numpy as np

logger = logging.getLogger(__name__)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None

    elif args.dataset == "cifar100":
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None
    elif args.dataset == "tiny_imagenet_200":
        original_folder_trainset = '/user36/code/ViT-pytorch-main/data/tiny_imagenet_200/train/'
        trainset = ModeMixDataset(data_path=original_folder_trainset)
        
        original_folder_testset = '/user36/code/ViT-pytorch-main/data/tiny_imagenet_200/test/'
        testset = ModeMixDataset(data_path=original_folder_testset)
        
                                    
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader
    
    
class ModeMixDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        if transform is None:
            self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform
        
        self.img_names = sorted(glob(self.data_path+'/*.'+'jpg')) #self.img_names = sorted(glob(self.data_path+'/*.'+'JPEG')) 
        self.start = len(self.data_path)
        self.num_img = len(self.img_names)
        
        print('self.num_mesh=',self.num_img)
        #print('self.num_mesh=',self.img_names)
    
    def __len__(self):
        """
        Return the length of data here
        """
        return self.num_img

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        filename = self.img_names[index]
        #print(filename)
        sample = Image.open(filename)
        #sample = np.array(sample).shape
        #print(np.array(sample).shape)
        #if len(np.array(sample).shape)==2: #or np.array(sample).shape[2]==1:
        #    #print(filename)
        #    #print(np.array(sample).shape)
        #    sample = np.array(sample)[:,:,np.newaxis]
        #    sample = np.repeat(sample,3,axis=2)
        #    sample = Image.fromarray(sample)
        #print(sample.size)
        #print(type(sample))
        if self.transform is not None:
            sample = self.transform(sample)
            #print(sample)
            #print(sample.shape)
            #print(aaa)
        #print(filename[self.start:self.start+3])
        labels = 0#int(filename[self.start])#0#int(filename[self.start+1:self.start+4]) # int(filename[59:62])
        print(labels)
        return sample, labels 

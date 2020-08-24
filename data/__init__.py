import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor
from data.datasets import get_dataset
from data.transform import RandomImgAugment

def create_test_dataloader(args):

    joint_transform_list = [
        RandomImgAugment(True, 
                        True,
                        Image.BICUBIC)]

    joint_transform = Compose(joint_transform_list)
       
    dataset = get_dataset(root=args.root, data_file=args.test_data_file, phase='test',
                        dataset=args.dataset, joint_transform=joint_transform)
    loader = torch.utils.data.DataLoader(
                                        dataset,
                                        batch_size=1, shuffle=False,
                                        num_workers=int(args.nThreads),
                                        pin_memory=True)
    
    return loader

def create_dataloader(args):

    return create_test_dataloader(args)

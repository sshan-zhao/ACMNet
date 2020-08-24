import torch
import time
import numpy as np
from PIL import Image
import os
from torchvision.transforms import ToPILImage
import cv2

def compute_errors(ground_truth, prediction):

    # accuracy
    threshold = np.maximum((ground_truth / prediction),(prediction / ground_truth))
    a1 = (threshold < 1.25 ).mean()
    a2 = (threshold < 1.25**2 ).mean()
    a3 = (threshold < 1.25**3 ).mean()

    # mm  
    # MSE
    rmse = (ground_truth * 1000 - prediction * 1000) ** 2
    rmse = np.sqrt(rmse.mean())
    
    # MAE
    mae = np.fabs(ground_truth * 1000 - prediction * 1000)
    mae = mae.mean()

    # 1/km
    # iMSE
    irmse = (1000 / ground_truth - 1000 / prediction) ** 2
    irmse = np.sqrt(irmse.mean())
    
    # iMAE
    imae = np.fabs(1000 / ground_truth - 1000 / prediction)

    #print(prediction.min())
    #print(ground_truth.min())
    #print(imae.max())
    imae = imae.mean()
    rel = (np.fabs(ground_truth - prediction) / ground_truth).mean()

    return mae, rmse, imae, irmse, a1, a2, a3, rel
    
# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) 
    image_numpy = image_numpy * 255
    return image_numpy.astype(imtype)

def tensor2depth(input_depth, imtype=np.int32):
    if isinstance(input_depth, torch.Tensor):
        depth_tensor = input_depth.data
    else:
        return input_depth
    depth_numpy = depth_tensor[0].cpu().float().numpy() 
    depth_numpy = depth_numpy.reshape((depth_numpy.shape[1], depth_numpy.shape[2]))
    return depth_numpy.astype(imtype)

def save_image(image_numpy, image_path, imtype):
    image_pil = Image.fromarray(image_numpy, imtype)
    image_pil.save(image_path)

class SaveResults:
    def __init__(self, opt):
       
        self.img_dir = os.path.join(opt.checkpoints_dir, opt.expr_name, 'image')
        mkdirs(self.img_dir) 
        self.log_name = os.path.join(opt.checkpoints_dir, opt.expr_name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)
        self.dataset = opt.dataset

    def save_current_results(self, visuals, epoch):
            
        for label, image in visuals.items():
            img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
            if image is None:
                continue
            if 'img' not in label:                
                if self.dataset in ['kitti']:
                    
                    if image.max() <= 1:
                        scale = 1
                    else:
                        scale = image.max()
                    image *= 255
                    depth_numpy = tensor2depth(image, imtype=np.uint16)
                    cv2.imwrite(img_path, depth_numpy)
            else:
                image_numpy = tensor2im(image)
                save_image(image_numpy, img_path, 'RGB')
            

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, i, lr, losses, t, t_data):
          
        message = '(epoch: %d, iters: %d, lr: %e, time: %.3f, data: %.3f) ' % (epoch, i, lr, t, t_data)
        for k, v in losses.items():
            message += '%s: %.6f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_validation_errors(self, mae, rmse, imae, irmse, a1, a2, a3, a4, time):
        message = '(mae: %.3f, rmse: %.3f, imae: %.3f, irmse: %.3f, a1: %.3f, \
                    a2: %.3f, a3: %.3f, a4: %.3f, time/img: %0.5f )' % (mae, rmse, imae, irmse, a1, a2, a3, a4, time)
        print(message)
        with open(self.log_name, 'a') as log_file:
            log_file.write('%s\n' % message)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

import torch.nn
import numpy as np
import os
from PIL import Image
from options.test_options import TestOptions
from data import create_dataloader
from models import create_model
import util
import cv2 

if __name__ == '__main__':
    opt = TestOptions().parse()
    data_loader = create_dataloader(opt)
    num_samples = len(data_loader)   
    print('#test images = %d' % num_samples)
    
    model = create_model(opt)
    model.setup(opt)
    total_steps = 0
    model.eval()

    if opt.suffix != '':
        opt.suffix = '_' + opt.suffix
    dirs = os.path.join('results', opt.model+opt.suffix)
    os.makedirs(dirs) 

    for ind, data in enumerate(data_loader):
        print(ind)
        model.set_input(data)        
        model.test()

        visuals = model.get_current_visuals()

        pred_depth = np.squeeze(visuals['pred'].data.cpu().numpy())
    
        pred_depth[pred_depth<=0.9] = 0.9
        pred_depth[pred_depth>85] = 85
        pred_depth *= 256.0
        pred_depth = pred_depth.astype(np.uint16)
        cv2.imwrite('%s/%010d.png'%(dirs, ind), pred_depth)
        


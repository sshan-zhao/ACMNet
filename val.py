import torch.nn
import numpy as np
import os
from PIL import Image
from options.test_options import TestOptions
from data import create_dataloader
from models import create_model
import util

def ToFalseColors(depth, mask=None):
    color_map = np.array([[0,0,0,114],[0,0,1,185],[1,0,0,114],[1,0,1,174],
                     [0,1,0,114],[0,1,1,185],[1,1,0,114],[1,1,1,0]],dtype=np.float32)
    sum = 0.0
    for i in range(8):
        sum += color_map[i][3]

    weights = np.zeros([8], dtype=np.float32)
    cumsum = np.zeros([8], dtype=np.float32)
    for i in range(7):
        weights[i] = sum / color_map[i][3]
        cumsum[i+1] = cumsum[i] + color_map[i][3] / sum
    H, W = depth.shape
    image = np.ones([H, W, 3], dtype=np.uint8)
    max_depth = np.max(depth)
    for i in range(int(H)):
        for j in range(int(W)):
            val = np.min([depth[i, j]/max_depth, 1.0])
            for k in range(7):
                if val<cumsum[k+1]:
                    break
            w = 1.0 - (val-cumsum[k]) * weights[k]
            r = int((w*color_map[k][0]+(1.0-w)*color_map[k+1][0]) * 255.0)
            g = int((w*color_map[k][1]+(1.0-w)*color_map[k+1][1]) * 255.0)
            b = int((w*color_map[k][2]+(1.0-w)*color_map[k+1][2]) * 255.0)
            image[i, j, 0] = r
            image[i, j, 1] = g
            image[i, j, 2] = b
    if mask is not None:
        image[:,:,0] = image[:,:,0] * mask + 255 * (1-mask)
        image[:,:,1] = image[:,:,1] * mask + 255 * (1-mask)
        image[:,:,2] = image[:,:,2] * mask + 255 * (1-mask)
    return image.astype(np.uint8)

if __name__ == '__main__':
    opt = TestOptions().parse()
    data_loader = create_dataloader(opt)
    num_samples = len(data_loader)   
    print('#test images = %d' % num_samples)
    
    model = create_model(opt)
    model.setup(opt)
    total_steps = 0
    model.eval()

    if opt.save:
        if opt.suffix != '':
            opt.suffix = '_' + opt.suffix
        dirs = os.path.join('results', opt.model+opt.suffix)
        os.makedirs(dirs) 

    mae    = np.zeros(num_samples, np.float32)
    rmse   = np.zeros(num_samples, np.float32)
    imae   = np.zeros(num_samples, np.float32)
    irmse  = np.zeros(num_samples, np.float32)
    a1     = np.zeros(num_samples, np.float32)
    a2     = np.zeros(num_samples, np.float32)
    a3     = np.zeros(num_samples, np.float32)
    a4     = np.zeros(num_samples, np.float32)    
    for ind, data in enumerate(data_loader):
        print(ind)
        model.set_input(data)        
        model.test()

        visuals = model.get_current_visuals()

        gt_depth = np.squeeze(data['gt'].data.cpu().numpy())
        pred_depth = np.squeeze(visuals['pred'].data.cpu().numpy())
        s_depth = np.squeeze(data['sparse'].data.cpu().numpy())
    
        pred_depth[pred_depth<=0.9] = 0.9
        pred_depth[pred_depth>85] = 85
        mask = (gt_depth > 0) & (gt_depth<=100)
        
        mae[ind], rmse[ind], imae[ind], irmse[ind], a1[ind], \
            a2[ind], a3[ind], a4[ind] = util.compute_errors(gt_depth[mask], pred_depth[mask])
        
        if opt.save:
            gt_depth = gt_depth[96:,:]            
            s_depth = s_depth[96:,:]            
            pred_depth = pred_depth[96:,:]            
            gt_image = ToFalseColors(gt_depth, mask=(gt_depth>0).astype(np.float32))
            pred_image = ToFalseColors(pred_depth)
            s_image = ToFalseColors(s_depth, mask=(s_depth>0).astype(np.float32))
        
            gt_img = Image.fromarray(gt_image, 'RGB')
            pred_img = Image.fromarray(pred_image, 'RGB')
            s_img = Image.fromarray(s_image, 'RGB')
            gt_img.save('%s/%05d_gt.png'%(dirs, ind))
            pred_img.save('%s/%05d_pred.png'%(dirs, ind))
            s_img.save('%s/%05d_sparse.png'%(dirs, ind))
            im = util.tensor2im(visuals['img'])
            util.save_image(im, '%s/%05d_img.png'%(dirs, ind), 'RGB')
        
    print(mae.mean(), rmse.mean(), imae.mean(), irmse.mean(), a1.mean(), a2.mean(), a3.mean(), a4.mean())


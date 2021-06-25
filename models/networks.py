import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.nn import init, Parameter
import torch.nn.functional as F
from torch.nn import DataParallel
from point_utils import knn_operation, gather_operation, grouping_operation
import numpy as np
###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            #lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            lr_l = (1-epoch/float(opt.niter))**0.9
            if epoch>opt.niter-1:
                lr_l = (1-(opt.niter-1)/float(opt.niter))**0.9*0.9**(epoch-opt.niter+1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.5)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if classname.find('NConv2d') == -1 and hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
    
def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], need_init=True):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    if need_init:
    	init_weights(net, init_type, gain=init_gain)
    return net


def deconv(in_channels, out_channels, kernel_size=4, padding=1, stride=2, relu=True):
    layers = []
    layers += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)]
    if relu:
        layers += [nn.ReLU(inplace=True)]

    return nn.Sequential(*layers)

def conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, relu=True):
    layers = []
    layers += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)]
    if relu:
        layers += [nn.ReLU(inplace=True)]

    return nn.Sequential(*layers)

def gen_3dpoints(depth, K, levels=3, knn=[9], nsamples=[10000]):
    n, c, h, w = depth.shape
    xx = torch.arange(0, w).view(1, -1).repeat(h, 1).float().cuda().view(1, 1, h, w).repeat(n, 1, 1, 1)
    yy = torch.arange(0, h).view(-1, 1).repeat(1, w).float().cuda().view(1, 1, h, w).repeat(n, 1, 1, 1)

    assert len(knn) == levels and len(nsamples) == levels
    
    spoints = []
    sidxs = []
    nnidxs = []
    masks = []
    for i in range(1, levels+1):
        depth, max_ind = F.max_pool2d(depth, kernel_size=2, stride=2, return_indices=True)
        xy = torch.cat([xx, yy], 1).view(n, 2, -1)
        xy = gather_operation(xy, max_ind.view(n, -1).int())
        xx = xy[:, 0, :].view(n, 1, h//2**i, w//2**i)
        yy = xy[:, 1, :].view(n, 1, h//2**i, w//2**i)
        
        mask = (depth > 0).int()
        new_mask = torch.zeros_like(mask.view(n, -1))

        # sampling
        vp_num = torch.sum(mask, (2,3)).min()
        num_sam = nsamples[i-1]          
        for j in range(n):
            
            all_idx = torch.arange(mask.shape[2]*mask.shape[3]).reshape(1, -1).cuda().int()
            v_idx = all_idx[mask[j].reshape(1, -1)>0].reshape(1,-1)
            sample = torch.randperm(mask[j].sum())
            
            if vp_num < num_sam:
                v_idx = v_idx[:, sample[:int(vp_num)]]
            else:
                v_idx = v_idx[:, sample[:int(num_sam)]]

            if j == 0:
                s_idx = v_idx
            else:
                s_idx = torch.cat([s_idx, v_idx], 0)

            new_mask[j, v_idx[0].long()] = 1
        
        # gather and 3d points
        xyd = torch.cat((xx, yy, depth), 1).view(n, 3, -1)
        s_pts = gather_operation(xyd, s_idx).permute(0, 2, 1) 
        cxy = torch.zeros(n,1,3).float().to(depth.get_device())
        fxy = torch.ones(n,1,3).float().to(depth.get_device())
        cxy[:,0,0] = K[:,0,2]
        cxy[:,0,1] = K[:,1,2]
        fxy[:,0,0] = K[:,0,0]
        fxy[:,0,1] = K[:,1,1]
        s_p3d = (s_pts - cxy) / fxy
        s_p3d[:,:,0:2] = s_p3d[:,:,0:2] * s_pts[:,:,2:]

        # knn
        #nnidx = knn_operation(s_p3d, s_p3d, knn[i-1])
        r=torch.sum(s_p3d*s_p3d, dim=2, keepdim=True)
        m=torch.matmul(s_p3d, s_p3d.transpose(2,1))
        d = r-2*m + r.transpose(2,1)
        _, nnidx=torch.topk(d, k=knn[i-1], dim=-1, largest=False)
        nnidx = nnidx.int()

        spoints.append(s_p3d.permute(0, 2, 1))
        sidxs.append(s_idx)
        nnidxs.append(nnidx)
        new_mask = new_mask.view(n, 1, h//2**i, w//2**i)
        masks.append(new_mask)
    
    return spoints, sidxs, nnidxs, masks

##############################################################################
# Classes
##############################################################################
class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()

    
    def forward(self, depth, image):
        def gradient_x(img):
            gx = img[:,:,:-1,:] - img[:,:,1:,:]
            return gx

        def gradient_y(img):
            gy = img[:,:,:,:-1] - img[:,:,:,1:]
            return gy

        depth_grad_x = gradient_x(depth)
        depth_grad_y = gradient_y(depth)
        image_grad_x = gradient_x(image)
        image_grad_y = gradient_y(image)

        weights_x = torch.exp(-torch.mean(torch.abs(image_grad_x),1,True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_grad_y),1,True))
        smoothness_x = depth_grad_x*weights_x
        smoothness_y = depth_grad_y*weights_y

        loss_x = torch.mean(torch.abs(smoothness_x))
        loss_y = torch.mean(torch.abs(smoothness_y))

        loss = loss_x + loss_y
        
        return loss

class MLP(nn.Module):
    def __init__(self, in_channels):
        super(MLP, self).__init__()

        self.ln1 = nn.Linear(in_channels, in_channels//2)
        self.ln2 = nn.Linear(in_channels//2, 1)

    def forward(self, x):

       return self.ln2(F.leaky_relu(self.ln1(x), 0.2))


class CoAttnGPBlock(nn.Module):
    def __init__(self, in_channels=64, channels=64, knn=9, downsample=False):
        super(CoAttnGPBlock, self).__init__()

        if downsample:
            stride = 2
        else:
            stride = 1
        self.d_conv0 = conv2d(in_channels, channels, stride=stride)
        self.d_conv1 = conv2d(in_channels, channels, stride=stride, relu=False)
        self.d_conv2 = conv2d(channels, channels, relu=False)
        self.r_conv0 = conv2d(in_channels, channels, stride=stride)
        self.r_conv1 = conv2d(in_channels, channels, stride=stride, relu=False)
        self.r_conv2 = conv2d(channels, channels, relu=False)

        self.d_mlp = MLP(channels*2+3)
        self.r_mlp = MLP(channels*2+3)
        self.d_bias = Parameter(torch.zeros(channels))
        self.r_bias = Parameter(torch.zeros(channels))
    """
    d_feat, r_feat: b*c*h*w
    spoints: b*3*ns
    sidxs: b*ns
    nnidxs: b*ns*k
    masks: b*1*h*w
    """
    def forward(self, d_feat, r_feat, spoints, sidxs,  nnidxs, masks, nsamples):

        d_feat0 = self.d_conv0(d_feat)
        d_feat1 = self.d_conv1(d_feat)
        r_feat0 = self.r_conv0(r_feat)
        r_feat1 = self.r_conv1(r_feat)

        b, c, h, w = d_feat0.shape
        k = nnidxs.shape[2]
        
        d_sfeat = gather_operation(d_feat0.view(b, c, -1), sidxs)
        r_sfeat = gather_operation(r_feat0.view(b, c, -1), sidxs)

        nnpoints = grouping_operation(spoints, nnidxs) 
        d_nnfeat = grouping_operation(d_sfeat, nnidxs) 
        r_nnfeat = grouping_operation(r_sfeat, nnidxs) 

        points_dist = (nnpoints - spoints.view(b, 3, -1, 1)).view(b, 3, -1)
        d_feat_dist = (d_nnfeat - d_sfeat.view(b, c, -1, 1)).view(b, c, -1)
        r_feat_dist = (r_nnfeat - r_sfeat.view(b, c, -1, 1)).view(b, c, -1) 
        feats = torch.cat((d_feat_dist, r_feat_dist, points_dist), 1).permute(0, 2, 1) 

        d_attn = torch.softmax(self.d_mlp(feats).view(b, -1, k, 1), 2).permute(0, 3, 1, 2) 
        r_attn = torch.softmax(self.r_mlp(feats).view(b, -1, k, 1), 2).permute(0, 3, 1, 2)

        d_feat = torch.sum(d_attn * d_nnfeat, 3) + self.d_bias.view(1, c, 1)
        r_feat = torch.sum(r_attn * r_nnfeat, 3) + self.r_bias.view(1, c, 1)
        
        d_feat_new = torch.zeros(b, c, h*w).to(d_feat.get_device())
        r_feat_new = torch.zeros(b, c, h*w).to(d_feat.get_device())
        for i in range(b):
            d_feat_new[i, :, sidxs[i].long()] = d_feat[i]
            r_feat_new[i, :, sidxs[i].long()] = r_feat[i]

        masks = masks.float()
        d_feat0 = (1 - masks) * d_feat0 + d_feat_new.view(b, c, h, w)
        r_feat0 = (1 - masks) * r_feat0 + r_feat_new.view(b, c, h, w)

        d_feat2 = self.d_conv2(d_feat0)
        r_feat2 = self.r_conv2(r_feat0)

        return F.relu_(d_feat2+d_feat1), F.relu_(r_feat2+r_feat1)


class ResBlock(nn.Module):
    def __init__(self, in_channels=64, channels=64, downsample=False):
        super(ResBlock, self).__init__()

        if downsample:
            stride = 2
        else:
            stride = 1
        self.conv0 = conv2d(in_channels, channels, stride=stride)
        self.conv1 = conv2d(in_channels, channels, stride=stride, relu=False)
        self.conv2 = conv2d(channels, channels, relu=False)

    def forward(self, feat):

        feat0 = self.conv0(feat)
        feat1 = self.conv1(feat)

        feat2 = self.conv2(feat0)

        return F.relu_(feat2+feat1)


class DCOMPNet(nn.Module):
    def __init__(self, channels=64, knn=[6,6,6], nsamples=[10000,5000,2500], scale=80):
        super(DCOMPNet, self).__init__()

        self.knn = [x + 1 for x in knn]
        self.nsamples = nsamples
        self.scale = scale

        self.d_conv00 = conv2d(1, 32)
        self.d_conv01 = conv2d(32, 32)
        self.r_conv00 = conv2d(3, 32)
        self.r_conv01 = conv2d(32, 32)

        self.cpblock10 = CoAttnGPBlock(32, channels, knn, True)
        self.cpblock11 = CoAttnGPBlock(channels, channels, knn, False)
        self.cpblock20 = CoAttnGPBlock(channels, channels, knn, True)
        self.cpblock21 = CoAttnGPBlock(channels, channels, knn, False)
        self.cpblock30 = CoAttnGPBlock(channels, channels, knn, True)
        self.cpblock31 = CoAttnGPBlock(channels, channels, knn, False)

        self.d_gate4 = conv2d(channels, channels, relu=False)
        self.d_resblock40 = ResBlock(channels*2, channels, False)
        self.d_resblock41 = ResBlock(channels, channels, False)
        self.d_deconv3 = deconv(channels, channels)
        self.d_gate3 = conv2d(channels, channels, relu=False)
        self.d_resblock50 = ResBlock(channels*3, channels, False)
        self.d_resblock51 = ResBlock(channels, channels, False)
        self.d_deconv2 = deconv(channels, channels)
        self.d_gate2 = conv2d(channels, channels, relu=False)
        self.d_resblock60 = ResBlock(channels*3, channels, False)
        self.d_resblock61 = ResBlock(channels, channels, False)
        self.d_deconv1 = deconv(channels, channels)
        self.d_gate1 = conv2d(channels, 32, relu=False)
        self.d_last_conv = conv2d(channels+64, 32)
        self.d_out = nn.Conv2d(32, 1, kernel_size=1, padding=0)

        self.r_gate4 = conv2d(channels, channels, relu=False)
        self.r_resblock40 = ResBlock(channels*2, channels, False)
        self.r_resblock41 = ResBlock(channels, channels, False)
        self.r_deconv3 = deconv(channels, channels)
        self.r_gate3 = conv2d(channels, channels, relu=False)
        self.r_resblock50 = ResBlock(channels*3, channels, False)
        self.r_resblock51 = ResBlock(channels, channels, False)
        self.r_deconv2 = deconv(channels, channels)
        self.r_gate2 = conv2d(channels, channels, relu=False)
        self.r_resblock60 = ResBlock(channels*3, channels, False)
        self.r_resblock61 = ResBlock(channels, channels, False)
        self.r_deconv1 = deconv(channels, channels)
        self.r_gate1 = conv2d(channels, 32, relu=False)
        self.r_last_conv = conv2d(channels+64, 32)
        self.r_out = nn.Conv2d(32, 1, kernel_size=1, padding=0)

        self.f_conv4_1 = conv2d(channels*2, channels, relu=False)
        self.f_conv4_2 = nn.Sequential(conv2d(channels, channels, stride=2), conv2d(channels, channels, relu=False))
        self.f_deconv3 = deconv(channels, channels)
        self.f_conv3_1 = conv2d(channels*3, channels, relu=False)
        self.f_conv3_2 = nn.Sequential(conv2d(channels, channels, stride=2), conv2d(channels, channels, relu=False))
        self.f_deconv2 = deconv(channels, channels)
        self.f_conv2_1 = conv2d(channels*3, channels, relu=False)
        self.f_conv2_2 = nn.Sequential(conv2d(channels, channels, stride=2), conv2d(channels, channels, relu=False))
        self.f_deconv1 = deconv(channels, channels)
        self.f_conv1_1 = conv2d(channels+64, 32, relu=False)
        self.f_conv1_2 = nn.Sequential(conv2d(32, 32, stride=2), conv2d(32, 32, relu=False))
        self.f_out = nn.Conv2d(32, 1, kernel_size=1, padding=0)

    def forward(self, depth, rgb, K):
       
        spoints, sidxs, nnidxs, masks = gen_3dpoints(depth, K, 3, self.knn, self.nsamples)
        d_feat0 = self.d_conv01(self.d_conv00(depth/self.scale))
        r_feat0 = self.r_conv01(self.r_conv00(rgb))

        d_feat1, r_feat1 = self.cpblock10(d_feat0, r_feat0, spoints[0], sidxs[0], nnidxs[0], masks[0], self.nsamples[0])
        d_feat1, r_feat1 = self.cpblock11(d_feat1, r_feat1, spoints[0], sidxs[0], nnidxs[0], masks[0], self.nsamples[0])
        d_feat2, r_feat2 = self.cpblock20(d_feat1, r_feat1, spoints[1], sidxs[1], nnidxs[1], masks[1], self.nsamples[1])
        d_feat2, r_feat2 = self.cpblock21(d_feat2, r_feat2, spoints[1], sidxs[1], nnidxs[1], masks[1], self.nsamples[1])
        d_feat3, r_feat3 = self.cpblock30(d_feat2, r_feat2, spoints[2], sidxs[2], nnidxs[2], masks[2], self.nsamples[2])
        d_feat3, r_feat3 = self.cpblock31(d_feat3, r_feat3, spoints[2], sidxs[2], nnidxs[2], masks[2], self.nsamples[2])

        d_gate4 = torch.sigmoid(self.d_gate4(d_feat3))
        d_feat = self.d_resblock40(torch.cat([d_feat3, d_gate4*r_feat3], 1))
        d_feat = self.d_resblock41(d_feat)
        r_gate4 = torch.sigmoid(self.r_gate4(r_feat3))
        r_feat = self.r_resblock40(torch.cat([r_feat3, r_gate4*d_feat3], 1))
        r_feat = self.r_resblock41(r_feat)

        f_feat = self.f_conv4_1(torch.cat([d_feat, r_feat], 1))
        f_feat_res = F.interpolate(self.f_conv4_2(F.relu(f_feat)), scale_factor=2, mode='bilinear')
        f_feat = self.f_deconv3(F.relu_(f_feat + f_feat_res))

        d_ufeat3 = self.d_deconv3(d_feat)
        r_ufeat3 = self.r_deconv3(r_feat)
        d_gate3 = torch.sigmoid(self.d_gate3(d_ufeat3))
        d_feat = self.d_resblock50(torch.cat([d_feat2, d_ufeat3, d_gate3*r_feat2], 1))
        d_feat = self.d_resblock51(d_feat)
        r_gate3 = torch.sigmoid(self.r_gate3(r_ufeat3))
        r_feat = self.r_resblock50(torch.cat([r_feat2, r_ufeat3, r_gate3*d_feat2], 1))
        r_feat = self.r_resblock51(r_feat)

        f_feat = self.f_conv3_1(torch.cat([d_feat, r_feat, f_feat], 1))
        f_feat_res = F.interpolate(self.f_conv3_2(F.relu(f_feat)), scale_factor=2, mode='bilinear')
        f_feat = self.f_deconv2(F.relu_(f_feat + f_feat_res))

        d_ufeat2 = self.d_deconv2(d_feat)
        r_ufeat2 = self.r_deconv2(r_feat)
        d_gate2 = torch.sigmoid(self.d_gate2(d_ufeat2))
        d_feat = self.d_resblock60(torch.cat([d_feat1, d_ufeat2, d_gate2*r_feat1], 1))
        d_feat = self.d_resblock61(d_feat)
        r_gate2 = torch.sigmoid(self.r_gate2(r_ufeat2))
        r_feat = self.r_resblock60(torch.cat([r_feat1, r_ufeat2, r_gate2*d_feat1], 1))
        r_feat = self.r_resblock61(r_feat)

        f_feat = self.f_conv2_1(torch.cat([d_feat, r_feat, f_feat], 1))
        f_feat_res = F.interpolate(self.f_conv2_2(F.relu(f_feat)), scale_factor=2, mode='bilinear')
        f_feat = self.f_deconv1(F.relu_(f_feat + f_feat_res))

        d_ufeat1 = self.d_deconv1(d_feat)
        r_ufeat1 = self.r_deconv1(r_feat)
        d_gate1 = torch.sigmoid(self.d_gate1(d_ufeat1))
        d_feat = torch.cat((d_feat0, d_ufeat1, d_gate1*r_feat0), 1)
        r_gate1 = torch.sigmoid(self.r_gate1(r_ufeat1))
        r_feat = torch.cat((r_feat0, r_ufeat1, r_gate1*d_feat0), 1)

        d_feat = self.d_last_conv(d_feat)
        r_feat = self.r_last_conv(r_feat)

        f_feat = self.f_conv1_1(torch.cat([d_feat, r_feat, f_feat], 1))
        f_feat_res = F.interpolate(self.f_conv1_2(F.relu(f_feat)), scale_factor=2, mode='bilinear')
        f_feat = F.relu_(f_feat + f_feat_res)

        d_out = self.d_out(d_feat)
        r_out = self.r_out(r_feat)

        f_out = self.f_out(f_feat)
        out = [f_out, d_out, r_out]
    
        return out

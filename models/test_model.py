import torch
from .base_model import BaseModel
from . import networks

class TESTModel(BaseModel):
    def name(self):
        return 'TESTModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        
        self.visual_names = ['sparse', 'pred', 'img']
        
        self.model_names = ['DC']
  
        self.netDC = networks.DCOMPNet(channels=opt.channels, knn=opt.knn, nsamples=opt.nsamples, scale=opt.scale)
        self.netDC = networks.init_net(self.netDC, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids, need_init=False)
            
    def set_input(self, input):

        self.img = input['img'].to(self.device)
        self.sparse = input['sparse'].to(self.device)
        self.K = input['K'].to(self.device)

    def forward(self):
        if self.opt.clip and self.opt.dataset == 'kitti':
            c = 352-256
        else:
            c = 0
        sparse = self.sparse
        self.sparse = self.sparse[:, :, c:, :]
        self.img = self.img[:, :, c:, :]

        if self.opt.flip_input:
            # according to https://github.com/kakaxi314/GuideNet,
            # this operation might be helpful to reduce the error greatly.
            input_s = torch.cat([self.sparse, self.sparse.flip(3)], 0)
            input_i = torch.cat([self.img, self.img.flip(3)], 0)
            input_K = torch.cat([self.K, self.K], 0)
        else:
            input_s = self.sparse
            input_i = self.img
            input_K = self.K
        
        with torch.no_grad():
            out = self.netDC(input_s, input_i, input_K)
            self.pred = out[0]

        if self.opt.flip_input:
            b = self.pred.shape[0]
            self.pred = (self.pred[:b//2] + self.pred[b//2:].flip(3)) / 2

        if c != 0:
            pred = torch.zeros_like(sparse)
            pred[:, :, :c, :] = self.pred[0:1, :, 0:1, :]
            pred[:, :, c:, :] = self.pred[0:1, :, :, :]
            self.pred = pred

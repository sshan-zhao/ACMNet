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
        
        self.model_names = ['G']
  
        self.netG = networks.DCOMPNet(channels=opt.channels, knn=opt.knn, nsamples=opt.nsamples, scale=opt.scale)
        self.netG = networks.init_net(self.netG, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids, need_init=False)
            
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
        
        with torch.no_grad():
            out = self.netG(self.sparse, self.img, self.K)
            self.pred = out[0]
         
        if c != 0:
            pred = torch.zeros_like(sparse)
            pred[:, :, :c, :] = self.pred[0:1, :, 0:1, :]
            pred[:, :, c:, :] = self.pred[0:1, :, :, :]
            self.pred = pred
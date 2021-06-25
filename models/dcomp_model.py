import torch
import itertools
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F

class DCOMPModel(BaseModel):
    def name(self):
        return 'DCOMPModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        if is_train:
            parser.add_argument('--lambda_R', type=float, default=1.0, help='weight for recon loss')
            parser.add_argument('--lambda_S', type=float, default=0.01, help='weight for smoothness loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        if self.isTrain:
            self.loss_names = ['R', 'S', 'Rd', 'Rr']
           
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        if self.isTrain:
            self.visual_names = ['sparse', 'gt', 'pred', 'img']
        else:
            self.visual_names = ['sparse', 'pred', 'img'] 

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['DC']
           
        else:  
            self.model_names = ['DC']
  
        self.netDC = networks.DCOMPNet(channels=opt.channels, knn=opt.knn, nsamples=opt.nsamples)
        self.netDC = networks.init_net(self.netDC, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids)

        if self.isTrain:           
            # define loss functions
           
            self.criterionRecon = torch.nn.MSELoss()
            self.criterionSmooth = networks.SmoothLoss()
            
            self.optimizers = []
            self.optimizer = torch.optim.Adam(itertools.chain(self.netDC.parameters()),
                                                        lr=opt.lr, betas=(opt.beta1, 0.999))
           
            self.optimizers += [self.optimizer]
            
    def set_input(self, input):

        if self.isTrain:
            self.sparse = input['sparse'].to(self.device) 
            self.img = input['img'].to(self.device)
            self.gt = input['gt'].to(self.device) 
            self.K = input['K'].to(self.device)
        else:
            self.img = input['img'].to(self.device)
            self.sparse = input['sparse'].to(self.device)
            self.K = input['K'].to(self.device)

    def forward(self):
        if self.opt.clip:
            c = 352-256
        else:
            c = 0

        self.sparse = self.sparse[:, :, c:, :]
        self.img = self.img[:, :, c:, :]        
        self.gt = self.gt[:, :, c:, :]

        b = self.sparse.shape[0]
        n = len(self.opt.gpu_ids)
        if b < n:
            self.sparse = self.sparse.repeat(n,1,1,1)
            self.img = self.img.repeat(n,1,1,1)
            self.gt = self.gt.repeat(n,1,1,1)
            self.K = self.K.repeat(n,1,1)
            self.sparse = self.sparse.narrow(0,0,n)
            self.img = self.img.narrow(0,0,n)
            self.gt = self.gt.narrow(0,0,n)
            self.K = self.K.narrow(0,0,n)

        out = self.netDC(self.sparse, self.img, self.K)
        self.pred = out[0]
        self.pred_d = out[1]
        self.pred_r = out[2]
    
    def backward(self):

        lambda_R = self.opt.lambda_R
        lambda_S = self.opt.lambda_S

        mask = (self.gt > 0).cuda()
            
        self.loss = 0
       
        self.loss_R = self.criterionRecon(self.pred[mask], self.gt[mask]) * lambda_R
        self.loss += self.loss_R
        self.loss_S = self.criterionSmooth(self.pred, self.img) * lambda_S
        self.loss += self.loss_S

        self.loss_Rd = self.criterionRecon(self.pred_d[mask], self.gt[mask]) * lambda_R * 0.5
        self.loss += self.loss_Rd
        self.loss_Rr = self.criterionRecon(self.pred_r[mask], self.gt[mask]) * lambda_R * 0.5
        self.loss += self.loss_Rr

        self.loss.backward()

    def optimize_parameters(self):

        self.forward()
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        self.backward()
        for optimizer in self.optimizers:
            optimizer.step()

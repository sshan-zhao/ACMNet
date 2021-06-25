import argparse
import os
import util
import torch
import models
import data

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--root', type=str, default='datasets', help='path to dataset')
        parser.add_argument('--dataset', type=str, default='kitti', help='dataset name')
        parser.add_argument('--test_data_file', type=str, default='sval.list', help='validatation data list')
        parser.add_argument('--train_data_file', type=str, default='train.list', help='validatation data list')
        parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--model', type=str, default='sd',
                            help='chooses which model to use, sd|...')
        parser.add_argument('--clip', action='store_true', help='clip the above part of the image')
        parser.add_argument('--channels', type=int, default=32, help='channels')
        parser.add_argument('--scale', type=float, default=80, help='scale')
        parser.add_argument('--knn', nargs='+', type=int, default=6, help='number of nearest-neighbour')
        parser.add_argument('--nsamples', nargs='+', type=int, default=10000, help='sampling ratio')
        parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--no_augment', action='store_true', help='if specified, do not use data augmentation, e.g., randomly shifting gamma')
        parser.add_argument('--init_type', type=str, default='kaiming', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{which_model_netG}_size{loadSize}')
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with the new defaults

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        if self.isTrain:
            expr_dir = os.path.join(opt.checkpoints_dir, opt.expr_name)
            util.mkdirs(expr_dir)
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')

    def parse(self):

        opt = self.gather_options()
        opt.isTrain = self.isTrain

        opt.expr_name = opt.dataset + '_' + opt.model
        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.expr_name = opt.expr_name + suffix
        
        if opt.isTrain:
            self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt

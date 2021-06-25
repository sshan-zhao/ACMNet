from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--print_freq', type=int, default=32, help='frequency of showing training results on console')
        parser.add_argument('--save_result_freq', type=int, default=3200, help='frequency of saving the latest prediction results')
        parser.add_argument('--save_latest_freq', type=int, default=6400, help='frequency of saving the latest trained model')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=5e-4, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy: lambda|step|plateau')
        parser.add_argument('--lr_decay_iters', type=int, default=10, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--optimizer', type=str, default='adam', help='adam | sgd | adabound')

        self.isTrain = True
        return parser

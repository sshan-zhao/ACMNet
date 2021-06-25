from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--model_path', type=str, default='', help='the path of the model file')
        parser.add_argument('--which_epoch', type=str, default='', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--save', action='store_true', help='save results')
        parser.add_argument('--flip_input', action='store_true', default=False)
        
        self.isTrain = False
        return parser

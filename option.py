import argparse
from utils.utils import createDir

parser = argparse.ArgumentParser()

# train config


parser.add_argument("--epochs", type=int, default=500,
                    help="number of epochs of training")

parser.add_argument("--batch_size", type=int, default=1,
                    help="size of the batches")

parser.add_argument('--crop', action='store_true',
                    help='训练过程是否crop')

parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')

parser.add_argument("--decay_type", type=str, default='step',
                    help="learning rate decay type 学习率调整策略")
parser.add_argument('--lr_decay', type=int, default=50,
                    help='learning rate decay per N epochs')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')

parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")

parser.add_argument("--decay_epoch", type=int, default=100,
                    help="epoch from which to start lr decay")


parser.add_argument('--optimizer', default='ADAM', choices=('SGD',
                                                            'ADAM', 'RMSprop'), help='optimizer to use (SGD | ADAM | RMSprop)')

parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')

parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')

parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')

parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')

parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')


parser.add_argument('--cpu', action='store_true',
                    help='use cpu')
# data config

parser.add_argument("--edge_type", type=str,
                    default="canny", help="type of the edge dataset")

parser.add_argument("--w_size", type=int, default=512, help="clip size")
parser.add_argument("--h_size", type=int, default=512, help="clip size")


# loss config

parser.add_argument("--lambda_l1", type=float, default=10.0,
                    help="perceptual loss weight")

parser.add_argument("--lambda_edge", type=float, default=5.0,
                    help="perceptual loss weight")
# model config
parser.add_argument("--model", type=str, default="sym")

# log config

parser.add_argument("--sample_interval", type=int, default=30,
                    help="interval between saving generator outputs")

parser.add_argument("--checkpoint_interval", type=int, default=10,
                    help="interval between saving model checkpoints")


# test config
parser.add_argument("--random_nums", type=int, default=1000,
                    help="当数据集图片数较大时，随机选取1000张作为训练集")

parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
# hardware config
parser.add_argument("--n_cpu", type=int, default=8,
                    help="number of cpu threads to use during batch generation")

args = parser.parse_args()
args.dataDir = f'input'
args.test_model_dir = f'output/bestpsnr.pth'
args.logDir = f'output/'

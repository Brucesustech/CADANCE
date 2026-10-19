import argparse
import torch

parser = argparse.ArgumentParser()

# 1) dataset / task
parser.add_argument('--dataset', type=str, default='synthetic', help='datasets')
parser.add_argument('--P', type=float, default=0.4)
parser.add_argument('--experiment_name', type=str, default='lp', help='which task')
parser.add_argument('--num_nodes', type=int, default=-1, help='num of nodes')
parser.add_argument('--input_dim', type=int, default=128, help='dim of input feature')
parser.add_argument('--testlength', type=int, default=3, help='length for test')
parser.add_argument('--vallength', type=int, default=1, help='length for val')
parser.add_argument('--num_classes', type=int, default=-1, help='num of classes')
parser.add_argument('--nc_layers', type=int, default=2, help='classifier layers')

# 2) model
parser.add_argument('--hid_dim', type=int, default=64, help='hidden dim')
parser.add_argument('--weight1', type=float, default=1.0, help='overall OOD regularization weight')
parser.add_argument('--dropout', type=float, default=0., help='general dropout rate')
parser.add_argument('--max_time_len', type=int, default=128, help='max timestamp length')

# flags
parser.add_argument('--residual', type=int, default=1)
parser.add_argument('--norm', type=int, default=1)
parser.add_argument('--fmask', type=int, default=1)
parser.add_argument('--lin_bias', type=int, default=0)
parser.add_argument('--skip', type=int, default=0)

# 3) optimization / training
parser.add_argument('--max_epoch', type=int, default=1000, help='max training epochs')
parser.add_argument('--min_epoch', type=int, default=100, help='min epoch for early-stop')
parser.add_argument('--warm_epoch', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='L2 weight decay')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--patience', type=int, default=10, help='early stop patience')
parser.add_argument('--device_id', type=str, default='0', help='gpu id, <0 for cpu')
parser.add_argument('--intervention_times', type=int, default=10, help='intervention times')
parser.add_argument('--grad_clip', type=float, default=5.0)

# 4) log / misc
parser.add_argument('--log_dir', type=str, default='logs/new/')
parser.add_argument('--log_interval', type=int, default=20)
parser.add_argument('--sampling_times', type=int, default=1, help='negative sampling times')
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--shift', type=int, default=0)
parser.add_argument('--use_cfg', type=int, default=1, help='use dataset preset config')

args = parser.parse_args()

# device
if int(args.device_id) >= 0 and torch.cuda.is_available():
    args.device = torch.device(f'cuda:{args.device_id}')
    print(f'using gpu:{args.device_id} to train the model')
else:
    args.device = torch.device('cpu')
    print('using cpu to train the model')

# dataset presets
if args.use_cfg:
    if args.dataset == 'collab':
        args.experiment_name = 'lp'
        args.hid_dim = 32
        args.weight1 = 0.6
        args.lr = 0.005
        args.weight_decay = 5e-3

    elif args.dataset == 'act':
        args.experiment_name = 'lp'
        args.hid_dim = 64
        args.weight1 = 1.0
        args.lr = 0.003
        args.weight_decay = 5e-4


    elif args.dataset == 'Aminer':
        args.experiment_name = 'nc'
        args.hid_dim = 64
        args.lr = 0.002 
        args.weight_decay = 5e-3
        args.weight1 = 1.0 
        args.min_epoch = 200
        args.patience = 50

    elif args.dataset == 'dymotif_data':
        args.experiment_name = 'nc'
        args.hid_dim = 128
        args.lr = 0.005
        args.weight_decay = 1e-5
        args.weight1 = 0.7
        args.min_epoch = 400
        args.patience = 50
        args.dropout = 0.1
        args.lin_bias = 1

    elif 'synthetic' in args.dataset:
        args.experiment_name = 'lp'

        if args.P == 0.4:
            args.hid_dim = 64
            args.weight1 = 1.0
            args.lr = 0.003
            args.weight_decay = 5e-3
        elif args.P == 0.6:
            args.hid_dim = 32
            args.weight1 = 0.5
            args.lr = 0.003
            args.weight_decay = 2e-2
            args.dropout = 0.1
            args.patience = 1
        elif args.P == 0.8:
            args.hid_dim = 128
            args.weight1 = 0.7
            args.lr = 0.001
            args.weight_decay = 2e-2
            args.dropout = 0.2
        else:
            raise ValueError(f'Unsupported P for synthetic: {args.P}')
    else:
        raise NotImplementedError(f'dataset {args.dataset} not implemented')
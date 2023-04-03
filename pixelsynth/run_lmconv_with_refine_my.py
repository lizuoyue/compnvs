from IPython import embed
import argparse
import itertools
from operator import itemgetter
import os
import re
import time
import tqdm
import pickle as pkl

from PIL import Image
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils

#from baseline import PixelCNN
from models.lmconv.layers import PONO
from models.lmconv.masking import *
from models.lmconv.model import OurPixelCNN
from models.lmconv.utils import *
from models.vqvae2.vqvae import VQVAETop
from models.networks.architectures import ResNetDecoder
from models.losses.synthesis import SynthesisLoss
from models.losses.gan_loss import DiscriminatorLoss

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,4"
N_CLASS=512
TEMPERATURE=1

# Generate outpaint and refinement results on validation set
# CUDA_VISIBLE_DEVICES=3 python run_lmconv_with_refine_my.py --run_dir models/lmconv_with_refine/runs/replica_mid -d replica_mid -b 24 -t 10 -c 4e6 -k 3 --normalization pono --order custom -dp 0 --test_interval 1 --sample_interval 5 --nr_resnet 2 --nr_filters 80 --sample_region custom --sample_batch_size 8 --max_epochs 150 --vqvae_path models/vqvae2/checkpoint/replica/vqvae_150.pt --predict_residual --dset_dir models/vqvae2/scene_embeddings --load_params_from models/lmconv_with_refine/runs/replica/0_ep27.pth

parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str,
                    default='data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='models',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--dataset', type=str, choices=['replica', 'replica_mid', 'arkit', 'fountain', 'westminster', 'pantheon', 'sacre', 'notre'])
parser.add_argument('--binarize', action='store_true')
parser.add_argument('-p', '--print_every', type=int, default=20,
                    help='how many iterations between print statements')
parser.add_argument('-t', '--save_interval', type=int, default=20,
                    help='Every how many epochs to write checkpoint?')
parser.add_argument('-ts', '--sample_interval', type=int, default=4,
                    help='Every how many epochs to write samples?')
parser.add_argument('-tt', '--test_interval', type=int, default=1,
                    help='Every how many epochs to test model?')
parser.add_argument('-r', '--load_params', type=str, default=None,
                    help='Restore training from previous model checkpoint?')
parser.add_argument('--load_last_params', action="store_true",
                    help='Restore training from the last model checkpoint in the run dir?')
parser.add_argument('-rd', '--run_dir', type=str, default=None,
                    help="Optionally specify run directory. One will be generated otherwise."
                         "Use to save log files in a particular place")
parser.add_argument('--exp_name', type=str, default=None)
parser.add_argument('-ID', '--exp_id', type=int, default=0)
parser.add_argument('--ours', action='store_true')
parser.add_argument('--dset_dir', type=str, default='models/vqvae2/scene_embeddings',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('--vqvae_path', type=str, default='',
                    help='Location for vqvae checkpoint - used for sampling')
parser.add_argument('--gen_order_dir', type=str, default='data',
                    help='Location for parameter checkpoints and samples')
# pixelcnn++ and our model
parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                    help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160,
                    help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                    help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-l', '--lr', type=float,
                    default=0.0002, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-wd', '--weight_decay', type=float,
                    default=0, help='Weight decay during optimization')
parser.add_argument('-c', '--clip', type=float, default=-1, help='Gradient norms clipped to this value')
parser.add_argument('-b', '--batch_size', type=int, default=64,
                    help='Batch size during training per GPU')
parser.add_argument('-x', '--max_epochs', type=int,
                    default=5000, help='How many epochs to run in total?')
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed to use')
parser.add_argument('--ema', type=float, default=1)
# our model
parser.add_argument('-k', '--kernel_size', type=int, default=5,
                    help='Size of conv kernels')
parser.add_argument('-md', '--max_dilation', type=int, default=2,
                    help='Dilation in downsize stream')
parser.add_argument('-dp', '--dropout_prob', type=float, default=0.5,
                    help='Dropout prob used with nn.Dropout2d in gated resnet layers. '
                         'Argument only used if --ours is provided. Set to 0 to disable '
                         'dropout entirely.')
parser.add_argument('-nm', '--normalization', type=str, default='weight_norm',
                    choices=["none", "weight_norm", "order_rescale", "pono"])
parser.add_argument('-af', '--accum_freq', type=int, default=1,
                    help='Batches per optimization step. Used for gradient accumulation')
parser.add_argument('--two_stream', action="store_true", help="Enable two stream model")
parser.add_argument('--order', type=str, nargs="+",
                    choices=["raster_scan", "s_curve", "hilbert", "gilbert2d", "s_curve_center_quarter_last", 'custom'],
                    help="Autoregressive generation order")
parser.add_argument('--randomize_order', action="store_true", help="Randomize between 8 variants of the "
                    "pixel generation order.")
parser.add_argument('--mode', type=str, choices=["train", "sample", "test", "count_params"],
                    default="test")
parser.add_argument('--load_params_from', type=str, default=None)
# refinement model
parser.add_argument("--refine_model_type", type=str, default="resnet_256W8UpDown3")
parser.add_argument("--predict_residual", action="store_true", default=False)
parser.add_argument("--norm_G", type=str, default="sync:spectral_batch")
parser.add_argument("--ngf", type=int, default=64)
parser.add_argument("--normalize_before_residual", action="store_true", default=False)
parser.add_argument("--losses", type=str, nargs="+", default=['1.0_l1','10.0_content'])
parser.add_argument("--discriminator_losses", type=str, default="pix2pixHD")
parser.add_argument("--lr_d", type=float, default=1e-3 * 2)
parser.add_argument("--lr_g", type=float, default=1e-3 / 2)
parser.add_argument("--beta1", type=float, default=0)
parser.add_argument("--beta2", type=float, default=0.9)
parser.add_argument("--ndf", type=int, default=64)
parser.add_argument("--output_nc", type=int, default=3)
parser.add_argument("--norm_D", type=str, default="spectralinstance")
parser.add_argument("--gan_mode", type=str, default="hinge", help="(ls|original|hinge)")
parser.add_argument("--no_ganFeat_loss", action="store_true")
parser.add_argument("--lambda_feat", type=float, default=10.0)

# depth estimation model, currently useless
parser.add_argument("--Unet_num_filters", type=int, default=32)
parser.add_argument("--min_z", type=float, default=0.5)
parser.add_argument("--max_z", type=float, default=10.0)
# configure training
parser.add_argument('--train_masks', nargs="*", type=int, help="Specify indices of masks in all_masks to use during training")
# configure sampling
parser.add_argument('--sample_region', type=str, choices=["full", "center", "random_near_center", "top", "custom"], default="full")
parser.add_argument('--sample_size_h', type=int, default=16, help="Only used for --sample_region center, top or random. =H of inpainting region.")
parser.add_argument('--sample_size_w', type=int, default=16, help="Only used for --sample_region center, top or random. =W of inpainting region.")
parser.add_argument('--sample_offset1', type=int, default=None, help="Manually specify box offset for --sample_region custom")
parser.add_argument('--sample_offset2', type=int, default=None, help="Manually specify box offset for --sample_region custom")
parser.add_argument('--sample_batch_size', type=int, default=25, help="Number of images to sample")
parser.add_argument('--sample_mixture_temperature', type=float, default=1.0)
parser.add_argument('--sample_logistic_temperature', type=float, default=1.0)
parser.add_argument('--sample_quantize', action="store_true", help="Quantize images during sampling to avoid train-sample distribution shift")
parser.add_argument('--save_nrow', type=int, default=4)
parser.add_argument('--save_padding', type=int, default=2)
# configure testing
parser.add_argument('--test_masks', nargs="*", type=int, help="Specify indices of masks in all_masks to use during testing")
parser.add_argument('--test_region', type=str, choices=["full", "custom"], default="full")
parser.add_argument('--test_minh', type=int, default=0, help="Specify conditional likelihood testing region. Only used with --test_region custom")
parser.add_argument('--test_maxh', type=int, default=32, help="Specify conditional likelihood testing region. Only used with --test_region custom")
parser.add_argument('--test_minw', type=int, default=0, help="Specify conditional likelihood testing region. Only used with --test_region custom")
parser.add_argument('--test_maxw', type=int, default=32, help="Specify conditional likelihood testing region. Only used with --test_region custom")
parser.add_argument('--order_variants', nargs="*", type=int)
# our model
parser.add_argument('--no_bias', action="store_true", help="Disable learnable bias for all convolutions")
parser.add_argument('--learn_weight_for_masks', action="store_true", help="Condition each masked conv on the mask itself, with a learned weight")
parser.add_argument('--minimize_bpd', action="store_true", help="Minimize bpd, scaling loss down by number of dimension")
parser.add_argument('--resize_sizes', type=int, nargs="*")
parser.add_argument('--resize_probs', type=float, nargs="*")
parser.add_argument('--base_order_reflect_rows', action="store_true")
parser.add_argument('--base_order_reflect_cols', action="store_true")
parser.add_argument('--base_order_transpose', action="store_true")
# memory and precision
parser.add_argument('--rematerialize', action="store_true", help="Recompute some activations during backwards to save memory")
# plotting
parser.add_argument('--plot_masks', action="store_true")

args = parser.parse_args()

# CUDA_VISIBLE_DEVICES=0 python run_lmconv_with_refine_my.py --run_dir models/lmconv_with_refine/runs/arkit -d arkit -b 24 -t 10 -c 4e6 -k 3 --normalization pono --order custom -dp 0 --test_interval 1 --sample_interval 5 --nr_resnet 2 --nr_filters 80 --sample_region custom --sample_batch_size 8 --max_epochs 100 --vqvae_path models/vqvae2/checkpoint/arkit/vqvae_100.pt --predict_residual --dset_dir models/vqvae2/scene_embeddings --load_params_from models/lmconv_with_refine/runs/arkit/0_ep32.pth


# Set seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Create run directory
if args.run_dir:
    run_dir = args.run_dir
    try: 
        os.makedirs(run_dir)
    except:
        pass
else:
    dataset_name = args.dataset if not args.binarize else f"binary_{args.dataset}"
    _name = "{:05d}_{}_lr{:.5f}_bs{}_gc{}_k{}_md{}".format(
        args.exp_id, dataset_name, args.lr, args.batch_size, args.clip, args.kernel_size, args.max_dilation)
    if args.normalization != "none":
        _name = f"{_name}_{args.normalization}"
    if args.exp_name:
        _name = f"{_name}+{args.exp_name}"
    run_dir = os.path.join("runs", _name)
    if args.mode == "train":
        os.makedirs(run_dir, exist_ok=args.load_last_params)

assert os.path.exists(run_dir), "Did not find run directory, check --run_dir argument"

# Log arguments
timestamp = time.strftime("%Y%m%d-%H%M%S")
if args.mode == "test" and args.test_region == "custom":
    logfile = f"{args.mode}_{args.test_minh}:{args.test_maxh}_{args.test_minw}:{args.test_maxw}_{timestamp}.log"
else:
    logfile = f"{args.mode}_{timestamp}.log"
logger = configure_logger(os.path.join(run_dir, logfile))
logger.info("Run directory: %s", run_dir)
logger.info("Arguments: %s", args)
for k, v in vars(args).items():
    logger.info(f"  {k}: {v}")


# Create data loaders
sample_batch_size = args.sample_batch_size
dataset_obs = {
    'replica': (1,32,32),
    'replica_mid': (1,32,32),
    'arkit': (1,24,32),
    'fountain': (1,48,64),
    'westminster': (1,48,64),
    'sacre': (1,48,64),
    'pantheon': (1,48,64),
    'notre': (1,48,64),
    # 'fountain': (1,96,128),
}[args.dataset]
input_channels = dataset_obs[0]
data_loader_kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':False, 'batch_size':args.batch_size}
data_sampler_kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':False, 'batch_size':args.sample_batch_size}
if args.resize_sizes:
    if not args.resize_probs:
        args.resize_probs = [1. / len(args.resize_sizes)] * len(args.resize_sizes)
    assert len(args.resize_probs) == len(args.resize_sizes)
    assert sum(args.resize_probs) == 1
    resized_obses = [(input_channels, s, s) for s in args.resize_sizes]
else:
    args.resize_sizes = [dataset_obs[1]]
    args.resize_probs = [1.]
    resized_obses = [dataset_obs]

def obs2str(obs):
    return 'x'.join(map(str, obs))

def random_resized_obs():
    idx = np.arange(len(resized_obses))
    obs_i = np.random.choice(idx, p=args.resize_probs)
    return resized_obses[int(obs_i)]

def get_resize_collate_fn(obs, default_collate=torch.utils.data.dataloader.default_collate):
    if obs == dataset_obs:
        return default_collate

    def resize_collate_fn(batch):
        X, y = default_collate(batch)
        X = torch.nn.functional.interpolate(X, size=obs[1:], mode="bilinear")
        return [X, y]
    return resize_collate_fn

def random_resize_collate(batch, default_collate=torch.utils.data.dataloader.default_collate):
    triplet = default_collate(batch)
    x, y, z = triplet
    obs = random_resized_obs()
    if obs != dataset_obs:
        x = torch.nn.functional.interpolate(x, size=obs[1:], mode="bilinear")
    return (x, y, z)

if 'replica' in args.dataset:
    rescaling = lambda x : (x - .5) * 2.  # rescale [0, 1] images into [-1, 1] range
    rescaling_inv = lambda x : .5 * x + .5
    ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])

    class ReplicaDispDataset(torch.utils.data.Dataset):
        def __init__(self, scene_embeddings, orders_dict, phase, skip_sample=1):
            super(ReplicaDispDataset, self).__init__()
            # scene_embeddings: [np.zeros([300, 32, 32]) for sid in scene_ids]
            # orders_dict: {view_name: [1024, 2]} (view_name = {sid:02d}_{empty_view_id:03d}_{i_view_id:03d}_{j_view_id:03d})
            if phase == 'val':
                self.scene_ids = [13, 14, 19, 20, 21, 42]
            elif phase == 'train':
                self.scene_ids = [i for i in range(48) if i not in [13, 14, 19, 20, 21, 42]]
            self.src_rgb_dir = f'gen_data/{phase}/rgb'

            self.scene_embeddings = scene_embeddings['embeddings']
            self.transform = transforms.Compose([
                                            transforms.Resize((256, 256)),
                                            transforms.CenterCrop((256, 256)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                          ])
            self.orders_dict = orders_dict
            self.view_names = sorted(list(self.orders_dict.keys()))
            if skip_sample > 1:
                self.view_names = self.view_names[::skip_sample]

        def __len__(self):
            return len(self.view_names)

        def __getitem__(self, i):
            view_name = self.view_names[i]
            order = self.orders_dict[view_name]
            sid = int(view_name[:2])
            empty_view_id = int(view_name[3:6])

            gt_img = Image.open(os.path.join(self.src_rgb_dir, view_name[:6]+'.png')).convert('RGB')
            gt_img = self.transform(gt_img)
            full_embedding = torch.from_numpy(self.scene_embeddings[self.scene_ids.index(sid)][empty_view_id])
            return (view_name, full_embedding.unsqueeze(0), order, gt_img)

    # data_train_ = np.load(os.path.join(args.dset_dir,'replica_train.npz'))
    # with open(os.path.join(args.gen_order_dir,'replica_train_gen_order_pytorch3d.pkl'), 'rb') as f:
    #     gen_order_train = pkl.load(f)
    data_val_ = np.load(os.path.join(args.dset_dir,'replica_val.npz'))
    with open(os.path.join(args.gen_order_dir,f'{args.dataset}_val_gen_order_pytorch3d.pkl'), 'rb') as f:
        gen_order_val = pkl.load(f)
    
    # data_train = ReplicaDispDataset(data_train_, gen_order_train, 'train')
    if args.dataset == 'replica_mid':
        data_val = ReplicaDispDataset(data_val_, gen_order_val, 'val', skip_sample=3)
    else:
        data_val = ReplicaDispDataset(data_val_, gen_order_val, 'val')

    # train_loader = torch.utils.data.DataLoader(data_train, shuffle=True, \
    #     collate_fn=random_resize_collate, **data_loader_kwargs)
    test_loader_by_obs = {
        obs: torch.utils.data.DataLoader(data_val, collate_fn=get_resize_collate_fn(obs), \
            **data_loader_kwargs)
        for obs in resized_obses
    } 
    sample_loader_by_obs = {
        obs: torch.utils.data.DataLoader(data_val, collate_fn=get_resize_collate_fn(obs), \
            **data_sampler_kwargs)
        for obs in resized_obses
    }
    
    # Default upper bounds for progress bars
    # train_total = len(data_train_) // args.batch_size
    test_total = len(data_val_) // args.batch_size
elif 'arkit' in args.dataset:
    rescaling = lambda x : (x - .5) * 2.  # rescale [0, 1] images into [-1, 1] range
    rescaling_inv = lambda x : .5 * x + .5
    ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])

    class ARKitDispDataset(torch.utils.data.Dataset):
        def __init__(self, scene_embeddings, orders_dict, phase):
            super(ARKitDispDataset, self).__init__()
            self.data_dir = '/home/lzq/lzy/ARKitScenes/Selected'
            self.src_rgb_dir = f'{self.data_dir}/rgb'  # {sid}_{vid}.png
            self.phase = phase
            self.split_file = f'{self.data_dir}/{phase}_split.txt'
            with open(self.split_file, 'r') as txt_file:
                self.scene_names = [line.strip() for line in txt_file.readlines()]

            self.scene_embeddings = scene_embeddings['embeddings'].item()
            self.transform = transforms.Compose([
                                            transforms.Resize((192, 256)),
                                            # transforms.CenterCrop((256, 256)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                          ])
            self.orders_dict = orders_dict
            self.view_names = sorted(list(self.orders_dict.keys()))

        def __len__(self):
            return len(self.view_names)

        def __getitem__(self, i):
            view_name = self.view_names[i]
            order = self.orders_dict[view_name]
            scene_name, k_name, i_name, j_name = view_name.split('_')

            gt_img = Image.open(os.path.join(self.src_rgb_dir, f'{scene_name}_{k_name}.png')).convert('RGB')
            gt_img = self.transform(gt_img)
            full_embedding = torch.from_numpy(self.scene_embeddings[scene_name][k_name])
            return view_name, full_embedding.unsqueeze(0), order, gt_img

    # data_train_ = np.load(os.path.join(args.dset_dir,'arkit_train.npz'), allow_pickle=True)
    # with open(os.path.join(args.gen_order_dir,'arkit_train_gen_order_pytorch3d.pkl'), 'rb') as f:
    #     gen_order_train = pkl.load(f)
    data_val_ = np.load(os.path.join(args.dset_dir,'arkit_val.npz'), allow_pickle=True)
    with open(os.path.join(args.gen_order_dir,'arkit_val_gen_order_pytorch3d.pkl'), 'rb') as f:
        gen_order_val = pkl.load(f)
    
    # data_train = ARKitDispDataset(data_train_, gen_order_train, 'train')
    data_val = ARKitDispDataset(data_val_, gen_order_val, 'val')

    # train_loader = torch.utils.data.DataLoader(data_train, shuffle=True, \
    #     collate_fn=random_resize_collate, **data_loader_kwargs)
    test_loader_by_obs = {
        obs: torch.utils.data.DataLoader(data_val, collate_fn=get_resize_collate_fn(obs), \
            **data_loader_kwargs)
        for obs in resized_obses
    } 
    sample_loader_by_obs = {
        obs: torch.utils.data.DataLoader(data_val, collate_fn=get_resize_collate_fn(obs), \
            **data_sampler_kwargs)
        for obs in resized_obses
    }
    
    # Default upper bounds for progress bars
    # train_total = len(data_train_) // args.batch_size
    test_total = len(data_val_) // args.batch_size
elif args.dataset in ['fountain', 'westminster', 'notre', 'sacre', 'pantheon']:
    rescaling = lambda x : (x - .5) * 2.  # rescale [0, 1] images into [-1, 1] range
    rescaling_inv = lambda x : .5 * x + .5
    ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])

    class FountainDispDataset(torch.utils.data.Dataset):
        def __init__(self, scene_embeddings, orders_dict, phase):
            super(FountainDispDataset, self).__init__()
            import glob
            self.data_dir = glob.glob(f'/home/lzq/lzy/bangbang/*{args.dataset}*')[0]
            self.data_dir += f'/dense/distill_{phase}'
            self.phase = phase
            # self.split_file = f'{self.data_dir}/{phase}_split.txt'
            # with open(self.split_file, 'r') as txt_file:
            #     self.scene_names = [line.strip() for line in txt_file.readlines()]

            self.scene_embeddings = scene_embeddings['embeddings'].item()
            self.transform = transforms.Compose([
                transforms.Resize((384, 512)),#.Resize((768, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ])
            self.orders_dict = orders_dict
            self.view_names = sorted(list(self.scene_embeddings.keys()))

        def __len__(self):
            return len(self.scene_embeddings)
            # return min(16, len(self.scene_embeddings))

        def __getitem__(self, i):
            view_name = self.view_names[i]
            order = self.orders_dict[str(i)]#np.random.randint(len(self.orders_dict)))]
            gt_img = Image.open(os.path.join(self.data_dir, f'{view_name}')).convert('RGB')
            gt_img = self.transform(gt_img)
            full_embedding = torch.from_numpy(self.scene_embeddings[view_name])
            triplet = (view_name, full_embedding.unsqueeze(0), order, gt_img)
            return triplet

    # data_train_ = np.load(os.path.join(args.dset_dir,f'{args.dataset}_train.npz'), allow_pickle=True)
    # with open(os.path.join(args.gen_order_dir,f'{args.dataset}_train_gen_order_pytorch3d.pkl'), 'rb') as f:
    #     gen_order_train = pkl.load(f)
    data_val_ = np.load(os.path.join(args.dset_dir,f'{args.dataset}_val.npz'), allow_pickle=True)
    with open(os.path.join(args.gen_order_dir,f'{args.dataset}_val_gen_order_pytorch3d.pkl'), 'rb') as f:
        gen_order_val = pkl.load(f)
    data_test_ = np.load(os.path.join(args.dset_dir,f'{args.dataset}_test.npz'), allow_pickle=True)
    with open(os.path.join(args.gen_order_dir,f'{args.dataset}_test_gen_order_pytorch3d.pkl'), 'rb') as f:
        gen_order_test = pkl.load(f)
    
    # data_train = FountainDispDataset(data_train_, gen_order_train, 'train')
    data_val = FountainDispDataset(data_val_, gen_order_val, 'val')
    data_test = FountainDispDataset(data_test_, gen_order_test, 'test')

    # train_loader = torch.utils.data.DataLoader(data_train, shuffle=True, \
    #     collate_fn=random_resize_collate, **data_loader_kwargs)
    test_loader_by_obs = {
        obs: torch.utils.data.DataLoader(data_test, collate_fn=get_resize_collate_fn(obs), \
            **data_loader_kwargs)
        for obs in resized_obses
    } 
    sample_loader_by_obs = {
        obs: torch.utils.data.DataLoader(data_test, collate_fn=get_resize_collate_fn(obs), \
            **data_sampler_kwargs)
        for obs in resized_obses
    }
    
    # Default upper bounds for progress bars
    # train_total = np.ceil(len(data_train_) / args.batch_size).astype(np.int32)
    test_total = np.ceil(len(data_test_) / args.batch_size).astype(np.int32)




def quantize(x):
    # Quantize [-1, 1] images to uint8 range, then put back in [-1, 1]
    # Can be used during sampling with --sample_quantize argument

    continuous_x = rescaling_inv(x) * 255  # Scale to [0, 255] range
    discrete_x = continuous_x.long().float()  # Round down
    quantized_x = discrete_x / 255.
    return rescaling(quantized_x)


if args.normalization == "order_rescale":
    norm_op = lambda num_channels: OrderRescale()
elif args.normalization == "pono":
    norm_op = lambda num_channels: PONO()
else:
    norm_op = None

assert not args.two_stream, "--two_stream cannot be used with --ours"
outpaint_model = OurPixelCNN(
            nr_resnet=args.nr_resnet,
            nr_filters=args.nr_filters, 
            input_channels=N_CLASS,
            nr_logistic_mix=args.nr_logistic_mix,
            kernel_size=(args.kernel_size, args.kernel_size),
            max_dilation=args.max_dilation,
            weight_norm=(args.normalization == "weight_norm"),
            feature_norm_op=norm_op,
            dropout_prob=args.dropout_prob,
            conv_bias=(not args.no_bias),
            conv_mask_weight=args.learn_weight_for_masks,
            rematerialize=args.rematerialize,
            binarize=args.binarize)

outpaint_model = outpaint_model.cuda()

channels_in = 3
refine_model = ResNetDecoder(args, channels_in=channels_in, channels_out=3)
refine_model = refine_model.cuda()

checkpoint_epochs = -1
checkpoint_step = -1

if args.load_params_from:
    if os.path.exists(args.load_params_from):
        ckpt = torch.load(args.load_params_from)
        outpaint_model.load_state_dict(ckpt['outpaint_model_state_dict'])
        refine_model.load_state_dict(ckpt['refine_model_state_dict'])
    else:
        raise Exception(f'Given dir of params: {args.load_params_from} does not exist.')

# Initialize exponential moving average of parameters
if args.ema < 1:
    ema = EMA(args.ema)
    ema.register(model)

# load vqvae, used for sampling
vqvae_model = VQVAETop()
ckpt = torch.load(args.vqvae_path)
ckpt = {k.replace('module.', ''):v for k, v in ckpt.items()}
# torch_devices = [0]
# device = "cuda:" + str(torch_devices[0])
# vqvae_model = nn.DataParallel(vqvae_model, torch_devices).to(device)
vqvae_model.load_state_dict(ckpt)
vqvae_model = vqvae_model.to('cuda')
# vqvae_model.eval()

obs = dataset_obs#(1, 24, 32) if args.dataset == 'arkit' else (1, 32, 32)

def get_sampling_images(loader):
    # Get batch of images to complete for inpainting, or None for --sample_region=full
    if args.sample_region == "full":
        return None
    logger.info('getting batch of images to complete...')
    # Get sample_batch_size images from test set
    batches_to_complete = []
    sample_iter = iter(loader)
    tmp = next(sample_iter)

    batch_to_complete = [tmp[0],tmp[1]]
    return batch_to_complete

def sample(model, generation_idx, mask_init, mask_undilated, mask_dilated, batch_to_complete, obs):
    batch_to_complete_full = torch.clone(batch_to_complete)
    batch_to_complete = (
        F.one_hot(batch_to_complete[:,0,:,:].to(torch.int64), N_CLASS).permute(0, 3, 1, 2).to(torch.float32)
    )
    model.eval()

    if args.sample_region == "full":
        data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
        data = data.cuda()
        sample_idx = generation_idx
        context = None
        batch_to_complete = None
    elif args.sample_region in ['custom']:
        # here, we have background mask
        data = batch_to_complete.clone().cuda()

        # Get indices of sampling region, need to do this for each image in batch
        sample_indices = []
        for image_number in range(batch_to_complete.shape[0]):
            sample_region = set()

            # Sort according to generation_idx
            sample_idx_this = []
            num_added = 0

            if generation_idx.shape[-1] == 3:
                # generation_idx[image_number] = generation_idx[image_number][generation_idx[image_number][:,0].argsort(descending=True)]

                # generation_idx[image_number] = generation_idx[image_number][generation_idx[image_number][:,0].argsort(descending=True)]
                sel_neg = generation_idx[image_number][:,0] < 0
                sel_pos = ~sel_neg

                generation_idx[image_number] = torch.cat([generation_idx[image_number][sel_pos], generation_idx[image_number][sel_neg]], dim=0)

            for idx, info in enumerate(generation_idx[image_number]):
                if len(info) == 3:
                    d, i, j = info
                else:
                    i, j = info
                if idx > int(obs[1]*obs[2]*0.5):
                # if d < 0:
                    sample_idx_this.append([i, j])
                    num_added += 1 
            sample_idx_this = np.array(sample_idx_this, dtype=np.int)

            sample_indices.append(sample_idx_this)

            # tosee = np.zeros((32, 32), np.float32)
            # d = 1000
            # for r, c in sample_idx_this:
            #     tosee[r, c] = d
            #     d -= 1
            # import matplotlib; matplotlib.use('agg')
            # import matplotlib.pyplot as plt
            # plt.imshow(tosee)
            # plt.colorbar()
            # plt.savefig('ccccc.png')
            # input('aaa')

            data[image_number, :, sample_idx_this[:, 0], sample_idx_this[:, 1]] = 0
            context = rescaling_inv(data).cpu()

    # aaaaa = np.zeros((16,32,32),np.uint8)
    for n_pix, (_, _) in enumerate(tqdm.tqdm(sample_indices[0], desc="Sampling pixels")):
        data_v = Variable(data)
        new_input = [data_v, mask_init, mask_undilated, mask_dilated]
        out = model(new_input, sample=True)
        
        for image_number in range(out.shape[0]):
            (i, j) = sample_indices[image_number][n_pix]
            # aaaaa[image_number, i, j] = 255
            prob = torch.softmax(out[:, :, i, j] / TEMPERATURE, 1)
            new_samples = torch.multinomial(prob, 1).squeeze(-1)
            data[image_number, :, i, j] = (
                F.one_hot(new_samples[image_number].to(torch.int64), N_CLASS).to(torch.float32)
            )
    # from PIL import Image
    # for caonima in range(16):
    #     Image.fromarray(aaaaa[caonima]).save(f'aaaaaaaaaa{caonima:02d}.png')
    # input('press')

    # print(loss_fn_op(data, batch_to_complete_full[:,0].to(torch.int64).cuda()))

    if batch_to_complete is not None and context is not None:
        # Interleave along batch dimension to visualize GT images
        difference = torch.abs(data.cpu() - batch_to_complete.cpu())
        data = torch.stack([context, data.cpu(), batch_to_complete.cpu(), difference], dim=1).view(-1, *data.shape[1:])

    return data

if __name__ == "__main__":
    logger.info("starting training")
    global_step = checkpoint_step + 1
    min_train_bpd = 1e12
    min_test_bpd_by_obs = {obs: 1e12 for obs in resized_obses}
    last_saved_epoch = -1

    os.makedirs(f'res_outpaint_{args.dataset}', exist_ok=True)
    os.makedirs(f'res_refined_{args.dataset}', exist_ok=True)
    
    time_ = time.time()
    possible_masks = []                
    
    outpaint_model.eval()
    refine_model.eval()
    with torch.no_grad():
        save_dict = {}
        for obs in resized_obses:
            pbar = tqdm.tqdm(sample_loader_by_obs[obs])
            for batch_idx, full_input in enumerate(pbar):
                view_name, full_embedding, order, gt_img = full_input
                # batch_to_complete = get_sampling_images(sample_loader_by_obs[obs])
                masks_init = []
                masks_undilated = []
                masks_dilated = []
                num_images = full_embedding.shape[0]

                kkk, sss = 3, 2

                for image_idx in range(num_images):
                    mask_init, mask_undilated, mask_dilated = get_masks(np.array(order[image_idx]), obs[1], obs[2], kkk, sss, plot=False)
                    masks_init.append(mask_init[0:1])
                    masks_undilated.append(mask_undilated[0:1])
                    masks_dilated.append(mask_dilated[0:1])

                masks_init = torch.stack(masks_init).repeat(1, 513, 1, 1).view(-1,kkk*kkk,obs[1]*obs[2]).cuda(non_blocking=True)
                masks_undilated = torch.stack(masks_undilated).repeat(1, 160, 1, 1).view(-1,kkk*kkk,obs[1]*obs[2]).cuda(non_blocking=True)
                masks_dilated = torch.stack(masks_dilated).repeat(1, 80, 1, 1).view(-1,kkk*kkk,obs[1]*obs[2]).cuda(non_blocking=True)
                all_masks = [masks_init, masks_undilated, masks_dilated]
                sample_t = sample(outpaint_model, order, *all_masks, full_embedding, obs)
                # sample_save_path = os.path.join(run_dir, f"tsample_obs{obs2str(obs)}_{epoch}.png")
                # decode sample
                
                sample_t = torch.argmax(sample_t, dim=1)
                sample_t_out = vqvae_model.decode_code(sample_t.to(torch.int64).to('cuda'))

                new_dim = sample_t_out.shape[0]//4
                sample_t_out_split = sample_t_out.reshape(new_dim, 4, *sample_t_out.shape[1:])  # bs x 4 x 3x256x256
                sample_t_out_gen = sample_t_out_split[:,1].contiguous() # bs x 3x256x256
                sample_t_refined = refine_model(sample_t_out_gen, None)

                for image_idx in range(num_images):
                    outpaint_img = sample_t_out_gen[image_idx] * 0.5 + 0.5
                    refined_img = sample_t_refined[image_idx] * 0.5 + 0.5
                    img_name = view_name[image_idx].replace('rgb/', '').replace('.jpg', '')
                    # utils.save_image(outpaint_img, f'res_outpaint_{args.dataset}/{img_name}.png')
                    utils.save_image(refined_img, f'res_refined_{args.dataset}/{img_name}.png')
            

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_points import knn
except (ModuleNotFoundError, ImportError):
    from torch_points_kernels import knn

def knn_func(coords1, coords2, nb, device, random_sel=1):
    assert(coords1.device == torch.device('cpu'))
    assert(coords2.device == torch.device('cpu'))
    out1, out2 = knn(coords1.contiguous(), coords2.contiguous(), nb*random_sel)
    if random_sel > 1:
        sel = torch.randperm(nb*random_sel)
        out1 = out1[:,:,sel]
        out2 = out2[:,:,sel]
    return out1.to(device), out2.to(device)

class SharedMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        transpose=False,
        padding_mode='zeros',
        bn=False,
        activation_fn=None
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding_mode=padding_mode
        )
        # self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.batch_norm = nn.GroupNorm(16, out_channels, eps=1e-6) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        r"""
            Forward pass of the network
            Parameters
            ----------
            input: torch.Tensor, shape (B, d_in, N, K)
            Returns
            -------
            torch.Tensor, shape (B, d_out, N, K)
        """
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Module):
    def __init__(self, d, num_neighbors, device):
        super(LocalSpatialEncoding, self).__init__()

        self.num_neighbors = num_neighbors
        self.mlp = SharedMLP(10, d, bn=True, activation_fn=nn.ReLU())

        self.device = device

    def forward(self, coords, features, knn_output):
        r"""
            Forward pass
            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: torch.Tensor, shape (B, d, N, 1)
                features of the point cloud
            neighbors: tuple
            Returns
            -------
            torch.Tensor, shape (B, 2*d, N, K)
        """
        # finding neighboring points
        idx, dist = knn_output
        B, N, K = idx.size()
        # idx(B, N, K), coords(B, N, 3)
        # neighbors[b, i, n, k] = coords[b, idx[b, n, k], i] = expanded_coords[b, i, extended_idx[b, i, n, k], k]
        expanded_idx = idx.unsqueeze(1).expand(B, 3, N, K)
        expanded_coords = coords.transpose(-2,-1).unsqueeze(-1).expand(B, 3, N, K)
        neighbor_coords = torch.gather(expanded_coords, 2, expanded_idx) # shape (B, 3, N, K)

        expanded_idx = idx.unsqueeze(1).expand(B, features.size(1), N, K)
        expanded_features = features.expand(B, -1, N, K)
        neighbor_features = torch.gather(expanded_features, 2, expanded_idx)
        # if USE_CUDA:
        #     neighbors = neighbors.cuda()

        # relative point position encoding
        concat = torch.cat((
            expanded_coords,
            neighbor_coords,
            expanded_coords - neighbor_coords,
            dist.unsqueeze(-3)
        ), dim=-3).to(features.device)
        return torch.cat((
            self.mlp(concat),
            neighbor_features
        ), dim=-3)



class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        # self.score_fn = nn.Sequential(
        #     nn.Linear(in_channels, in_channels, bias=False),
        #     nn.Softmax(dim=-2)
        # )
        self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.ReLU())

    def forward(self, x):
        r"""
            Forward pass
            Parameters
            ----------
            x: torch.Tensor, shape (B, d_in, N, K)
            Returns
            -------
            torch.Tensor, shape (B, d_out, N, 1)
        """
        # computing attention scores
        # scores = self.score_fn(x.permute(0,2,3,1)).permute(0,3,1,2)

        # sum over the neighbors
        # features = torch.sum(scores * x, dim=-1, keepdim=True) # shape (B, d_in, N, 1)

        features = torch.mean(x, dim=-1, keepdim=True)

        return self.mlp(features)



class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors, device):
        super(LocalFeatureAggregation, self).__init__()

        self.num_neighbors = num_neighbors

        self.mlp1 = SharedMLP(d_in, d_out//2, activation_fn=nn.LeakyReLU(0.2))
        self.mlp2 = SharedMLP(d_out, 2*d_out)
        self.shortcut = SharedMLP(d_in, 2*d_out, bn=True)

        self.lse1 = LocalSpatialEncoding(d_out//2, num_neighbors, device)
        self.lse2 = LocalSpatialEncoding(d_out//2, num_neighbors, device)

        self.pool1 = AttentivePooling(d_out, d_out//2)
        self.pool2 = AttentivePooling(d_out, d_out)

        self.lrelu = nn.LeakyReLU()

    def forward(self, coords, features):
        r"""
            Forward pass
            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: torch.Tensor, shape (B, d_in, N, 1)
                features of the point cloud
            Returns
            -------
            torch.Tensor, shape (B, 2*d_out, N, 1)
        """
        knn_output = knn_func(coords, coords, self.num_neighbors, features.device)
        # knn_output = knn(coords.cpu().contiguous(), coords.cpu().contiguous(), self.num_neighbors)
        coords = coords.to(features.device)

        x = self.mlp1(features)

        x = self.lse1(coords, x, knn_output)
        x = self.pool1(x)

        x = self.lse2(coords, x, knn_output)
        x = self.pool2(x)

        return self.lrelu(self.mlp2(x) + self.shortcut(features))



class RandLANet(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors=8, decimation=4, device=torch.device('cpu')):
        super(RandLANet, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_neighbors = num_neighbors
        self.decimation = decimation

        self.fc_start = nn.Linear(d_in, 64)
        self.bn_start = nn.Sequential(
            # nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            nn.GroupNorm(16, 64, eps=1e-6),
            nn.LeakyReLU(0.2)
        )

        # encoding layers
        self.encoder = nn.ModuleList([
            LocalFeatureAggregation(64, 32, num_neighbors, device),
            LocalFeatureAggregation(64, 64, num_neighbors, device),
            LocalFeatureAggregation(128, 128, num_neighbors, device),
            LocalFeatureAggregation(256, 256, num_neighbors, device)
        ])

        self.mlp = SharedMLP(512, 512, activation_fn=nn.ReLU())

        # decoding layers
        decoder_kwargs = dict(
            transpose=True,
            bn=True,
            activation_fn=nn.ReLU()
        )
        self.decoder = nn.ModuleList([
            SharedMLP(1024, 256, **decoder_kwargs),
            SharedMLP(512, 128, **decoder_kwargs),
            SharedMLP(256, 64, **decoder_kwargs),
            SharedMLP(128, 64, **decoder_kwargs)
        ])

        # final semantic prediction
        self.fc_end = nn.Sequential(
            SharedMLP(64, 64, bn=True, activation_fn=nn.ReLU()),
            SharedMLP(64, 32, bn=True, activation_fn=nn.ReLU()),
            nn.Dropout(),
            SharedMLP(32, d_out)#, activation_fn=nn.Tanh())
        )
        self.device = device

        self = self.to(device)

    def forward(self, input):
        r"""
            Forward pass
            Parameters
            ----------
            input: torch.Tensor, shape (B, N, d_in)
                input points
            Returns
            -------
            torch.Tensor, shape (B, num_classes, N)
                segmentation scores for each point
        """
        N = input.size(1)
        d = self.decimation

        coords = input[...,:3].clone().cpu()
        x = self.fc_start(input[...,3:]).transpose(-2,-1).unsqueeze(-1)
        x = self.bn_start(x) # shape (B, d, N, 1)

        decimation_ratio = 1

        # <<<<<<<<<< ENCODER
        x_stack = []

        permutation = torch.randperm(N)
        coords = coords[:,permutation]
        x = x[:,:,permutation]

        for lfa in self.encoder:
            # at iteration i, x.shape = (B, N//(d**i), d_in)
            x = lfa(coords[:,:N//decimation_ratio], x)
            x_stack.append(x.clone())
            decimation_ratio *= d
            x = x[:,:,:N//decimation_ratio]

        # # >>>>>>>>>> ENCODER

        x = self.mlp(x)

        # <<<<<<<<<< DECODER
        for mlp in self.decoder:
            neighbors, _ = knn(
                coords[:,:N//decimation_ratio].cpu().contiguous(), # original set
                coords[:,:d*N//decimation_ratio].cpu().contiguous(), # upsampled set
                1
            ) # shape (B, N, 1)
            neighbors = neighbors.to(self.device)

            extended_neighbors = neighbors.unsqueeze(1).expand(-1, x.size(1), -1, 1)

            x_neighbors = torch.gather(x, -2, extended_neighbors)

            x = torch.cat((x_neighbors, x_stack.pop()), dim=1)

            x = mlp(x)

            decimation_ratio //= d

        # >>>>>>>>>> DECODER
        # inverse permutation
        x = x[:,:,torch.argsort(permutation)]

        scores = self.fc_end(x)

        return scores.squeeze(-1)


if __name__ == '__main__':
    import time
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    d_in = 32
    cloud = 1000*torch.randn(1, 50000, d_in+3).to(device)
    model = RandLANet(d_in, 32, device=device)
    # model.load_state_dict(torch.load('checkpoints/checkpoint_100.pth'))
    # model.eval()

    t0 = time.time()
    pred = model(cloud)
    t1 = time.time()
    # print(pred)
    print(t1-t0)
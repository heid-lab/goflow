import torch
import torch.nn.functional as F
from torch.nn import Module, Sequential, ModuleList, Linear, Embedding
from torch_geometric.nn import MessagePassing, radius_graph
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from math import pi as PI

from utils.chem import BOND_TYPES
from ..common import MeanReadout, SumReadout, MultiLayerPerceptron


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class AsymmetricSineCosineSmearing(Module):
    def __init__(self, num_basis=50):
        super().__init__()
        num_basis_k = num_basis // 2
        num_basis_l = num_basis - num_basis_k
        self.register_buffer("freq_k", torch.arange(1, num_basis_k + 1).float())
        self.register_buffer("freq_l", torch.arange(1, num_basis_l + 1).float())

    @property
    def num_basis(self):
        return self.freq_k.size(0) + self.freq_l.size(0)

    def forward(self, angle):
        # If we don't incorporate `cos`, the embedding of 0-deg and 180-deg will be the
        #  same, which is undesirable.
        s = torch.sin(
            angle.view(-1, 1) * self.freq_k.view(1, -1)
        )  # (num_angles, num_basis_k)
        c = torch.cos(
            angle.view(-1, 1) * self.freq_l.view(1, -1)
        )  # (num_angles, num_basis_l)
        return torch.cat([s, c], dim=-1)


class SymmetricCosineSmearing(Module):
    def __init__(self, num_basis=50):
        super().__init__()
        self.register_buffer("freq_k", torch.arange(1, num_basis + 1).float())

    @property
    def num_basis(self):
        return self.freq_k.size(0)

    def forward(self, angle):
        return torch.cos(
            angle.view(-1, 1) * self.freq_k.view(1, -1)
        )  # (num_angles, num_basis)


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff, smooth):
        super(CFConv, self).__init__(aggr="add")
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff
        self.smooth = smooth

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_length, edge_attr):
        if self.smooth:
            C = 0.5 * (torch.cos(edge_length * PI / self.cutoff) + 1.0)
            C = (
                C * (edge_length <= self.cutoff) * (edge_length >= 0.0)
            )  # Modification: cutoff
        else:
            C = (edge_length <= self.cutoff).float()
        W = self.nn(edge_attr) * C.view(-1, 1)
        #W = self.nn(edge_attr)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff, smooth):
        super(InteractionBlock, self).__init__()
        mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(
            hidden_channels, hidden_channels, num_filters, mlp, cutoff, smooth
        )
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_length, edge_attr):
        x = self.conv(x, edge_index, edge_length, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class SchNetEncoder(Module):
    def __init__(
        self,
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        edge_channels=100,
        cutoff=10.0,
        smooth=False,
        embedding=False,
        edge_emb=None,
        edge_activation="ReLU"
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.cutoff = cutoff
        self.embedding = embedding
        if self.embedding:
            self.node_emb = Embedding(100, hidden_channels, max_norm=10.0)

        if edge_emb is not None:
            self.edge_emb = edge_emb
            self.edge_cat = torch.nn.Sequential(
                    torch.nn.Linear(hidden_channels *2, hidden_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_channels, hidden_channels))
            self.edge_d_emb = MultiLayerPerceptron(
                    1, 
                    [hidden_channels, hidden_channels], 
                    activation=edge_activation
                    )

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(
                hidden_channels, edge_channels, num_filters, cutoff, smooth
            )
            self.interactions.append(block)
    
    @classmethod
    def from_config(cls, config):
        if config.edge_emb:
            from edge import MLPEdgeEncoder
            edge_emb = MLPEdgeEncoder(config.hidden_dim, config.mlp_act)
        else:
            edge_emb = None

        #print(f"hidden_channels:{config.hidden_dim}")
        #print(f"num_filters:{config.hidden_dim}")
        #print(f"num_interactions:{config.num_convs}")
        #print(f"cutoff:{config.cutoff}")
        #print(f"smooth:{config.smooth_conv}")
        #print(f"embedding:{False}")
        #print(f"edge_emb:{edge_emb}")
        #print(f"edge_activation:{config.mlp_act}")

        encoder = cls(
                hidden_channels=config.hidden_dim,
                num_filters=config.hidden_dim,
                num_interactions=config.num_convs,
                edge_channels=config.hidden_dim,
                cutoff=config.cutoff,
                smooth=config.smooth_conv,
                embedding=False,
                edge_emb=edge_emb,
                edge_activation=config.mlp_act
                )
        return encoder

    def forward(
        self, z, edge_index, edge_length, edge_attr=None, embed_node=False, **kwargs
    ):
        if embed_node:
            assert z.dim() == 1 and z.dtype == torch.long and self.embedding
            h = self.node_emb(z)
        else:
            h = z

        if edge_attr is None:
            if hasattr(kwargs, "edge_type"):
                edge_type_r, edge_type_p = kwargs["edge_type"]
                edge_emb_r = self.edge_emb(edge_type_r) 
                edge_emb_p = self.edge_emb(edge_type_p) 
                edge_d_emb = self.edge_d_emb(edge_length)
                edge_attr = self.edge_cat(
                        torch.cat(
                            [edge_d_emb * edge_emb_r, edge_d_emb * edge_emb_p],
                            -1)
                        )
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_length, edge_attr)
        return h

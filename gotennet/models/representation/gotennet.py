from functools import partial
from typing import Callable, Optional, Tuple, Mapping, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter, to_dense_adj, dense_to_sparse
from torch_geometric.typing import OptTensor
from einops import rearrange, reduce
import math

from gotennet.models.components.ops import Dense, str2basis, get_weight_init_by_string, str2act, MLP, Distance, \
    CosineCutoff, \
    VecLayerNorm
from gotennet.models.components.ops import parse_update_info, TensorInit, NodeInit, EdgeInit, AtomCGREmbedding, \
    EdgeCGREmbedding
from gotennet.utils import (
    RankedLogger,
    _extend_condensed_graph_edge
)

log = RankedLogger(__name__, rank_zero_only=True)

# num_nodes and hidden_dims are placeholder values, will be overwritten by actual data
num_nodes = hidden_dims = 1

from torch_geometric.utils import softmax


# noinspection PyMethodOverriding

def lmax_tensor_size(lmax):
    return ((lmax + 1) ** 2) - 1


# splits tensor into list of size lmax.
# list at index l contains irreps of order l (with d channels)
def split_degree(tensor, lmax, dim=-1):  # default to last dim
    cumsum = 0
    tensors = []
    for i in range(1, lmax + 1):
        l_vec_size = lmax_tensor_size(i) - lmax_tensor_size(i - 1)
        # Create slice object for the specified dimension
        slc = [slice(None)] * tensor.ndim  # Create list of slice(None) for all dims
        slc[dim] = slice(cumsum, cumsum + l_vec_size)  # Replace desired dim with actual slice
        tensors.append(tensor[tuple(slc)])  # take slice of tensor at that dim
        cumsum += l_vec_size
    return tensors


class GATA(MessagePassing):
    def __init__(self, n_atom_basis: int, activation: Callable, weight_init=nn.init.xavier_uniform_,
                 bias_init=nn.init.zeros_,
                 aggr="add", node_dim=0, epsilon: float = 1e-7,
                 layer_norm="", vector_norm="", cutoff=5.0, scaling=1.0, num_heads=8, dropout=0.0,
                 edge_updates=True, last_layer=False, scale_edge=True,
                 edge_ln='', evec_dim=None, emlp_dim=None, sep_vecj=True, lmax=1):
        """
        Args:
            n_atom_basis (int): Number of features to describe atomic environments.
            activation (Callable): Activation function to be used. If None, no activation function is used.
            weight_init (Callable): Weight initialization function.
            bias_init (Callable): Bias initialization function.
            aggr (str): Aggregation method ('add', 'mean' or 'max').
            node_dim (int): The axis along which to aggregate.
        """
        super(GATA, self).__init__(aggr=aggr, node_dim=node_dim)
        self.lmax = lmax
        self.sep_vecj = sep_vecj
        self.epsilon = epsilon
        self.last_layer = last_layer
        self.edge_updates = edge_updates
        self.scale_edge = scale_edge
        self.activation = activation

        self.update_info = parse_update_info(edge_updates)

        self.dropout = dropout
        self.n_atom_basis = n_atom_basis

        InitDense = partial(Dense, weight_init=weight_init, bias_init=bias_init)
        self.gamma_s = nn.Sequential(
            InitDense(n_atom_basis, n_atom_basis, activation=activation),
            InitDense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )

        self.num_heads = num_heads
        self.q_w = InitDense(n_atom_basis, n_atom_basis, activation=None)
        self.k_w = InitDense(n_atom_basis, n_atom_basis, activation=None)

        self.gamma_v = nn.Sequential(
            InitDense(n_atom_basis, n_atom_basis, activation=activation),
            InitDense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )

        self.phik_w_ra = InitDense(
            n_atom_basis,
            n_atom_basis,
            activation=activation,
        )

        InitMLP = partial(MLP, weight_init=weight_init, bias_init=bias_init)

        self.edge_vec_dim = n_atom_basis if evec_dim is None else evec_dim
        self.edge_mlp_dim = n_atom_basis if emlp_dim is None else emlp_dim
        if not self.last_layer and self.edge_updates:
            if self.update_info["mlp"] or self.update_info["mlpa"]:
                dims = [n_atom_basis, self.edge_mlp_dim, n_atom_basis]
            else:
                dims = [n_atom_basis, n_atom_basis]
            self.edge_attr_up = InitMLP(dims, activation=activation,
                                        last_activation=None if self.update_info["mlp"] else self.activation,
                                        norm=edge_ln)
            self.vecq_w = InitDense(n_atom_basis, self.edge_vec_dim, activation=None, bias=False)

            if self.sep_vecj:
                self.veck_w = nn.ModuleList([
                    InitDense(n_atom_basis, self.edge_vec_dim, activation=None, bias=False)
                    for _ in range(self.lmax)
                ])
            else:
                self.veck_w = InitDense(n_atom_basis, self.edge_vec_dim, activation=None, bias=False)

            if self.update_info["lin_w"] > 0:
                modules = []
                if self.update_info["lin_w"] % 10 == 2:
                    modules.append(self.activation)
                self.lin_w_linear = InitDense(self.edge_vec_dim, n_atom_basis, activation=None,
                                              norm="layer" if self.update_info["lin_ln"] == 2 else "")
                modules.append(self.lin_w_linear)
                self.lin_w = nn.Sequential(*modules)

        self.down_proj = nn.Identity()

        self.cutoff = CosineCutoff(cutoff, scaling)
        self._alpha = None

        self.w_re = InitDense(
            n_atom_basis,
            n_atom_basis * 3,
            None,
        )

        self.layernorm_ = layer_norm
        self.vector_norm_ = vector_norm
        if layer_norm != "":
            self.layernorm = nn.LayerNorm(n_atom_basis)
        else:
            self.layernorm = nn.Identity()
        if vector_norm != "":
            self.tln = VecLayerNorm(n_atom_basis, trainable=False, norm_type=vector_norm)
        else:
            self.tln = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        if self.layernorm_:
            self.layernorm.reset_parameters()
        if self.vector_norm_:
            self.tln.reset_parameters()
        for l in self.gamma_s:
            l.reset_parameters()

        self.q_w.reset_parameters()
        self.k_w.reset_parameters()
        for l in self.gamma_v:
            l.reset_parameters()
        self.w_re.reset_parameters()

        if not self.last_layer and self.edge_updates:
            self.edge_attr_up.reset_parameters()
            self.vecq_w.reset_parameters()

            if self.sep_vecj:
                for w in self.veck_w:
                    w.reset_parameters()
            else:
                self.veck_w.reset_parameters()

            if self.update_info["lin_w"] > 0:
                self.lin_w_linear.reset_parameters()


    def forward(
            self,
            edge_index,
            s: torch.Tensor,  # h in the paper
            t: torch.Tensor,  # X^(l) in the paper
            dir_ij: torch.Tensor,  # edge dir vector
            r_ij: torch.Tensor,  # Naming r_ij is misleading. This is t_ij in the paper.
            d_ij: torch.Tensor,  # edge lengths
            num_edges_expanded: torch.Tensor,
            # per edge the degree of the edge_index[0] (src) node (default flow is src to dst)
    ):
        """Compute interaction output. """
        s = self.layernorm(s)
        t = self.tln(t)

        q = self.q_w(s).reshape(-1, self.num_heads, self.n_atom_basis // self.num_heads)
        k = self.k_w(s).reshape(-1, self.num_heads, self.n_atom_basis // self.num_heads)

        x = self.gamma_s(s)  # MLP in split
        val = self.gamma_v(s)  # MLP for values
        f_ij = r_ij
        r_ij_attn = self.phik_w_ra(r_ij)
        r_ij = self.w_re(r_ij)

        # propagate_type: (x: Tensor, ten: Tensor, q:Tensor, k:Tensor, val:Tensor, r_ij: Tensor, r_ij_attn: Tensor, d_ij:Tensor, dir_ij: Tensor, num_edges_expanded: Tensor)
        su, tu = self.propagate(edge_index=edge_index, x=x, q=q, k=k, val=val,
                                ten=t, r_ij=r_ij, r_ij_attn=r_ij_attn, d_ij=d_ij, dir_ij=dir_ij,
                                num_edges_expanded=num_edges_expanded)

        s = s + su
        t = t + tu

        if not self.last_layer and self.edge_updates:
            vec = t

            w1 = self.vecq_w(vec)
            if self.sep_vecj:
                vec_split = split_degree(vec, self.lmax, dim=1)
                w_out = torch.concat([
                    w(vec_split[i]) for i, w in enumerate(self.veck_w)
                ], dim=1)

            else:
                w_out = self.veck_w(vec)

            # edge_updater_type: (w1: Tensor, w2:Tensor,  d_ij: Tensor, f_ij: Tensor)
            # w1: EQ in paper
            # w2: EK in paper
            # f_ij: t_ij in the paper (above it is renamed from r_ij to f_ij)
            df_ij = self.edge_updater(edge_index, w1=w1, w2=w_out, d_ij=dir_ij, f_ij=f_ij)
            df_ij = f_ij + df_ij
            self._alpha = None
            return s, t, df_ij
        else:
            self._alpha = None
            return s, t, f_ij

    # https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
    # we generally refer to i as the central nodes that aggregates information, and refer to 
    # j as the neighboring nodes, since this is the most common notation
    def message(
            self,
            edge_index,
            x_i: torch.Tensor,
            x_j: torch.Tensor,
            q_i: torch.Tensor,
            k_j: torch.Tensor,
            val_j: torch.Tensor,
            ten_j: torch.Tensor,  # mu/X^(l)
            r_ij: torch.Tensor,
            r_ij_attn: torch.Tensor,
            d_ij: torch.Tensor,
            dir_ij: torch.Tensor,  # tensor_dir
            num_edges_expanded: torch.Tensor,
            # index: maps each edge feature to its target node. In PyTorch Geometric, this is typically the second row of the edge_index tensor
            # ptr: https://github.com/pyg-team/pytorch_geometric/discussions/4332
            index: torch.Tensor, ptr: OptTensor,
            dim_size: Optional[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute message passing.
        """

        r_ij_attn = r_ij_attn.reshape(-1, self.num_heads, self.n_atom_basis // self.num_heads)
        attn = (q_i * k_j * r_ij_attn).sum(dim=-1, keepdim=True)

        attn = softmax(attn, index, ptr, dim_size)

        # Normalize the attention scores
        if self.scale_edge:
            norm = torch.sqrt(num_edges_expanded.reshape(-1, 1, 1)) / np.sqrt(self.n_atom_basis)
        else:
            norm = 1.0 / np.sqrt(self.n_atom_basis)
        attn = attn * norm
        self._alpha = attn
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        self_attn = attn * val_j.reshape(-1, self.num_heads, (self.n_atom_basis * 3) // self.num_heads)
        SEA = self_attn.reshape(-1, 1, self.n_atom_basis * 3)  # extra dim at 1 for irreps (up to L_max)

        x = SEA + (r_ij.unsqueeze(1) * x_j * self.cutoff(d_ij.unsqueeze(-1).unsqueeze(-1)))  # extra dim at 1 for irreps

        o_s, o_d, o_t = torch.split(x, self.n_atom_basis, dim=-1)
        dmu = o_d * dir_ij[..., None] + o_t * ten_j  # dir_ij has irreps up to L_max
        return o_s, dmu

    @staticmethod
    def rej(vec, d_ij):
        d_ij_1 = rearrange(d_ij, 'b l -> b l 1')
        vec_proj = reduce(vec * d_ij_1, 'b l c -> b 1 c', 'sum')
        return vec - vec_proj * d_ij_1

    # w1_i: EQ_i in paper
    # w2_j: EK_j in paper
    # d_ij: name misleading. It is dir_ij (irreps vector)
    # f_ij: t_ij in the paper
    def edge_update(self, w1_i, w2_j, d_ij, f_ij):
        if self.sep_vecj:
            vi = w1_i
            vj = w2_j
            vi_split = split_degree(vi, self.lmax, dim=1)
            vj_split = split_degree(vj, self.lmax, dim=1)
            d_ij_split = split_degree(d_ij, self.lmax, dim=1)

            pairs = []
            for i in range(len(vi_split)):
                if self.update_info["rej"]:
                    w1 = self.rej(vi_split[i], d_ij_split[i])
                    w2 = self.rej(vj_split[i], -d_ij_split[i])
                    pairs.append((w1, w2))
                else:
                    w1 = vi_split[i]
                    w2 = vj_split[i]
                    pairs.append((w1, w2))
        elif not self.update_info["rej"]:
            w1 = w1_i
            w2 = w2_j
            pairs = [(w1, w2)]
        else:
            w1 = self.rej(w1_i, d_ij)
            w2 = self.rej(w2_j, -d_ij)
            pairs = [(w1, w2)]

        w_dot_sum = None
        for el in pairs:
            w1, w2 = el
            w_dot = (w1 * w2).sum(dim=1)
            if w_dot_sum is None:
                w_dot_sum = w_dot
            else:
                w_dot_sum = w_dot_sum + w_dot
        w_dot = w_dot_sum
        if self.update_info["lin_w"] > 0:
            w_dot = self.lin_w(w_dot)

        if self.update_info["gated"] == "gatedt":
            w_dot = torch.tanh(w_dot)
        elif self.update_info["gated"] == "gated":
            w_dot = torch.sigmoid(w_dot)
        elif self.update_info["gated"] == "act":
            w_dot = self.activation(w_dot)

        df_ij = self.edge_attr_up(f_ij) * w_dot
        return df_ij

    # noinspection PyMethodOverriding
    def aggregate(
            self,
            features: Tuple[torch.Tensor, torch.Tensor],
            index: torch.Tensor,
            ptr: Optional[torch.Tensor],
            dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x_ = scatter(x, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        vec_ = scatter(vec, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return x_, vec_

    def update(
            self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs


class EQFF(nn.Module):
    def __init__(self, n_atom_basis: int, activation: Callable, epsilon: float = 1e-8,
                 weight_init=nn.init.xavier_uniform_, bias_init=nn.init.zeros_, vec_dim=None):
        """Equiavariant Feed Forward layer."""
        super(EQFF, self).__init__()
        self.n_atom_basis = n_atom_basis

        InitDense = partial(Dense, weight_init=weight_init, bias_init=bias_init)

        vec_dim = n_atom_basis if vec_dim is None else vec_dim

        self.gamma_m = nn.Sequential(
            InitDense(2 * n_atom_basis, n_atom_basis, activation=activation),
            InitDense(n_atom_basis, 2 * n_atom_basis, activation=None),
        )
        self.w_vu = InitDense(
            n_atom_basis, vec_dim, activation=None, bias=False
        )

        self.epsilon = epsilon

    def reset_parameters(self):
        self.w_vu.reset_parameters()
        for l in self.gamma_m:
            l.reset_parameters()

    # s: h in paper (node scalar features): [nodes, n_atom_basis]
    # v: X^(l) in paper (node higher-order features): [nodes, (l+1)**2, n_atom_basis]
    def forward(self, s, v):
        """Compute Equivariant Feed Forward output."""

        # t_prime: [nodes, (l+1)**2, n_atom_basis]
        t_prime = self.w_vu(v)
        # t_prime_mag has same dim as s: [nodes, 1, n_atom_basis]. dim=-2 is the irreps (size (l+1)**2) dim
        t_prime_mag = torch.sqrt(torch.sum(t_prime ** 2, dim=-2, keepdim=True) + self.epsilon)
        # combined_tensor: [nodes, 1, 2*n_atom_basis]
        combined_tensor = torch.cat([s, t_prime_mag], dim=-1)
        # m12: [nodes, 1, 2*n_atom_basis]
        m12 = self.gamma_m(combined_tensor)
        # m1, m2: [nodes, 1, n_atom_basis]
        m_1, m_2 = torch.split(m12, self.n_atom_basis, dim=-1)

        return s + m_1, v + m_2 * t_prime


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            Dense(embedding_dim, hidden_dim, norm='layer', activation=nn.SiLU()),
            Dense(hidden_dim, output_dim, norm='layer', activation=nn.SiLU()),
        )

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(timesteps.squeeze(), self.embedding_dim)
        t_emb_mlp = self.mlp(t_emb)
        return t_emb_mlp


class GotenNet(nn.Module):
    def __init__(
            self,
            n_atom_basis: int = 128,
            n_atom_feat_basis: int = 128,
            n_atom_rdkit_feats: int = 28,
            n_interactions: int = 8,
            radial_basis: Union[Callable, str] = 'BesselBasis',
            n_rbf: int = 20,
            cutoff_fn: Optional[Union[Callable, str]] = None,
            edge_order: int = 4,
            activation: Optional[Union[Callable, str]] = F.silu,
            max_z: int = 100,
            epsilon: float = 1e-8,
            weight_init=nn.init.xavier_uniform_,
            bias_init=nn.init.zeros_,
            max_num_neighbors: int = 32,
            int_layer_norm="",
            int_vector_norm="",
            num_heads=8,
            attn_dropout=0.0,
            edge_updates=True,
            scale_edge=True,
            lmax=2,
            aggr="add",
            edge_ln='',
            evec_dim=None,
            emlp_dim=None,
            sep_int_vec=True,
    ):
        """
        Representation for GotenNet
        """
        super().__init__()

        self.scale_edge = scale_edge
        if type(weight_init) == str:
            log.info(f'Using {weight_init} weight initialization')
            weight_init = get_weight_init_by_string(weight_init)

        if type(bias_init) == str:
            bias_init = get_weight_init_by_string(bias_init)

        if type(activation) is str:
            activation = str2act(activation)

        self.n_atom_basis = self.hidden_dim = n_atom_basis
        # self.hidden_dim = n_atom_basis + n_atom_feat_basis
        self.n_interactions = n_interactions
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff
        self.scaling = cutoff_fn.scaling
        self.edge_order = edge_order

        self.distance = Distance(self.cutoff, max_num_neighbors=max_num_neighbors, loop=True)

        self.neighbor_embedding = NodeInit([self.hidden_dim // 2, self.hidden_dim], n_atom_rdkit_feats, n_rbf,
                                           self.cutoff, self.scaling, max_z=max_z,
                                           weight_init=weight_init, bias_init=bias_init, concat=False,
                                           proj_ln='layer', activation=activation).jittable()
        self.edge_embedding = EdgeInit(n_rbf, [self.hidden_dim // 2, self.hidden_dim], weight_init=weight_init,
                                       bias_init=bias_init,
                                       proj_ln='').jittable()

        self.time_embedding = TimestepEmbedding(128, 128, self.hidden_dim)

        radial_basis = str2basis(radial_basis)
        self.radial_basis = radial_basis(cutoff=self.cutoff, scaling=self.scaling, n_rbf=n_rbf)

        self.atom_cgr_embedding = AtomCGREmbedding(n_atom_rdkit_feats, n_atom_basis)
        self.edge_cgr_embedding = EdgeCGREmbedding(self.hidden_dim)

        self.tensor_init = TensorInit(l=lmax)

        self.gata = nn.ModuleList([
            GATA(
                n_atom_basis=self.n_atom_basis, activation=activation, aggr=aggr,
                weight_init=weight_init, bias_init=bias_init,
                layer_norm=int_layer_norm, vector_norm=int_vector_norm, cutoff=self.cutoff, scaling=self.scaling, epsilon=epsilon,
                num_heads=num_heads, dropout=attn_dropout,
                edge_updates=edge_updates, last_layer=(i == self.n_interactions - 1),
                scale_edge=scale_edge, edge_ln=edge_ln,
                evec_dim=evec_dim, emlp_dim=emlp_dim,
                sep_vecj=sep_int_vec, lmax=lmax
            ).jittable() for i in range(self.n_interactions)
        ])

        self.eqff = nn.ModuleList([
            EQFF(
                n_atom_basis=self.n_atom_basis, activation=activation, epsilon=epsilon,
                weight_init=weight_init, bias_init=bias_init
            ) for i in range(self.n_interactions)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        self.edge_embedding.reset_parameters()
        self.neighbor_embedding.reset_parameters()
        for l in self.gata:
            l.reset_parameters()
        for l in self.eqff:
            l.reset_parameters()

    def forward(self, x_t_N_3: Tensor, t_G: Tensor, inputs: Mapping[str, torch.Tensor]) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Compute atomic representations/embeddings.

        Args:
            inputs (Mapping[str, Tensor]): Dictionary of input tensors containing
            atomic_numbers, pos, batch, edge_index, r_ij, and dir_ij.

        Returns:
            Tuple[Tensor, Tensor]: Returns tuple of atomic representation and intermediate
            atom-wise representation q and mu, of respective shapes
            [num_nodes, 1, hidden_dims] and [num_nodes, 3, hidden_dims].
        """
        # get tensors from input dictionary
        # atomic_numbers, pos, batch, edge_index = inputs.z, inputs.pos, inputs.batch, inputs.edge_index
        edge_index, edge_type, batch = inputs.edge_index, inputs.edge_type, inputs.batch
        r_feat, p_feat, atom_type = inputs.r_feat, inputs.p_feat, inputs.atom_type

        edge_index, _, edge_type_r, edge_type_p = _extend_condensed_graph_edge(x_t_N_3, edge_index, edge_type, batch,
                                                                               cutoff=self.cutoff,
                                                                               edge_order=self.edge_order)

        edge_vec = x_t_N_3[edge_index[0]] - x_t_N_3[edge_index[1]]
        edge_weight = torch.norm(edge_vec, dim=-1)
        edge_attr = self.radial_basis(edge_weight)

        q = self.atom_cgr_embedding(atom_type, r_feat, p_feat)
        q = self.neighbor_embedding(atom_type, r_feat, p_feat, q, edge_index, edge_weight, edge_attr, edge_type_r,
                                    edge_type_p)
        t_emb_G = self.time_embedding(t_G)
        if batch is None:  # batch size = 1
            batch = torch.tensor([0] * len(atom_type), device=q.device)
        t_emb_N = t_emb_G[batch]
        q = q + t_emb_N

        edge_emb = self.edge_embedding(edge_index, edge_attr, edge_type_r, edge_type_p)
        edge_emb_t = edge_emb + t_emb_N[edge_index[0]]
        assert torch.allclose(t_emb_N[edge_index[0]], t_emb_N[edge_index[1]])

        mask = edge_index[0] != edge_index[1]
        # direction vector
        dist = torch.norm(edge_vec[mask], dim=1).unsqueeze(1)
        edge_vec[mask] = edge_vec[mask] / dist

        edge_vec = self.tensor_init(edge_vec)
        equi_dim = ((self.tensor_init.l + 1) ** 2) - 1
        # count number of edges for each node
        num_edges = scatter(torch.ones_like(edge_weight), edge_index[0], dim=0, reduce="sum")
        # the shape of num_edges is [num_nodes, 1], we want to expand this to [num_edges, 1]
        # Map num_edges back to the shape of attn using edge_index
        num_edges_expanded = num_edges[edge_index[0]]

        qs = q.shape
        mu = torch.zeros((qs[0], equi_dim, qs[1]), device=q.device)
        q.unsqueeze_(1)
        for i, (interaction, mixing) in enumerate(zip(self.gata, self.eqff)):
            # q: h in paper (node scalar features)
            # mu: X^(l) in paper (node higher-order features)
            q, mu, edge_attr = interaction(edge_index, q, mu, dir_ij=edge_vec, r_ij=edge_emb_t, d_ij=edge_weight,
                                           num_edges_expanded=num_edges_expanded)  # idx_i, idx_j, n_atoms, # , f_ij=f_ij
            q, mu = mixing(q, mu)

        q = q.squeeze(1)
        return q, mu

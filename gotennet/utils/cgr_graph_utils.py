import torch
from torch_geometric.nn import radius_graph, radius
from torch_sparse import coalesce
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from rdkit.Chem.rdchem import BondType

BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}


def _extend_condensed_graph_edge(pos, bond_index, bond_type, batch, cutoff=5.0, edge_order=4):
    N = pos.size(0)
    # global index            : atom pairs (i, j) which are closer than cutoff are added to local_index.
    # local index             : (i, j) pairs which are edge of R or P.
    # edge_type_r/edge_type_p : 0, 1, 2, ... 23, 24, ...
    #                           0 -> no edge (bond)
    #                           1, 2, 3 ... -> bond type
    #                           23, 24 -> meaning no bond, but higher order edge. (2-hop or 3-hop)
    edge_index_global, edge_index_local, edge_type_r, edge_type_p = extend_ts_graph_order_radius(
        N, pos, bond_index, bond_type, batch, order=edge_order, cutoff=cutoff
    )

    edge_type_global = torch.zeros_like(edge_index_global[0]) - 1
    adj_global = to_dense_adj(
        edge_index_global, edge_attr=edge_type_global, max_num_nodes=N
    )
    adj_local_r = to_dense_adj(
        edge_index_local, edge_attr=edge_type_r, max_num_nodes=N
    )
    adj_local_p = to_dense_adj(
        edge_index_local, edge_attr=edge_type_p, max_num_nodes=N
    )
    # if in the reactant graph there is no edge between two nodes
    # then add one (-1 value) if there is one in the global graph (which also contains radius-graph)
    adj_global_r = torch.where(adj_local_r != 0, adj_local_r, adj_global)
    adj_global_p = torch.where(adj_local_p != 0, adj_local_p, adj_global)
    # convert to dense again
    edge_index_global_r, edge_type_global_r = dense_to_sparse(adj_global_r)
    edge_index_global_p, edge_type_global_p = dense_to_sparse(adj_global_p)
    # those -1 values from before are now converted to 0 again. I am sure this is inefficient
    edge_type_global_r[edge_type_global_r < 0] = 0
    edge_type_global_p[edge_type_global_p < 0] = 0
    edge_index_global = edge_index_global_r

    return edge_index_global, edge_index_local, edge_type_global_r, edge_type_global_p


def extend_ts_graph_order_radius(
        num_nodes,
        pos,
        edge_index,
        edge_type,
        batch,
        order=3,
        cutoff=10.0,
):
    edge_index_local, edge_type_r, edge_type_p = _extend_ts_graph_order(
        num_nodes, edge_index, edge_type, batch, order=order
    )

    edge_index_global, _ = _extend_to_radius_graph(
        pos, edge_index_local, edge_type_r, cutoff, batch
    )

    return edge_index_global, edge_index_local, edge_type_r, edge_type_p


def _extend_ts_graph_order(num_nodes, edge_index, edge_type, batch, order=3):
    """
    Extend the connectivity of a graph by adding edges between nodes that 
    are connected via a specified number of intermediate edges (up to a specified "order"). 
    This is commonly referred to as adding higher-order connections
    Regarding the connection type (value):
    - if order 1 (bond): the value is the bond-type < num_types.
    - if order > 1: the value is the order-number + num_types.
    - if no bond: the value is 0
    """

    def binarize(x):
        # return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))
        return (x > 0).float()

    def get_higher_order_adj_matrix(adj, order):
        """
        Args:
            adj:        (N, N)
            type_mat:   (N, N)
        Returns:
            Dense adjacendy matrix. An edge has its order number (not just 1).
            Following attributes will be updated:
              - edge_index
              - edge_type
            Following attributes will be added to the data object:
              - bond_edge_index:  Original edge_index.
        """
        adj_mats = [
            torch.eye(adj.size(0), dtype=torch.long, device=adj.device),
            binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device)),
        ]

        for i in range(2, order + 1):
            adj_mats.append(binarize(adj_mats[i - 1] @ adj_mats[1]))
        order_mat = torch.zeros_like(adj)

        for i in range(1, order + 1):
            order_mat += (adj_mats[i] - adj_mats[i - 1]) * i

        return order_mat

    num_types = len(BOND_TYPES)
    N = num_nodes

    # get reactant bonds
    bond_type_r = edge_type // num_types
    mask_r = bond_type_r != 0
    bond_index_r = edge_index[:, mask_r]
    bond_type_r = bond_type_r[mask_r]
    # get product bonds
    bond_type_p = edge_type % num_types
    mask_p = bond_type_p != 0
    bond_index_p = edge_index[:, mask_p]
    bond_type_p = bond_type_p[mask_p]

    # for r: add higher order adj-mat to adj-mat
    adj_r = to_dense_adj(bond_index_r, max_num_nodes=N).squeeze(0)
    adj_order_r = get_higher_order_adj_matrix(adj_r, order)
    type_mat_r = to_dense_adj(
        bond_index_r, edge_attr=bond_type_r, max_num_nodes=N
    ).squeeze(0)
    type_highorder_r = torch.where(
        adj_order_r > 1,
        num_types + adj_order_r - 1,
        torch.zeros_like(adj_order_r),
    )
    assert (type_mat_r * type_highorder_r == 0).all()
    type_new_r = type_mat_r + type_highorder_r
    type_mask_r = -(type_new_r != 0).to(torch.float)

    # for p: add higher order adj-mat to adj-mat
    adj_p = to_dense_adj(bond_index_p, max_num_nodes=N).squeeze(0)
    adj_order_p = get_higher_order_adj_matrix(adj_p, order)
    type_mat_p = to_dense_adj(
        bond_index_p, edge_attr=bond_type_p, max_num_nodes=N
    ).squeeze(0)
    type_highorder_p = torch.where(
        adj_order_p > 1,
        num_types + adj_order_p - 1,
        torch.zeros_like(adj_order_p),
    )
    assert (type_mat_p * type_highorder_p == 0).all()
    # encoding is as follows:
    # all values < num_types are order 1, 
    # all >= num_types are higher order with their order-type added to (num_types - 1)
    type_new_p = type_mat_p + type_highorder_p
    type_mask_p = -(type_new_p != 0).to(torch.float)

    # if in, say the reactant graph, there no bond between 2 nodes (incl. no higher order)
    # then add -1 if there is a bond between those 2 nodes in the product graph
    type_r = torch.where(type_new_r != 0, type_new_r, type_mask_p).to(torch.long)
    type_p = torch.where(type_new_p != 0, type_new_p, type_mask_r).to(torch.long)

    # from dense to sparse
    edge_index_r, edge_type_r = dense_to_sparse(type_r)
    edge_index_p, edge_type_p = dense_to_sparse(type_p)
    # -1 edges (as expl. in prev. block) converted to 0
    edge_type_r[edge_type_r < 0] = 0
    edge_type_p[edge_type_p < 0] = 0

    # must be equivalent, since if no connection was present in say r (or p)
    # we added one, if there was one in p (or r)
    assert (edge_index_r == edge_index_p).all()

    # sort edge_index, along with its values, by edge_index[0]
    edge_index_local, edge_type_r = coalesce(edge_index_r, edge_type_r.long(), N, N)  # modify data
    _, edge_type_p = coalesce(edge_index_p, edge_type_p.long(), N, N)  # modify data

    return edge_index_local, edge_type_r, edge_type_p


def _extend_to_radius_graph(
        pos,
        edge_index,
        edge_type,
        cutoff,
        batch,
        unspecified_type_number=0,
        is_sidechain=None,
):
    """
    1. Contructs a sparse radius graph using radius_graph
    2. Adds that sparse graph to the existing one (received from _extend_ts_graph_order)
    3. Returns its indices and values (here we use 0 as values for edges)
    """

    assert edge_type.dim() == 1
    N = pos.size(0)

    # bgraph_adj = torch.sparse.LongTensor(edge_index, edge_type, torch.Size([N, N]))
    bgraph_adj = torch.sparse_coo_tensor(
        edge_index,
        edge_type,
        torch.Size([N, N]),
        dtype=torch.long,
        device=pos.device
    )

    if is_sidechain is None:
        rgraph_edge_index = radius_graph(pos, r=cutoff, batch=batch)  # (2, E_r)
    else:
        # fetch sidechain and its batch index
        is_sidechain = is_sidechain.bool()
        dummy_index = torch.arange(pos.size(0), device=pos.device)
        sidechain_pos = pos[is_sidechain]
        sidechain_index = dummy_index[is_sidechain]
        sidechain_batch = batch[is_sidechain]

        assign_index = radius(
            x=pos, y=sidechain_pos, r=cutoff, batch_x=batch, batch_y=sidechain_batch
        )
        r_edge_index_x = assign_index[1]
        r_edge_index_y = assign_index[0]
        r_edge_index_y = sidechain_index[r_edge_index_y]

        rgraph_edge_index1 = torch.stack((r_edge_index_x, r_edge_index_y))  # (2, E)
        rgraph_edge_index2 = torch.stack((r_edge_index_y, r_edge_index_x))  # (2, E)
        rgraph_edge_index = torch.cat(
            (rgraph_edge_index1, rgraph_edge_index2), dim=-1
        )  # (2, 2E)
        # delete self loop
        rgraph_edge_index = rgraph_edge_index[
                            :, (rgraph_edge_index[0] != rgraph_edge_index[1])
                            ]

    """rgraph_adj = torch.sparse.LongTensor(
        indices=rgraph_edge_index,
        values=torch.ones(rgraph_edge_index.size(1)).long().to(pos.device) * unspecified_type_number,
        size=torch.Size([N, N]),
    )"""
    rgraph_adj = torch.sparse_coo_tensor(
        indices=rgraph_edge_index,
        values=torch.ones(rgraph_edge_index.size(1), dtype=torch.long, device=pos.device) * unspecified_type_number,
        size=torch.Size([N, N]),
        dtype=torch.long,
        device=pos.device
    )

    composed_adj = (bgraph_adj + rgraph_adj).coalesce()  # Sparse (N, N, T)
    # edge_index = composed_adj.indices()
    # dist = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)

    new_edge_index = composed_adj.indices()
    new_edge_type = composed_adj.values().long()

    return new_edge_index, new_edge_type

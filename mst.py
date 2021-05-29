#!/usr/bin/env python

import torch
from tqdm.auto import tqdm

def prepare_weights_batched(weights, eps = 1e-6):
    """
    Prepare attention matrix for finding the maximum spanning tree
    """
#     weights = weights[:, -3:, ...]
    weights = torch.min(weights, weights.transpose(-2, -1))
    weights[:, :, :, torch.eye(weights.shape[-1]).bool()] = 0
    weights = (weights / (torch.sum(weights, axis=-1, keepdims=True) + eps))
    return weights

def get_edges_from_adj_batched(adj):
    # Duplicate edges are fine for the downstream task purposes.
    device = adj.device
    print(device)
    batch_size = adj.size(0)
    n = adj[0].size(1)
    edges = torch.combinations(torch.arange(n))
    edges = edges.expand(batch_size, edges.shape[0], edges.shape[1]).to(device)
    weights = adj[:, edges[0, :, 0], edges[0, :, 1]][:, None].to(device)
    weights_and_edges = torch.cat((weights.transpose(1, 2), edges), dim=2)
    return weights_and_edges, n

def get_root_pytorch_batched(parents, node, n):
    bs = parents.size(0)
    arange = torch.arange(bs)
    # Find path of nodes leading to the root.
    path = torch.zeros_like(parents)
    path[:, 0] = node
    root = parents[arange, node]
    # root1 = root
    # pth = root.expand(n, bs).transpose(0,1)
    for i in range(1, n):
        # print(torch.all(torch.eq(root1, root)))
        path[:, i] = root
        root = parents[arange, root]
    # print(torch.all(torch.eq(pth, path)))
    # Compress the path and return.
    # есть ощущение, что этот цикл ни на что не влияет, в смысле, что parents не
    # не меняется и равен входному.
#     for i in range(1, n):
#         parents[arange, path[:, i]] = root
    return parents, root


def kruskals_pytorch_batched(weights_and_edges, n):
    """Batched kruskal's algorithm in Pytorch.
    Args:
        weights_and_edges: Shape (batch size, n * (n - 1) / 2, 3), where
            weights_and_edges[.][i] = [weight_i, node1_i, node2_i] for edge i.
        n: Number of nodes.
    Returns:
        Adjacency matrix with diagonal removed. Shape (batch size, n, n)
    """
    
    device = weights_and_edges.device
    batch_size = weights_and_edges.size(0)
    arange = torch.arange(batch_size)
    h0s = torch.zeros(batch_size, device=device)
    # Sort edges based on weights, in descending order.
    sorted_weights = torch.argsort(
        weights_and_edges[:, :, 0], -1, descending=True)
    dummy = sorted_weights.unsqueeze(2).expand(
        *(sorted_weights.shape + (weights_and_edges.size(2),)))
    gather = torch.gather(weights_and_edges, 1, dummy)[:, :, 0:]
    sorted_w = gather[:, :, 0].transpose(0, 1)
    sorted_edges = gather[:, :, 1:].transpose(0, 1).long()
    weights = torch.ones((batch_size, n)).to(device)
    parents = torch.arange(n).repeat((batch_size, 1)).to(device)
    adj_matrix = torch.zeros((batch_size, n, n)).to(device)
    for k, edge in enumerate(tqdm(sorted_edges)):
        i, j = edge.transpose(0, 1)
        parents, root_i = get_root_pytorch_batched(parents, i, n)
        parents, root_j = get_root_pytorch_batched(parents, j, n)


        is_i_and_j_not_in_same_forest = (root_i != root_j).int()

        # Combine two forests if i and j are not in the same forest.
        is_i_heavier_than_j = (
                weights[arange, root_i] > weights[arange, root_j]).int()
        weights_root_i = weights[arange, root_i] + (
                (weights[arange, root_j] * is_i_heavier_than_j)
                * is_i_and_j_not_in_same_forest +
                0.0 * (1.0 - is_i_and_j_not_in_same_forest))
        parents_root_i = (
                (root_i * is_i_heavier_than_j + root_j * (1 - is_i_heavier_than_j))
                * is_i_and_j_not_in_same_forest +
                root_i * (1 - is_i_and_j_not_in_same_forest))
        weights_root_j = weights[arange, root_j] + (
                weights[arange, root_i] * (1 - is_i_heavier_than_j)
                * is_i_and_j_not_in_same_forest +
                0.0 * (1.0 - is_i_and_j_not_in_same_forest))
        parents_root_j = (
                (root_i * is_i_heavier_than_j + root_j * (1 - is_i_heavier_than_j))
                * is_i_and_j_not_in_same_forest +
                root_j * (1 - is_i_and_j_not_in_same_forest))
        weights[arange, root_i] = weights_root_i
        weights[arange, root_j] = weights_root_j
        parents[arange, root_i] = parents_root_i
        parents[arange, root_j] = parents_root_j

        # Update adjacency matrix.
#         adj_matrix[arange, i, j] = is_i_and_j_not_in_same_forest.float() * sorted_w[k, arange]
#         adj_matrix[arange, j, i] = is_i_and_j_not_in_same_forest.float() * sorted_w[k, arange]
        h0s[arange] += is_i_and_j_not_in_same_forest.float() * sorted_w[k, arange]
    return h0s

def get_max_spanning_tree(W):
    W = prepare_weights_batched(W)
    bs, layer, head, n, _ = W.shape
    W = W.reshape(bs*layer*head, n, n)
    W, n = get_edges_from_adj_batched(W)
    h0s = kruskals_pytorch_batched(W, n)
    h0s = h0s.reshape(bs, layer, head)
#     adj_matrix = kruskals_pytorch_batched(W, n)
#     adj_matrix = adj_matrix.reshape(bs, layer, head, n, n)
    return h0s #adj_matrix
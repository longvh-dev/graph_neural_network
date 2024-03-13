import json

import numpy as np
import scipy.sparse as sparse
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = { c: np.identity(len(classes))[i, :] for i, c in
                     enumerate(classes) }
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype = np.int32)
    return labels_onehot


def load_data(path, dataset = 'ESConv'):
    """Load citation network dataset (ESConv only for now)"""
    print('Loading {} dataset...'.format(dataset))
    # dataset in type of ESConv only for now
    with open(path, 'r') as f:
        conversations = json.load(f)
    print(type(conversations[0]))
    return conversations[0]


def normalize_adjacency(adj):

    adj = adj + sparse.eye(adj.shape[0])

    degrees = np.array(adj.sum(1))
    degrees = np.power(degrees, -0.5).flatten()
    degrees[np.isinf(degrees)] = 0.
    degrees[np.isnan(degrees)] = 0.
    degree_matrix = sparse.diags(degrees)

    adj = degree_matrix @ adj @ degree_matrix
    return adj


def spicy_to_torch_sparse(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def accuracy(output, labels):
    predictions = output.max(1)[1].type_as(labels)
    correct = predictions.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

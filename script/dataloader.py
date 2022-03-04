import os
import numpy as np
import scipy.sparse as sp

def norm_feat(feature):
    feature = feature.astype(dtype=np.float32)
    if sp.issparse(feature):
        row_sum = feature.sum(axis=1).A1
        row_sum_inv = np.power(row_sum, -1)
        row_sum_inv[np.isinf(row_sum_inv)] = 0.
        deg_inv = sp.diags(row_sum_inv, format='csc')
        norm_feature = deg_inv.dot(feature)
    else:
        row_sum_inv = np.power(np.sum(feature, axis=1), -1)
        row_sum_inv[np.isinf(row_sum_inv)] = 0.
        deg_inv = np.diag(row_sum_inv)
        norm_feature = deg_inv.dot(feature)
        norm_feature = np.array(norm_feature, dtype=np.float32)

    return norm_feature

def load_citation_data(dataset_name):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)

    if dataset_name == 'corar' or dataset_name == 'citeseerr' or dataset_name == 'ogbn-arxiv':
        feature = np.genfromtxt(os.path.join(dataset_path, 'features.csv'))
        n_feat = feature.shape[1]
    else:
        feature = sp.load_npz(os.path.join(dataset_path, 'features.npz'))
        feature = feature.tocsc()
        n_feat = feature.shape[1]
    feature = feature.astype(dtype=np.float32)

    adj = sp.load_npz(os.path.join(dataset_path, 'adj.npz'))
    adj = adj.tocsc()

    label = np.genfromtxt(os.path.join(dataset_path, 'labels.csv'))
    if dataset_name == 'corar':
        n_class = 7
    elif dataset_name == 'citeseerr':
        n_class = 6
    elif dataset_name == 'pubmed':
        n_class = 3
    elif dataset_name == 'ogbn-arxiv':
        n_class = 40

    idx_train = np.genfromtxt(os.path.join(dataset_path, 'idx_train.csv'))
    idx_valid = np.genfromtxt(os.path.join(dataset_path, 'idx_valid.csv'))
    idx_test = np.genfromtxt(os.path.join(dataset_path, 'idx_test.csv'))

    return feature, adj, label, idx_train, idx_valid, idx_test, n_feat, n_class

def load_webkb_data(dataset_name):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)

    feature = sp.load_npz(os.path.join(dataset_path, 'features.npz'))
    feature = feature.tocsc()
    n_feat = feature.shape[1]
    feature = norm_feat(feature)
    feature = feature.astype(dtype=np.float32)

    adj = sp.load_npz(os.path.join(dataset_path, 'adj.npz'))
    adj = adj.tocsc()

    label = np.genfromtxt(os.path.join(dataset_path, 'labels.csv'))
    n_class = 5

    idx_train = np.genfromtxt(os.path.join(dataset_path, 'idx_train.csv'))
    idx_valid = np.genfromtxt(os.path.join(dataset_path, 'idx_valid.csv'))
    idx_test = np.genfromtxt(os.path.join(dataset_path, 'idx_test.csv'))

    return feature, adj, label, idx_train, idx_valid, idx_test, n_feat, n_class
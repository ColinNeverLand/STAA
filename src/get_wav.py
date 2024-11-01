import numpy as np
import pygsp as pg
import scipy.sparse as sp
import torch
from wavelet.app_wav_coeff import app_wav_coeff
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygsp")

def getWavCoeffs(Ft, At, nScales, signal_type):
    device = At.device
    At = At.coalesce()
    
    adjacency_matrix = sp.coo_matrix((At.values().cpu().numpy(), (At.indices().cpu().numpy())), shape=At.size())
    
    epsilon = 1e-8
    self_loops = sp.eye(adjacency_matrix.shape[0], format='csr') * epsilon
    adjacency_matrix = adjacency_matrix + self_loops
    
    adjacency_matrix = adjacency_matrix.maximum(adjacency_matrix.T)

    graph = pg.graphs.Graph(adjacency_matrix)
    graph.compute_laplacian("normalized")
    graph.estimate_lmax()

    if signal_type == 'feature':
        min_vals = torch.min(Ft, dim=0).values
        max_vals = torch.max(Ft, dim=0).values
        node_features = (Ft - min_vals) / (max_vals - min_vals + 1e-6)
        center = torch.mean(node_features, axis=0)
        euclidean_distances = torch.sqrt(torch.sum((node_features - center) ** 2, axis=1))
        signal = euclidean_distances.cpu().numpy()
    elif signal_type == 'digital':
        signal = At.to_dense().sum(dim=1).cpu().numpy()
    else:
        raise ValueError("Invalid signal_type!")

    laplacian_matrix = graph.L.toarray()
    M = 40

    coefficients, _ = app_wav_coeff(signal, laplacian_matrix, nScales, M)
    
    # get abs of high freq coeffs
    low_freq_coeffs = np.sum(coefficients[:nScales//2], axis=0)
    high_freq_coeffs = np.sum(np.abs(coefficients[nScales//2:]), axis=0)

    return torch.from_numpy(low_freq_coeffs).to(device), torch.from_numpy(high_freq_coeffs).to(device), signal
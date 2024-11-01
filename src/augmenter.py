import torch, dgl
from tqdm.autonotebook import tqdm
from loguru import logger
import utils
import os, multiprocessing
import numpy as np

class STAA():
    def __init__(
        self,
        restart_probs,
        time_travel_probs,
        eps=1e-3,
        K=100,
        symmetric_trick=True,
        device='cuda',
        dense=False,
        verbose=False
    ):
        self.restart_probs = restart_probs
        self.time_travel_probs = time_travel_probs
        self.eps = eps
        self.K = K
        self.symmetric_trick = symmetric_trick
        self.device = device
        self.dense = dense
        self.verbose = verbose

    def __call__(self, dataset):
        N = dataset[0].num_nodes()
        for graph in dataset:
            assert N == graph.num_nodes()

        if self.dense:
            I = torch.eye(N, device=self.device)
        else:
            I = utils.sparse_eye(N, self.device)

        Xt_1 = I
        new_graphs = list()

        for index, graph in enumerate(tqdm(dataset, desc='augmentation'), start=0):
            At = graph.adj(ctx=graph.device)
            if self.dense:
                At = At.to_dense()
            At = At + I
            At = self.normalize(At, ord='row')
            restart_probs = self.restart_probs[index]
            time_travel_probs = self.time_travel_probs[index]
            restart_diag, random_walk_diag, time_travel_diag = self.diag(restart_probs, time_travel_probs)
            
            inv_Ht = self.approx_inv_Ht(At, random_walk_diag, time_travel_diag, self.K)

            try:                
                Xt = torch.matmul(torch.matmul(restart_diag.float(), inv_Ht.float()), I.float()) + \
                     torch.matmul(torch.matmul(time_travel_diag.float(), inv_Ht).float(), Xt_1.float())
            except:
                Xt = (torch.matmul(torch.matmul(restart_diag.float().to('cpu'), inv_Ht.float().to('cpu')), I.float().to('cpu')) + \
                      torch.matmul(torch.matmul(time_travel_diag.float().to('cpu'), inv_Ht.to('cpu')).float(), Xt_1.float().to('cpu'))).to(self.device)

            Xt = self.normalize(Xt, ord='col')
            Xt = self.filter_matrix(Xt, self.eps)
            Xt_1 = Xt

            if self.symmetric_trick:
                Xt = (Xt + Xt.transpose(1, 0)) / 2

            if self.dense:
                A = Xt.to_sparse().transpose(0, 1).coalesce()
            else:
                A = Xt.transpose(1, 0).coalesce()

            if self.symmetric_trick:
                ones = torch.ones(A._nnz(), device=self.device)
                A = torch.sparse_coo_tensor(A.indices(), ones, A.shape, device=self.device)
                A = self.normalize(A, ord='sym', dense=False)
            else:
                A = self.normalize(A, ord='row', dense=False)

            new_graph = utils.weighted_adjacency_to_graph(A)
            new_graphs.append(new_graph)

            if self.verbose:
                logger.info('number of edge in this time step: {}'.format(new_graph.num_edges()))

        return new_graphs

    def diag(self, restart_probs, time_travel_probs):
        restart_probs = restart_probs.to(self.device)
        time_travel_probs = time_travel_probs.to(self.device)
        
        random_walk_probs = 1 - restart_probs - time_travel_probs

        indices = torch.arange(len(restart_probs), device=self.device)
        indices = torch.stack([indices, indices])

        restart_diag = torch.sparse_coo_tensor(
            indices=indices,
            values=restart_probs,
            size=(len(restart_probs), len(restart_probs)),
            device=self.device
        )

        random_walk_diag = torch.sparse_coo_tensor(
            indices=indices,
            values=random_walk_probs,
            size=(len(random_walk_probs), len(random_walk_probs)),
            device=self.device
        )

        time_travel_diag = torch.sparse_coo_tensor(
            indices=indices,
            values=time_travel_probs,
            size=(len(time_travel_probs), len(time_travel_probs)),
            device=self.device
        )

        return restart_diag, random_walk_diag, time_travel_diag



    def row_sum(self, A, dense=None):
        if dense is None:
            dense = self.dense
        if dense:
            return A.sum(dim=1)
        else:
            return torch.sparse.sum(A.to(self.device), dim=1).to_dense()

    def normalize(self, A, ord='row', dense=None):
        if dense is None:
            dense = self.dense
        N = A.shape[0]
        A = A if ord == 'row' else A.transpose(0, 1)
        norm = self.row_sum(A, dense=dense)
        norm[norm<=0] = 1
        if ord == 'sym':
            norm = norm ** 0.5

        if dense:
            inv_D = torch.diag(1 / norm)
        else:
            inv_D = utils.sparse_diag(1 / norm)

        if ord == 'sym':
            nA = inv_D @ A @ inv_D
        else:
            nA = inv_D @ A
        return nA if ord == 'row' else nA.transpose(0, 1)

    def approx_inv_Ht(self, A, restart_prob, time_travel_prob, K=10):
        if self.dense:
            I = torch.eye(A.shape[0], device=self.device)
        else:
            I = utils.sparse_eye(A.shape[0], self.device)

        # restart_prob and time_travel_prob are sparse diagonal matrices
        # We need to calculate c as 1 minus the sum of restart_prob and time_travel_prob
        ones = torch.ones(A.shape[0], device=self.device)
        restart_diag = restart_prob.coalesce().values().to(self.device)
        time_travel_diag = time_travel_prob.coalesce().values().to(self.device)
        c_diag = ones - (restart_diag + time_travel_diag)
        c = torch.sparse_coo_tensor(indices=torch.stack([torch.arange(A.shape[0]), torch.arange(A.shape[0])]), 
                                    values=c_diag, 
                                    size=A.shape, 
                                    device=self.device)

        # Calculate ATc
        ATc = A.transpose(0, 1) @ c.float()

        inv_H_k = I
        for i in range(K):
            try:
                inv_H_k = I + ATc @ inv_H_k
            except:
                inv_H_k = (I.to('cpu') + ATc.to('cpu') @ inv_H_k.to('cpu')).to(self.device)

        return inv_H_k

    def filter_matrix(self, X, eps):
        assert eps < 1.0
        if self.dense:
            mask = X < eps
            if mask.all():
                print('All nums in X < eps')
                return torch.eye(X.size(0))
            else:
                X[mask] = 0.0
                return self.normalize(X, ord='col')
        else:
            X_filter = utils.sparse_filter(X, eps)
            if torch.nonzero(X_filter.to_dense()).size(0) == 0:
                print('All nums in sparse X < eps')
                return torch.eye(X.size(0)).to_sparse()
        return self.normalize(X_filter, ord='col')

class Merge():
    def __init__(self, device):
        self.device = device

    def __call__(self, dataset):
        merged_graphs = [dataset[0]]

        for graph in dataset[1:]:
            merged_graph = dgl.merge([merged_graphs[-1], graph])
            merged_graph = merged_graph.cpu().to_simple().to(self.device)
            del merged_graph.edata['count']
            merged_graphs.append(merged_graph)

        return GCNNorm(self.device)(merged_graphs)

class GCNNorm():
    def __init__(self, device):
        self.device = device

    def __call__(self, dataset):
        normalized = [utils.graph_to_normalized_adjacency(graph) for graph in dataset]
        return [utils.weighted_adjacency_to_graph(adj) for adj in normalized]

class TGAC():
    def __init__(self, device):
        # Initialize the device for the TGAC model
        self.device = device
        # Set alpha for temporal weighting (adjust as necessary)
        self.alpha = 0.1
        # Set the pruning ratio (adjust as necessary)
        self.pruning_ratio = 0.9

    def centrality_func(self, graph):
        # Degree centrality for nodes is the number of edges incident to each node
        deg = graph.in_degrees().float()  # Use float for subsequent calculations
        return deg.to(self.device)

    def compute_edge_centrality(self, graph):
        # Calculate the edge centrality using the method provided in the image
        deg = self.centrality_func(graph)
        u, v = graph.edges()

        # edge_centrality = deg[u] + self.alpha * graph.edata['time']
        edge_centrality = deg[u] + self.alpha
        return edge_centrality

    def prune_edges(self, edge_centrality, num_edges):
        # Prune edges based on centrality scores, keep top k percentage
        _, indices_to_keep = torch.topk(edge_centrality, int(num_edges * self.pruning_ratio))
        return indices_to_keep

    def __call__(self, dataset):
        new_graphs = []
        for graph in tqdm(dataset, desc='Augmentation'):
            graph = graph.to(self.device)
            # Add self-loops because degree centrality considers them
            graph = dgl.add_self_loop(graph)
            edge_centrality = self.compute_edge_centrality(graph)
            indices_to_keep = self.prune_edges(edge_centrality, graph.number_of_edges())
            
            # Create a new graph using the edges that have been kept
            src_nodes, dst_nodes = graph.edges()
            src_nodes = src_nodes[indices_to_keep]
            dst_nodes = dst_nodes[indices_to_keep]
            new_graph = dgl.graph((src_nodes, dst_nodes), num_nodes=graph.number_of_nodes(), device=self.device)
            
            
            # Set the weights for the edges in the new graph
            new_graph.edata['w'] = edge_centrality[indices_to_keep]
            new_graph = dgl.add_self_loop(new_graph)
            
            # Append the new DGL graph to the list of new graphs
            new_graphs.append(new_graph)

        return new_graphs
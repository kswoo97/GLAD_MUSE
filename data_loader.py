import pickle
import torch
import numpy as np
import argparse
import warnings
import torch_geometric

from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from torch_geometric.utils.subgraph import subgraph
from torch_geometric.utils import to_networkx, to_dense_adj
from tqdm import tqdm

def load_torch_dataset(data, device) : 
    
    diff = "easy"
    
    if data == "mutag" : 
        dataset = TUDataset(root='/tmp/ENZYMES', name='Mutagenicity', use_node_attr = True, 
                           transform = torch_geometric.transforms.remove_isolated_nodes.RemoveIsolatedNodes())
        BSize = 256 # Batch size
        
    elif data == "mmp" : 
        dataset = TUDataset(root='/tmp/ENZYMES', name='Tox21_MMP_training', use_node_attr = True, 
                           transform = torch_geometric.transforms.remove_isolated_nodes.RemoveIsolatedNodes())
        
        CD = [] # Remove graphs without edges
        for ie, d in enumerate(dataset) : 
            if d.edge_index.shape[1] >= 2 : 
                CD.append(ie)
                
        dataset = dataset[CD]
        BSize = 256 # Batch size
        
    elif data == "er" : 
        dataset = TUDataset(root='/tmp/ENZYMES', name='Tox21_ER_training', use_node_attr = True, 
                           transform = torch_geometric.transforms.remove_isolated_nodes.RemoveIsolatedNodes())
        
        CD = [] # Remove graphs without edges
        for ie, d in enumerate(dataset): 
            if d.edge_index.shape[1] >= 2 : 
                CD.append(ie)
                
        dataset = dataset[CD]
        BSize = 256 # Batch size
        
    elif data == "protein" :
        dataset = TUDataset(root='/tmp/ENZYMES', name='PROTEINS', use_node_attr = True)
        BSize = 256
        
    elif data == "nci1" :
        dataset = TUDataset(root='/tmp/ENZYMES', name='NCI1', use_node_attr = True)
        BSize = 256
        
    elif data == "dhfr" :
        dataset = TUDataset(root='/tmp/ENZYMES', name='DHFR', use_node_attr = True)
        BSize = 256
        
    elif data == "bzr" :
        dataset = TUDataset(root='/tmp/ENZYMES', name='BZR', use_node_attr = True)
        BSize = 256
        
    elif data == "aids" :
        dataset = TUDataset(root='/tmp/ENZYMES', name='AIDS', use_node_attr = True,
                            transform = torch_geometric.transforms.remove_isolated_nodes.RemoveIsolatedNodes())
        BSize = 256
        
    elif data == "dd" :
        dataset = TUDataset(root='/tmp/ENZYMES', name='DD', use_node_attr = True)
        BSize = 128
        
    elif data == "reddit" :
        
        dataset = TUDataset(root='/tmp/ENZYMES', name='REDDIT-BINARY', use_node_attr = True)
        total_degree = set()
        X_bucket = None
        
        for i in range(2000) : 
            EE1, EE2 = torch.unique(dataset[i].edge_index, return_counts = True)
            EE1 = EE1.numpy()
            EE2 = EE2.numpy()

            for v1, v2 in zip(EE1, EE2) : 
                total_degree.add(int(v2/2))
                
        dataset = TUDataset(root='/tmp/ENZYMES', name='REDDIT-BINARY', use_node_attr = True, 
                   transform = torch_geometric.transforms.OneHotDegree(max_degree = max(total_degree)))
        BSize = 32
        
    elif data == "imdb" :
        dtype = 'bio'
        X_bucket = None
        dataset = TUDataset(root='/tmp/ENZYMES', name='IMDB-BINARY', use_node_attr = True)
        total_degree = set()
        
        for i in range(1000) : 
            EE1, EE2 = torch.unique(dataset[i].edge_index, return_counts = True)
            EE1 = EE1.numpy()
            EE2 = EE2.numpy()

            for v1, v2 in zip(EE1, EE2) : 
                total_degree.add(int(v2/2))
        
        dataset = TUDataset(root='/tmp/ENZYMES', name='IMDB-BINARY', use_node_attr = True, 
                   transform = torch_geometric.transforms.OneHotDegree(max_degree = max(total_degree)))
        BSize = 128
        
    else : 
        raise TypeError("Check data name")
        
    return dataset, BSize

def create_train_valid_test(dataset, difficulty = 'easy', anom_type = 0) : 
    
    np.random.seed(0)
    ratio = 1.0
    
    Y = []
    for d in dataset : 
        Y.append(int(d.y.item()))

    if anom_type == 0 : 
        normD = np.where(np.array(Y) == 1)[0]
        orig_anomD = np.where(np.array(Y) == 0)[0]

    elif anom_type == 1 : 
        normD = np.where(np.array(Y) == 0)[0]
        orig_anomD = np.where(np.array(Y) == 1)[0]


    splits = []
    n_normal_train = int(normD.shape[0] * 0.8)
    n_normal_valid = int(normD.shape[0] * 0.1)
    n_normal_test = normD.shape[0] - n_normal_train - n_normal_valid
    
    anomD = np.random.choice(orig_anomD, int(orig_anomD.shape[0] * 0.1), replace = False) # This is just for number of anomalies

    n_anom_train = int(anomD.shape[0] * 0.0)
    n_anom_valid = int(anomD.shape[0] * 0.5)

    n_anom_test = anomD.shape[0] - n_anom_train - n_anom_valid
    
    if isinstance(difficulty, float) : 
            
        n_noisy = int(n_normal_train * difficulty)
        
        print("Anomalies are mixed up for: {0}".format(n_noisy))

    for i in range(5) : 

        np.random.seed(i)
        anomD = np.random.choice(orig_anomD, int(orig_anomD.shape[0] * 0.1), replace = False)
        
        np.random.shuffle(normD)
        np.random.shuffle(anomD)
            
        to_be_added = []
            
        if isinstance(difficulty, float) : 
            
            anom_candids = np.array(list(set(orig_anomD) - set(anomD)))
            
            if anom_candids.shape[0] <= n_noisy : 
                
                n_noisy = anom_candids.shape[0]
                
                to_be_added = list(np.random.choice(a = anom_candids, size =  n_noisy, replace = False))

        valid_graphs = []
        test_graphs = []

        train_graphs = list(normD[:int(n_normal_train * ratio)])
        
        if isinstance(difficulty, float) : # Mix noise if difficulty is not 0.0
            
            train_graphs = train_graphs + to_be_added
            
        valid_graphs = (list(normD[n_normal_train:n_normal_train+n_normal_valid]) + list(anomD[n_anom_train:n_anom_train+n_anom_valid]), [1] * n_normal_valid + [0] * n_anom_valid)
        test_graphs = (list(normD[n_normal_train+n_normal_valid:]) + list(anomD[n_anom_train+n_anom_valid:]), [1] * n_normal_test + [0] * n_anom_test)

        splits.append((train_graphs, valid_graphs, test_graphs))
        
    return splits

def prepare_data(dataset, device) : 
    
    labels = []
    labels_pos_weights = []
    labels_norms = []
    
    for d in dataset : 
        adj = to_dense_adj(d.edge_index)
        idd = list(range(adj.shape[0]))
        adj[idd, idd] = 1.0
        adj = adj.flatten().to(device)
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)        
        labels.append(adj)
        labels_pos_weights.append(pos_weight)
        labels_norms.append(norm)
        
    return labels, labels_pos_weights, labels_norms
import pickle
import torch
import numpy as np
import argparse
import warnings
import torch_geometric

from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from tqdm import tqdm

from GNNs import *
from src import *
from data_loader import *

if __name__ == "__main__" : 
    
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser('Proposed Method.')
    
    parser.add_argument('-data', '--data', type=str, default='mutag')
    parser.add_argument('-device', '--device', type=str, default='cuda:0')
    parser.add_argument('-anom_type', '--anom_type', type=int, default=0)
    parser.add_argument('-diff', '--diff', type=float, default=0.0)
    parser.add_argument('-dim', '--dim', type=int, default=32)
    parser.add_argument('-n_layers', '--n_layers', type=int, default=3)
    parser.add_argument('-lr', '--lr', type=float, default=0.001)
    parser.add_argument('-gamma', '--gamma', type=float, default=1.0)
    
    """
    data: Data name.
    device: GPU device.
    anom_type: Anomaly class index. 0 indicates class 0 graphs are anomalous and class 1 graph is normal.
    diff: Training contamination ratio. 0.0 indicates no contamination
    dim: GNN hidden dimension.
    n_layers: GNN number of layers.
    lr: Learning rate of GNN.
    gamma: Weighting scalar on 1-entry of adjacency matrix.
    """
    
    args = parser.parse_args()
    
    data = args.data
    device = args.device
    anom_type = args.anom_type
    diff = args.diff
    lr = args.lr
    dim = args.dim
    n_layer = args.n_layers
    dp = 0.3
    w_decay = 1e-6
    gamma = args.gamma
    
    if diff == 0.0 : # This is without training dataset contamination / Otherwise contaminate training set
        diff = "easy" 
    
    dataset, BSize = load_torch_dataset(data, device) # Torch geometric dataset and batch size
    S = create_train_valid_test(dataset, diff, anom_type) # Training/validation/test splits
    adj, adj_pos_weights = prepare_data(dataset, device, gamma) # Prepare adjacency matrices
    n_feat = dataset[0].x.shape[1]
    
    total_results = dict()
                    
    adj, adj_pos_weights = prepare_data(dataset, device, gamma) # Prepare adjacency matrices

    ## Total result tables

    ValidAUROC = np.zeros((5, 10)) 
    ValidAP = np.zeros((5, 10))
    ValidTopK = np.zeros((5, 10))

    TestAUROC = np.zeros((5, 10))
    TestAP = np.zeros((5, 10))
    TestTopK = np.zeros((5, 10))

    for K in range(5) : ## Simulations

        torch.manual_seed(K)
        torch.random.manual_seed(K)
        np.random.seed(K)

        ## There are no training labels, since we are training an unsupervised anomaly detector.

        train_idxs = S[K][0]
        valid_idxs = S[K][1][0]
        valid_labels = S[K][1][1]
        test_idxs = S[K][2][0]
        test_labels = S[K][2][1]

        ## Neural networks

        model = GIN(num_features = n_feat, num_classes = 1, hidden_units=dim, num_layers=n_layer, dropout = dp,
                                 mlp_layers=2, train_eps=False).to(device)

        edge_encoder = MLP_Decoder(int(n_layer * dim), dim, dim).to(device)
        feature_encoder = MLP_Decoder(int(n_layer * dim), dim, n_feat).to(device)

        gnn_trainer = MUSE_representation_learning(datasets = dataset, device = device, labels = adj,
                                                    labels_pos_weights = adj_pos_weights)

        ## Representation Learning of MUSE
        loss, trained_parameters = gnn_trainer.train(model = model, feature_head = feature_encoder, edge_head = edge_encoder, saving_interval = 20,
                                                 train_idxs = train_idxs, lr = lr, weight_decay = w_decay, 
                                                 epochs = 200, batch_size = BSize, 
                                                 return_loss = True, seed = K)

        for p_iter in range(len(trained_parameters)) : 

            ## One class classification of MUSE
            curP = trained_parameters[p_iter]
            ae_trainer = MUSE_oneclass_classification(model = model, feature_encoder = feature_encoder, 
                                                               edge_encoder = edge_encoder,            
                                                               datasets = dataset, device = device, B_size = BSize, 
                                                              labels = adj, pos_weights = adj_pos_weights)

            N_FEAT = 4

            valid_acc1 = 0
            test_acc1 = 0

            valid_acc2 = 0
            test_acc2 = 0

            valid_acc3 = 0
            test_acc3 = 0

            ## Hyperparameters of One-class Classifier

            for r4_dim in [32, 64, 128] : 

                for r4_lr in [1e-2, 1e-3, 1e-4] : 

                    torch.manual_seed(K)
                    torch.random.manual_seed(K)
                    np.random.seed(K)

                    MLP_autoencoder = AutoEncoderOneclassClassifier(in_dim = N_FEAT, hid_dim = r4_dim, 
                                                                   n_layers = 3, drop_p = 0.0).to(device)

                    p_valid_auroc, p_valid_ap, p_valid_K, p_test_auroc, p_test_ap, p_test_K = ae_trainer.train_MLP(
                                                encoder_param = curP, classifier = MLP_autoencoder, train_idxs = train_idxs, valid_idxs = valid_idxs, valid_labels = valid_labels, 
                                                  test_idxs = test_idxs, test_labels = test_labels, 
                                                  lr = r4_lr, epochs = 500, saving_interval = 1, w_decay = 1e-4, seed = K)

                    if p_valid_auroc > valid_acc1 : 
                        valid_acc1 = p_valid_auroc
                        test_acc1 = p_test_auroc

                    if p_valid_ap > valid_acc2 : 
                        valid_acc2 = p_valid_ap
                        test_acc2 = p_test_ap

                    if p_valid_K > valid_acc3 : 
                        valid_acc3 = p_valid_K
                        test_acc3 = p_test_K

            ValidAUROC[K, p_iter] = valid_acc1
            ValidAP[K, p_iter] = valid_acc2
            ValidTopK[K, p_iter] = valid_acc3
            TestAUROC[K, p_iter] = test_acc1
            TestAP[K, p_iter] = test_acc2
            TestTopK[K, p_iter] = test_acc3

    # Pick the best validation epochs
    avg_valid1 = np.argmax(np.mean(ValidAUROC, 0))
    avg_valid2 = np.argmax(np.mean(ValidAP, 0))
    avg_valid3 = np.argmax(np.mean(ValidTopK, 0))

    ## Test performance
    print("===============")
    print("LR: {0} | Hid-Dim: {1} | Layers: {2} | Gamma: {3}".format(lr, dim, n_layer, gamma))
    print("Data: {0}".format(data))
    print("Average of AUROC: {0} | STD of AUROC: {1}".format(np.mean(TestAUROC[:, avg_valid1]), np.std(TestAUROC[:, avg_valid1])))
    print("Average of AP: {0} | STD of AP: {1}".format(np.mean(TestAP[:, avg_valid2]), np.std(TestAP[:, avg_valid2])))
    print("Average of TopK: {0} | STD of TopK: {1}".format(np.mean(TestTopK[:, avg_valid3]), np.std(TestTopK[:, avg_valid3])))
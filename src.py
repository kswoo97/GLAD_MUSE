from tqdm import tqdm

import torch
import copy
import numpy as np
import networkx as nx
import torch_geometric

from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge
from torch.utils.data import Sampler
from torch_geometric.utils import to_networkx, to_dense_adj
from torch_geometric.loader import DataLoader
import torch.nn as nn

import torch.nn.functional as F

from sklearn.metrics import average_precision_score as ap_score
from sklearn.metrics import roc_auc_score as auroc

class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
    
class EarlyStopper :
    
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = 0.0

    def early_stop(self, validation_loss, classifier):
        if validation_loss >= self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.params = copy.deepcopy(classifier.state_dict())
        elif validation_loss < (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return (True, self.params)
        return (False, self.params)

class MUSE_representation_learning() : 
    
    def __init__(self, datasets, device, labels, labels_pos_weights) : 
        
        self.datasets = datasets
        self.device = device
        self.labels = labels
        self.labels_pos_weights = labels_pos_weights
        self.cos = torch.nn.CosineSimilarity(dim = 1, eps = 1e-6)
            
    def fit(self, IDXs) : 
        
        self.optimizer.zero_grad()
            
        D_loader = DataLoader(self.datasets, batch_size = len(IDXs), sampler = SubsetSampler(IDXs))
        D = next(iter(D_loader)).to(self.device)
        new_edge_index, edge_id = dropout_edge(D.edge_index, force_undirected=True, p = 0.5)
        D.edge_index = new_edge_index
            
        X_copy = copy.deepcopy(D.x)
        
        Z = self.model(D) # Node embeddings
        Z_X = self.feature_head(Z)
        Z_E = self.edge_head(Z)
        L_X = 1 - self.cos(X_copy, Z_X) # Feature reconstruction via cosine similarity
        curL = 0.0
        
        TL1 = 0.0
        TL2 = 0.0
        
        for b_id, idx in zip(IDXs, range(D.ptr.shape[0] - 1)) : 
            start_indptr = D.ptr[idx]
            end_indptr = D.ptr[idx + 1]
            curZ = Z_E[start_indptr : end_indptr, :]
            A_tilde = torch.matmul(curZ, curZ.T).flatten()
            
            pos_weight = self.labels_pos_weights[b_id]
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight, reduction = 'mean')
            L1 = 0.5 * criterion(A_tilde, self.labels[b_id])
            L2 = 0.5 * torch.mean(L_X[start_indptr : end_indptr])
            
            curL += L1
            curL += L2
            
            TL1 += L1
            TL2 += L2
            
        curL /= len(IDXs)
        curL.backward()
        self.optimizer.step()
        
        TL1 = TL1.detach().cpu().item()
        TL2 = TL2.detach().cpu().item()
        
        return TL1, TL2
        
    def train(self, model, feature_head, edge_head, train_idxs, lr = 1e-3, weight_decay = 1e-6, epochs = 200, 
              saving_interval = 20, batch_size = 50, return_loss = True, fixed_epochs = False, seed = 0) : 
        
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        np.random.seed(seed)
    
        parameters = []        
        self.model = model
        self.feature_head = feature_head
        self.edge_head = edge_head
        self.optimizer = torch.optim.Adam(list(model.parameters()) + list(self.feature_head.parameters()) + list(self.edge_head.parameters()), 
                                          lr = lr, weight_decay = weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=0)

        train_graphs = np.array(train_idxs)
        total_loss = [[], []]
            
        for ep in tqdm(range(epochs)) : 

            np.random.shuffle(train_graphs)
            TL1 = 0
            TL2 = 0

            for idx in range(0, train_graphs.shape[0], batch_size) :

                self.model.train()
                self.feature_head.train()
                self.edge_head.train()

                curB = train_graphs[idx: idx + batch_size]
                loss1, loss2 = self.fit(curB)
                TL1 += loss1
                TL2 += loss2

            if return_loss : 
                total_loss[0].append(TL1/len(train_idxs))
                total_loss[1].append(TL2/len(train_idxs))

            self.scheduler.step()
            
            if int(ep + 1) % saving_interval == 0 : 

                parameters.append([copy.deepcopy(self.model.state_dict()), 
                                          copy.deepcopy(self.feature_head.state_dict()), 
                                          copy.deepcopy(self.edge_head.state_dict())])

        if return_loss : 
            return total_loss, parameters
        else : 
            return parameters
    
class MUSE_oneclass_classification() : 
    
    def __init__(self, model, feature_encoder, edge_encoder, datasets, 
                 device, labels, pos_weights,
                 B_size = 30, with_feature = True) : 
        
        self.datasets = datasets
        self.device = device
        self.B_size = B_size
        self.model = model
        self.feature_encoder = feature_encoder
        self.edge_encoder = edge_encoder
        self.labels = labels
        self.labels_pos_weights = pos_weights
        
        self.TX = []
            
    def obtain_error_representations(self, parameter, train_idx) : 
        
        self.TX = []
        TX = []
        IDXs = list(np.arange(len(self.datasets)))
        B_size = self.B_size
        nG = len(IDXs)        
        
        with torch.no_grad() : 

            self.model.load_state_dict(parameter[0])
            self.feature_encoder.load_state_dict(parameter[1])
            self.edge_encoder.load_state_dict(parameter[2])

            self.model.eval()
            self.feature_encoder.eval()
            self.edge_encoder.eval()

            criterion1 = torch.nn.BCEWithLogitsLoss(reduction = 'none')
            criterion2 = torch.nn.CosineSimilarity(dim = 1, eps = 1e-6)
            
            self.curTX = []
            self.curTE = []
            
            
            for idx in range(0, nG, B_size) :

                curB = IDXs[idx: idx + B_size]

                D_loader = DataLoader(self.datasets, batch_size = len(curB), sampler=SubsetSampler(curB))
                D = next(iter(D_loader)).to(self.device)
                GOAL = copy.deepcopy(D.x)
                Z = self.model(D).detach() # Node embeddings
                Z1 = self.feature_encoder(Z) # Feature related term
                Z2 = self.edge_encoder(Z) # Edge related term
                X_loss = (1 - criterion2(Z1, GOAL)).cpu()
                
                
                for b_id, idx in zip(curB, range(D.ptr.shape[0] - 1)) : 
                    
                    start_indptr = D.ptr[idx]
                    end_indptr = D.ptr[idx + 1]
                    curZ = Z2[start_indptr : end_indptr, :]
                    A_torch = torch.matmul(curZ, curZ.T).flatten()
                    
                    TL = criterion1(A_torch, self.labels[b_id]).cpu().flatten()
                    avg_L1, std_L1 = TL.mean().item(), TL.std().item()

                    avg_X = torch.mean(X_loss[start_indptr : end_indptr]).item()
                    std_X = torch.std(X_loss[start_indptr : end_indptr]).item()

                    curX = torch.tensor([avg_L1, std_L1, avg_X, std_X], 
                                        dtype = torch.float32).to(self.device) # 4 dimsnional vectors
                    self.TX.append(curX)
                        
                del Z, D, GOAL

        self.TX = torch.stack(self.TX).to(self.device)
    
    def train_MLP(self, encoder_param, classifier, train_idxs, valid_idxs, valid_labels, 
                      test_idxs, test_labels, lr = 1e-3, epochs = 200, w_decay = 1e-5, saving_interval = 1, 
                                    return_parameter = False, early_stop = 10, seed = 0) : 
        
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        
        if len(self.TX) == 0 :
            self.obtain_error_representations(encoder_param, train_idxs)
            
        else :
            None
            
        early_stopper = EarlyStopper(patience=early_stop, min_delta=0.0) # Original: 10
        
        optimizer = torch.optim.Adam(classifier.parameters(), lr = lr, weight_decay = w_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0)
        self.cos_criterion = torch.nn.CosineSimilarity(dim = 1, eps = 1e-6)
        torch.manual_seed(0) # Training reproducibility
        
        classifier.train()
        val_auroc = 0
            
        for ep in range(epochs) : 

            classifier.train()
            optimizer.zero_grad()

            curZ = classifier(self.TX[train_idxs, :])
            L2_loss = torch.mean(torch.sqrt(torch.sum((curZ - self.TX[train_idxs])**2, 1)))
            L2_loss.backward()
            optimizer.step()
            scheduler.step()

            if int(ep + 1) % saving_interval == 0 : 

                stds = self.return_variance(idxs = train_idxs, classifier = classifier)

                cur_val = self.evaluate_with_Reconstruction(valid_idxs, valid_labels, classifier, stds)

                cur_res, cur_param = early_stopper.early_stop(cur_val, classifier)

                if cur_res : 

                    break
                        
        classifier.load_state_dict(cur_param)
        stds = self.return_variance(idxs = train_idxs, classifier = classifier)
        val_auroc, val_ap, val_K = self.evaluate_with_Reconstruction(valid_idxs, valid_labels, classifier, stds, final = True)
        test_auroc, test_ap, test_K = self.evaluate_with_Reconstruction(test_idxs, test_labels, classifier, stds, final = True)
        
        
        return val_auroc, val_ap, val_K, test_auroc, test_ap, test_K
    
    def evaluate_with_Reconstruction(self, idxs, label, classifier, stds, final = False) : 
        
        with torch.no_grad() : 
            classifier.eval()
            curZ = classifier(self.TX[idxs])
            score = -(torch.sum(((curZ - self.TX[idxs])**2)/stds, 1)).detach().to('cpu').numpy()
            
        if final : 
            pred = np.argsort(score)[:10]
            tmp_K = 1 - np.mean(np.array(label)[pred])
            return auroc(label, score), ap_score(label, score), tmp_K
        else : 
            return auroc(label, score)
    
    def return_variance(self, idxs, classifier) : 
        
        with torch.no_grad() : 
            classifier.eval()
            curZ = classifier(self.TX[idxs])
            stds = torch.std(curZ, 0) + 1e-6
            
        return stds

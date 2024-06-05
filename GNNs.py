import torch
from torch.nn import functional as func
import torch.nn.functional as F
from torch import nn
from torch_geometric import nn as gnn
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool,global_max_pool, SAGEConv
from torch_geometric.utils import softmax
import torch_scatter

class MLP(nn.Module):
    def __init__(self, num_features, num_classes, hidden_units=32, num_layers=1, bias_term = True):
        super(MLP, self).__init__()
        if num_layers == 1:
            self.layers = nn.Linear(num_features, num_classes, bias = bias_term)
        elif num_layers > 1:
            layers = [nn.Linear(num_features, hidden_units, bias = bias_term),
                      #nn.BatchNorm1d(hidden_units),
                      nn.ReLU()]
            for _ in range(num_layers - 2):
                layers.extend([nn.Linear(hidden_units, hidden_units, bias = bias_term),
                               #nn.BatchNorm1d(hidden_units),
                               nn.ReLU()])
            layers.append(nn.Linear(hidden_units, num_classes, bias = bias_term))
            self.layers = nn.Sequential(*layers)
        else:
            raise ValueError()

    def forward(self, x):
        return self.layers(x)

class GIN(nn.Module):
    def __init__(self, num_features, num_classes, hidden_units=32, decoder_out_dim = 128,num_layers=3, dropout=0.15,
                 mlp_layers=2, train_eps=False, is_encoder = True):
        super(GIN, self).__init__()
        convs, bns = [], []
        for i in range(num_layers):
            input_dim = num_features if i == 0 else hidden_units
            if is_encoder : 
                hidden_dim = hidden_units
            else : 
                hidden_dim = hidden_units if i != num_layers - 1 else decoder_out_dim
            convs.append(gnn.GINConv(MLP(input_dim, hidden_dim, hidden_dim, mlp_layers),
                                     train_eps=train_eps))
            bns.append(nn.BatchNorm1d(hidden_dim))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.num_layers = num_layers
        self.dropout = dropout
        self.dropout_layer = torch.nn.Dropout(p = self.dropout)
        self.is_encoder = is_encoder
        if self.is_encoder != True : # Add learnable mask parameters
            self.encoder_mask = torch.nn.Parameter(torch.zeros(decoder_out_dim))
            self.decoder_mask = torch.nn.Parameter(torch.zeros(int(num_features)))
        self.final_layers = torch.nn.Linear(hidden_dim, hidden_dim)
            

    def forward(self, data): # Do not consider batchwise training now
        
        x = data.x 
        edge_index = data.edge_index
        
        if self.is_encoder : # Use layerwise-concatenation
            h_list = [x] # Do not add its original feature
            for conv, bn in zip(self.convs, self.bns):
                h = conv(h_list[-1], edge_index)
                h = self.dropout_layer(h)
                h = torch.relu(h)
                h_list.append(h)
                #h_list.append(torch.relu(bn(h)))
            out = torch.cat(h_list[1:], 1)
            return out
        else : 
            for conv, bn in zip(self.convs, self.bns):
                x = conv(x, edge_index)
                x = self.dropout_layer(x)
                x = torch.relu(x)
            x = self.final_layers(x)
            return x

class MLP_Decoder(nn.Module) : 
    def __init__(self, in_dim, hid_dim, out_dim) :
        super(MLP_Decoder, self).__init__()
        
        self.lin1 = torch.nn.Linear(in_dim, hid_dim)
        self.lin2 = torch.nn.Linear(hid_dim, out_dim)
        
    def forward(self, x) : 
        
        x = torch.relu(self.lin1(x))
        x = self.lin2(x)
        
        return x
    
class AutoEncoderOneclassClassifier(nn.Module) : 
    def __init__(self, in_dim, hid_dim, n_layers, drop_p = 0.5, use_bias = True) :
        super(AutoEncoderOneclassClassifier, self).__init__()
    
        self.layers = torch.nn.ModuleList()
        self.drop_layer = torch.nn.Dropout(p = drop_p)
        self.layers.append(torch.nn.Linear(in_dim, hid_dim, bias = use_bias))
        for _ in range(n_layers - 2) : 
            self.layers.append(torch.nn.Linear(hid_dim, hid_dim, bias = use_bias))
        self.layers.append(torch.nn.Linear(hid_dim, in_dim, bias = use_bias))
        
    def forward(self, x) : 
        
        for ik, lay in enumerate(self.layers[:-1]) : 
            x = lay(x)
            x = torch.relu(x)
            x = self.drop_layer(x)
            
        x = self.layers[-1](x)

        return x
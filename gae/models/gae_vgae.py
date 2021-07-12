import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
    
class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, activation = F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.activation = activation
        
    def forward(self, features, adj):
        b, n, d = features.shape
        x = features
        x = torch.einsum('bnd,df->bnf',(x, self.weight))
        x = torch.bmm(adj, x)
        outputs = self.activation(x)
        return outputs
        

def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	return nn.Parameter(initial)

class gae(nn.Module):
    def __init__(self, feature_dim): #cfg/model kwargs=dict(feature_dim=256)
        super(gae, self).__init__()
        # self.bn0 = nn.BatchNorm1d(feature_dim, affine=False)
        self.base_gcn = GraphConvSparse(feature_dim, 128)
        self.gcn_mean = GraphConvSparse(128, 128, activation=lambda x:x)
        self.gcn_logstddev = GraphConvSparse(128, 128, activation=lambda x:x)
        
        # self.CrossEnt = F.binary_cross_entropy() #nn.CrossEntropyLoss() #loss_fn   

    def encode(self, x, A, one_hop_idxs):
        B, N, D = x.shape
        # print('org x:',x.shape)

        x = self.base_gcn(x, A)
        # print('1st x:',x.shape)

        n=x.size(-2)
        dout = x.size(-1)
        
        self.mean = self.gcn_mean(x, A).cuda()
        self.logstd = torch.exp(self.gcn_logstddev(x, A)).cuda()
        gaussian_noise = torch.randn(B, n, dout).cuda()
        
        # print('[shape] mean={} logstd={} gn={}'.format(self.mean.shape,self.logstd.shape,gaussian_noise.shape))
        
        sampled_z = self.logstd * gaussian_noise + self.mean
        # print('sampled_z shape:', sampled_z.shape)
        
        return sampled_z
    
    def decode(self, Z):
        # print('Z shape={}'.format(Z.shape))
        Zt=torch.transpose(Z,1,2)
        A_pred = torch.sigmoid(torch.matmul(Z,Zt)).cuda()
        return A_pred

    def forward(self, data, return_loss=False):
        x, A, one_hop_idxs, labels = data
        Z = self.encode(x, A, one_hop_idxs)
        A_pred = self.decode(Z)
        
        
        # k1 = one_hop_idxs.size(-1)
        # dout = x.size(-1)
        # edge_feat = torch.zeros(B, k1, dout).cuda()
        # for b in range(B):
        #     edge_feat[b, :, :] = x[b, one_hop_idxs[b]]
        
        # edge_labels = (center_label == one_hop_labels).astype(np.int64)
        
        if return_loss:
            loss = (1/A_pred.size(0))*F.binary_cross_entropy(A_pred.view(-1),A.view(-1))
            # print('crossent loss={}'.format(loss))
            kl_divergence=((0.5/A_pred.size(0))*(1+ 2*self.logstd - self.mean**2 - torch.exp(self.logstd)**2)).sum(1).mean()            
            loss-=kl_divergence
            # print('total loss={}, kl_div={}'.format(loss, kl_divergence))
            
            
            exit()
            return A_pred, loss
        else:
            assert("Error: calculation loss")


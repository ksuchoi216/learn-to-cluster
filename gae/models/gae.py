import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np


class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, features, A):
        x = torch.bmm(A, features)
        return x


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, agg, act=F.relu): #activation func modified.
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.FloatTensor(in_dim * 2, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        init.xavier_uniform_(self.weight)
        init.constant_(self.bias, 0)
        self.agg = agg()
        self.act = act #activation func

    def forward(self, features, A):
        b, n, d = features.shape
        assert (d == self.in_dim)
        agg_feats = self.agg(features, A)
        cat_feats = torch.cat([features, agg_feats], dim=2)
        out = torch.einsum('bnd,df->bnf', (cat_feats, self.weight))
        out = self.act(out + self.bias) #activation func modified
        return out
    
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
        # data normalization l2 -> bn
        B, N, D = x.shape
        print('org x:',x.shape)

        # BN
        # x = x.view(-1, D)
        # x = self.bn0(x)
        # x = x.view(B, N, D)

        x = self.base_gcn(x, A)
        print('1st x:',x.shape)

        # x = self.conv2(x, A)
        # print('2nd x:',x.shape)
        # k1 = one_hop_idxs.size(-1)
        n=x.size(-2)
        dout = x.size(-1)
        
        self.mean = self.gcn_mean(x, A).cuda()
        self.logstd = torch.exp(self.gcn_logstddev(x, A)).cuda()
        gaussian_noise = torch.randn(B, n, dout).cuda()
        
        print('[shape] mean={} logstd={} gn={}'.format(self.mean.shape,self.logstd.shape,gaussian_noise.shape))
        
        sampled_z = self.logstd * gaussian_noise + self.mean
        print('sampled_z shape:', sampled_z.shape)
        
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
        
        # print('A org shape:',A.shape)
        # print('A pred shape:',A_pred.shape)

        # print('A org e.g.',A[1])
        # print('A_pred e.g.',A_pred[1])
        
        if return_loss:
            loss = F.binary_cross_entropy(A_pred.view(-1),A.view(-1))
            print('crossent loss={}'.format(loss))
            kl_divergence=((0.5/A_pred.size(0))*(1+ 2*self.logstd - self.mean**2 - torch.exp(self.logstd)**2)).sum(1).mean()            
            loss-=kl_divergence
            print('total loss={}, kl_div={}'.format(loss, kl_divergence))
            
            return A_pred, loss
        
        else:
            assert("Error: calculation loss")


#----------------------------------------modification-----------------

# class MeanAggregator(nn.Module):
#     def __init__(self):
#         super(MeanAggregator, self).__init__()

#     def forward(self, features, A):
#         x = torch.bmm(A, features)
#         return x


# class GraphConv(nn.Module):
#     def __init__(self, in_dim, out_dim, agg):
#         super(GraphConv, self).__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.weight = nn.Parameter(torch.FloatTensor(in_dim * 2, out_dim))
#         self.bias = nn.Parameter(torch.FloatTensor(out_dim))
#         init.xavier_uniform_(self.weight)
#         init.constant_(self.bias, 0)
#         self.agg = agg()

#     def forward(self, features, A):
#         b, n, d = features.shape
#         assert (d == self.in_dim)
#         agg_feats = self.agg(features, A)
#         cat_feats = torch.cat([features, agg_feats], dim=2)
#         out = torch.einsum('bnd,df->bnf', (cat_feats, self.weight))
#         out = F.relu(out + self.bias)
#         return out    

# class gae(nn.Module):
#     def __init__(self, feature_dim): #cfg/model kwargs=dict(feature_dim=256)
#         super(gae, self).__init__()
#         self.bn0 = nn.BatchNorm1d(feature_dim, affine=False)
#         self.conv1 = GraphConv(feature_dim, 64, MeanAggregator)
#         self.conv2 = GraphConv(64, 64, MeanAggregator)
#         self.conv3 = GraphConv(64, 32, MeanAggregator)
#         self.conv4 = GraphConv(32, 32, MeanAggregator)

#         self.classifier = nn.Sequential(nn.Linear(32, 32), nn.PReLU(32),
#                                         nn.Linear(32, 2))
#         self.loss = nn.CrossEntropyLoss() #loss_fn

#     def extract(self, x, A, one_hop_idxs):
#         # data normalization l2 -> bn
#         B, N, D = x.shape
#         # xnorm = x.norm(2,2,keepdim=True) + 1e-8
#         # xnorm = xnorm.expand_as(x)
#         # x = x.div(xnorm)

#         x = x.view(-1, D)
#         x = self.bn0(x)
#         x = x.view(B, N, D)

#         x = self.conv1(x, A)
#         x = self.conv2(x, A)
#         x = self.conv3(x, A)
#         x = self.conv4(x, A)
#         k1 = one_hop_idxs.size(-1)
#         dout = x.size(-1)
#         edge_feat = torch.zeros(B, k1, dout).cuda()
#         for b in range(B):
#             edge_feat[b, :, :] = x[b, one_hop_idxs[b]]
#         edge_feat = edge_feat.view(-1, dout)
#         pred = self.classifier(edge_feat)

#         # shape: (B*k1, 2)
#         return pred

#     def forward(self, data, return_loss=False):
#         x, A, one_hop_idxs, labels = data
#         x = self.extract(x, A, one_hop_idxs)
#         if return_loss:
#             loss = self.loss(x, labels.view(-1))
#             return x, loss
#         else:
#             return x




        
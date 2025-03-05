import numpy as np
import torch
from tqdm import tqdm
from torch.nn.init import xavier_normal_
import time
from collections import defaultdict
from torch.optim.lr_scheduler import ExponentialLR
import argparse
from tqdm import tqdm
import random


class TuckER(torch.nn.Module):
    """
    TuckER model. for detatils see (https://arxiv.org/abs/1901.09590)

    Attributes
    ----------
    E - Entity Embeddings (n_e x d1)
    R - Relation Embeddings (n_r x d2)
    W - TuckER Core (d1 x d1 x d2)
    input_dropout - Dropout of Input layers
    hidden_dropout1 - Dropout after E x W x R
    hidden_dropout2 - Dropout after tensor flatten
    loss - Binary Cross Entrhropy
    bn0 - BatchNorm of Entity Embeddings
    bn1 - BatchNorm of E x W x R 
    """
    def __init__(self, d, d1, d2, **kwargs):
        """
        Initialization

        Parameters
        ----------
        d - data
        d1 - entity embeddings dimension
        d2 - relation embeddings dimension
        """      
        super(TuckER, self).__init__()

        self.E = torch.nn.Embedding(len(d.entities), d1)
        self.R = torch.nn.Embedding(len(d.relations), d2)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)),
                                    dtype=torch.float, device="cuda", requires_grad=True))

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)


    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, e1_idx, r_idx):
        """
        x = S(W, U, T, V) = [W; U, T, V] - TuckER score function

        Parameters
        ----------
        e1_idx - Indexes of entities in a batch
        r_idx - Indicies of relations in a batch
        """       
        e1 = self.E(e1_idx)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = self.R(r_idx)

        W_mat = torch.mm(r, self.W.view(r.size(1), -1))

        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1,0))
        pred = torch.sigmoid(x)
        return pred
    
    
#PITF model
class PITF(torch.nn.Module):
    """
    Pairwize Interaction Tenzor Factorization model
    
    Attributes
    ----------
    U - Entity Embeddings (n_e x d)
    V - Entity Embeddings (n_e x d)
    T - Relation Embeddings (n_r x d)
    loss - Binary Cross Entropy   
    n - number of entities
    """
    def __init__(self, data, d, ro1 = 1, ro2 = 1, **kwargs):
        super(PITF, self).__init__()
        """
        Initialization

        Parameters
        ----------
        data - data
        d - embedding dimension
        ro1 - varience for entity embeddings initizlization
        ro2 - variance for relation embeddings initizlization
        """     
        self.U = torch.nn.Embedding(len(data.entities), d)
        self.V = torch.nn.Embedding(len(data.entities), d)
        self.T = torch.nn.Embedding(len(data.relations), d)
        self.ro1 = ro1
        self.ro2 = ro2
        self.loss = torch.nn.BCELoss()
        self.n = len(data.entities)


    def init(self):
        self.U.weight.data.normal_(0, self.ro1)
        self.V.weight.data.normal_(0, self.ro1)
        self.T.weight.data.normal_(0, self.ro2)

    def forward(self, e1_idx, r_idx):
        """
        x = S(u, v, t) = uV^T + tv^T + ut^T - score function

        Parameters
        ----------
        e1_idx - Indexes of entities in a batch
        r_idx - Indicies of relations in a batch
        """       
        u = self.U(e1_idx)
        t = self.T(r_idx)
        nu = e1_idx.shape[0]
        nt = r_idx.shape[0]
        indets = torch.arange(nu)
        nv = self.n

        a = torch.mm(u, t.transpose(1, 0))
        a = a[indets, indets].unsqueeze(1).repeat(1, nv)
        b = torch.mm(u, self.V.weight.transpose(1, 0))
        c = torch.mm(t, self.V.weight.transpose(1, 0))
        x = a + b + c

        pred = torch.sigmoid(x)
        return pred



#Lorentzian model
class LPITF(torch.nn.Module):
    """
    Lorentzian Pairwize Interaction Tenzor Factorization model

    Attributes
    ----------
    U - Entity Embeddings (n_e x d)
    T - Relation Embeddings (n_r x d)
    ro1 - varience for U initizlization
    ro2 - variance for T initizlization
    loss - Binary Cross Entropy   
    n - number of entities
    input_dropout - Dropout of Input layers
    """
    def __init__(self, data, d, ro1 = 1, ro2 = 1, **kwargs):
        super(LPITF, self).__init__()
        """
        Initialization

        Parameters
        ----------
        data - data
        d - embedding dimension
        ro1 - varience for entity embeddings initizlization
        ro2 - variance for relation embeddings initizlization
        """       
        self.U = torch.nn.Embedding(len(data.entities), d)
        self.T = torch.nn.Embedding(len(data.relations), d)
        self.ro1 = ro1
        self.ro2 = ro2
        self.loss = torch.nn.BCELoss()
        self.n = len(data.entities)
        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])


    def init(self):
        self.U.weight.data.normal_(0, self.ro1)
        self.T.weight.data.normal_(0, self.ro2)

    def forward(self, e1_idx, r_idx):
        """
        x = S(u, v, t) = d(u, V) - d(u, t) - d(t, V) + d(0, t) - d(0, u) - d(0, V) 
            - hyperbolic score function
        
        Parameters
        ----------
        e1_idx - Indexes of entities in a batch
        r_idx - Indicies of relations in a batch
        """
        u = self.U(e1_idx)
        t = self.T(r_idx)
        u = self.input_dropout(u)
        t = self.input_dropout(t)
        V = self.input_dropout(self.U.weight)
        nu = e1_idx.shape[0]
        nt = r_idx.shape[0]
        indets = torch.arange(nu)
        nv = self.n

        a = torch.mm(u, t.transpose(1, 0))
        a = a[indets, indets].unsqueeze(1).repeat(1, nv)
        b = torch.mm(u, V.transpose(1, 0))
        c = torch.mm(t, V.transpose(1, 0))
        u0 = self.zero_count(u, self.U.weight)
        t0 = self.zero_count(t, self.U.weight)
        v0 = self.zero_count(self.U.weight, u).transpose(1, 0)
        x = -1 + u0 * t0 - a + 1 - u0 + 1 - t0 + b + c - u0 * v0 - u0 * t0 + 2 - 1 + v0  
        
        pred = torch.sigmoid(x)
        return pred  

    def zero_count(self, u, v):
        """
        Zero coordinate computing

        Parameters
        ----------
        u, v - vectors
        """
        u = torch.sum(u ** 2, dim = -1, keepdim = True)
        u = torch.sqrt(1 + u)
        n = v.shape[0]
        u = u.repeat(1, n)
        return u

    
#Mixed Geometry Tenzor Factorization model
class MIGTF(torch.nn.Module):
    """
    Mixed Geometry Tenzor Factorization

    Attributes
    ----------
    E - Euclidean Entity Embeddings (n_e x d1)
    R - Euclidean Relation Embeddings (n_r x d2)
    W - TuckER Core (d1 x d1 x d2)
    Eh - Hyperbolic Entity Embeddings (n_e x d3)
    Rh - Hyperbolic Relation Embeddings (n_r x d3)
    input_dropout - Dropout of Input layers
    hidden_dropout1 - Dropout after E x W x R
    hidden_dropout2 - Dropout after tensor flatten
    loss - Binary Cross Entrhropy
    bn0 - BatchNorm of Entity Embeddings
    bn1 - BatchNorm of E x W x R
    bn2 - BatchNorm for hyperbolic part
    ro1 - varience for Eh initizlization
    ro2 - variance for Rh initizlization   
    n - number of entities
    """
    def __init__(self, d, d1, d2, E_pr, R_pr, W_pr, d3 = 50,
                 c = 1.3, ro1 = 0.0001, ro2 = 0.0001, qr = False, **kwargs):
        """
        Initialization

        Parameters
        ----------
        d - Data
        d1 - Euclidean entity embeddings dimension
        d2 - Euclidean relation embeddings dimension
        E_pr - Pretrained Euclidean entity embeddings
        R_pr - Pretrained Euclidean relation embeddings
        W_pr - Pretarined TuckER core
        d3 - Hyperbolic embeddings dimension
        c - Curvature of Lorentz hyperbolic model
        qr - Apply QR decomposition to 
        """
        super(TuckER_T, self).__init__()
        self.E = torch.nn.Embedding.from_pretrained(E_pr.weight)
        self.E.weight.requires_grad=False
        self.R = torch.nn.Embedding.from_pretrained(R_pr.weight)
        self.E.weight.requires_grad=False
        self.W = torch.nn.Parameter(W_pr, requires_grad=False)

        self.Eh = torch.nn.Embedding(len(d.entities), d3)
        self.Rh  = torch.nn.Embedding(len(d.relations), d3)
        self.c = c
        self.qr = qr

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()
        self.ro1 = ro2
        self.ro2 = ro1
        self.n = len(d.entities)

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)
        self.bn2 = torch.nn.BatchNorm1d(d3)


    def init(self):
        self.Eh.weight.data.normal_(0, self.ro1)
        self.Rh.weight.data.normal_(0, self.ro2)


    def forward(self, e1_idx, r_idx):
        """
        x = S(W, U, T, V) = [W; U, T, V] - TuckER score function
        x1 = S(u, v, t) = d(u, V) - d(u, t) - d(t, V) + d(0, t) - d(0, u) - d(0, V)/ 
            (u0v0 + u0t0 + t0v0)
            - hyperbolic score function with normalization
        x_f = x + x1 - MIGTF score function

        Parameters
        ----------
        e1_idx - Indexes of entities in a batch
        r_idx - Indicies of relations in a batch
        """
        self.E.weight.requires_grad=False
        self.R.weight.requires_grad=False
        self.W.requires_grad = False
        self.Eh.weight.requires_grad=True
        self.Rh.weight.requires_grad=True

        e1 = self.E(e1_idx)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))
        r = self.R(r_idx)
        c = self.c

        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1,0))

        Q = self.Rh.weight

        if self.qr:
            U2 =  self.Rh.weight
            Q, R = torch.linalg.qr(torch.transpose(U2, 0, 1))
            Q = torch.transpose(Q, 0, 1)

        ut1 = self.Eh(e1_idx)
        ut2 = self.Rh(r_idx)
        if self.qr:
            ut2 = Q[r_idx]

        ut1 = self.bn2(ut1)
        ut2 = self.bn2(ut2)
        ut1 = self.input_dropout(ut1)
        ut2 = self.input_dropout(ut2)
        Ut3 = self.input_dropout(self.Eh.weight)
        nv = self.n
        
        a1 = torch.mm(ut1, Ut3.transpose(1, 0))
        b1 = torch.mm(ut2, Ut3.transpose(1, 0))
        c1 = torch.mm(ut1, ut2.transpose(1, 0))
        c1 = torch.diag(c1).unsqueeze(1).repeat(1, nv)
        u10 = self.zero_count(ut1, Ut3, c)
        u20 = self.zero_count(ut2, Ut3, c)
        u30 = self.zero_count(Ut3, ut1, c).transpose(1, 0)
            
        x1 = ((u10 * u20 - c1 - u10 - u20 + a1 + b1 - u10 * u30 - u20 * u30 + u30) + 2 * c) 
            / (u10 * u20 + u10 * u30 + u20 * u30)
        
        pred = torch.sigmoid(x + x1)
        return pred
            
    
    def zero_count(self, u, v, c = 1.3):
        """
        Zero coordinate computing

        Parameters
        ----------
        u, v - vectors
        c - curvature
        """
        u = torch.sum(u ** 2, dim = -1, keepdim = True)
        u = torch.sqrt(c + u)
        n = v.shape[0]
        u = u.repeat(1, n)
        return u

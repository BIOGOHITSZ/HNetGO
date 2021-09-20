#-*- encoding:utf8 -*-
#! /usr/bin/python

import os
import math
import pickle
import warnings
import argparse
import collections
import torch
import dgl
import pandas as pd
import numpy as np
import dgl.function as fn
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt

from tqdm import tqdm
from dgl import save_graphs, load_graphs
from dgl.heterograph import DGLHeteroGraph
from dgl.nn.functional import edge_softmax
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, average_precision_score
from tqdm import tqdm


warnings.filterwarnings("ignore")

VERBOSE = False
BASE_DIR = "../data/models/"
OUT_DIR = "../data/bak"
TEMP_DIR = "../data/temp"
FEATURE_DIMENSION = 1024

INPUT_DATASET = {
    "human": {
        "id": "9606",
        "nets-min": "9606-nets-min.pkl",
        "graphs-min": "9606-graphs-min.pkl"
    },
    "mouse": {
        "id": "10090",
        "nets-min": "10090-nets-min.pkl",
        "graphs-min": "9606-graphs-min.pkl"
    },
    
    # for ablation experiments
    "human-ppi": {
        "id": "9606",
        "nets-min": "9606-ppi-nets-min.pkl",
        "graphs-min": "9606-ppi-graphs-min.pkl"
    },
    "mouse-ppi": {
        "id": "10090",
        "nets-min": "10090-ppi-nets-min.pkl",
        "graphs-min": "9606-ppi-graphs-min.pkl"
    },
    "human-diamond": {
        "id": "9606",
        "nets-min": "9606-diamond-nets-min.pkl",
        "graphs-min": "9606-diamond-graphs-min.pkl"
    },
    "mouse-diamond": {
        "id": "10090",
        "nets-min": "10090-diamond-nets-min.pkl",
        "graphs-min": "10090-diamond-graphs-min.pkl"
    }
}

sub_ontologies = {
    "bp": "GO:0008150",
    "cc": "GO:0005575",
    "mf": "GO:0003674"
}


# Load and Construct Network
def load_graphs(species: str, output: bool = True) -> dict:
    
    graph_path = BASE_DIR + INPUT_DATASET[species]["graphs-min"]

    # load graphs
    with open(graph_path, "rb") as f:
        graphs = pickle.load(f)

    if output:
        print("load graphs:")
        for branch, cur_graphs in graphs.items():
            print("\n" + "-"*30, branch, "-"*30)
            for freq, graph in cur_graphs.items():
                print('\033[1;36m' + branch + ": " + freq + '\033[0m')
                print(graph)
    elif VERBOSE:
        print("load graphs:")
        for branch, cur_graphs in graphs.items():
            print("\n" + "-"*30, branch, "-"*30)
            for freq, graph in cur_graphs.items():
                print('\033[1;36m' + freq + '\033[0m', end = " ")
    return graphs


def split_k_fold(G: DGLHeteroGraph, fold: int):
    
    fold = fold if fold!=1 else 5
    G.nodes['protein'].data['train_mask'] = torch.zeros(G.num_nodes('protein'), dtype=torch.bool).bernoulli(1.0 / fold)
    G.nodes['protein'].data['test_mask'] = ~G.nodes['protein'].data['train_mask']


    
def build_labels(G: DGLHeteroGraph):
    
    # build labels
    annotated_by = G.edges(etype = 'annotated_by', form = 'all')
    labels = torch.zeros(G.number_of_nodes('protein'), G.number_of_nodes('term'), dtype=torch.int8)
    for i in tqdm(range(len(annotated_by[0]))):
        labels[annotated_by[0][i]][annotated_by[1][i]] = 1
    G.nodes['protein'].data['labels'] = labels
    return labels
    
    
    
def build_node_edge_features(G: DGLHeteroGraph, random: bool):
    
    node_dict = {}
    edge_dict = {}
    for ntype in G.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in G.etypes:
        edge_dict[etype] = len(edge_dict)
        G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * edge_dict[etype] 
    
    # Random initialize input feature
    for ntype in G.ntypes:
        emb = nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), FEATURE_DIMENSION), requires_grad = False)
        nn.init.xavier_uniform_(emb)
        G.nodes[ntype].data['inp'] = emb
        
    if not random:
        build_seq_features(G, True)
    return node_dict, edge_dict

    
def build_seq_features(G: DGLHeteroGraph, random: bool):
    
    with open(BASE_DIR + "9606-avg-emb.pkl", "rb") as f:
        seq2emd = pickle.load(f)
    
    with open(BASE_DIR + "9606-nets-min.pkl", "rb") as f:
        ppsn = pickle.load(f)
        
    id2emb = list(map(lambda x: seq2emd[x], ppsn['id2node']))
    G.nodes['protein'].data['emb'] = torch.Tensor(id2emb)
    G.nodes['protein'].data['inp'] = G.nodes['protein'].data['emb']
    return
    

def build_subgraph(G: DGLHeteroGraph):
    
    sub_edges = []
    masked = G.nodes['protein'].data['test_mask']
    annotated_by = G.edges(etype = 'annotated_by', form = 'all')
    for i in tqdm(range(len(annotated_by[2]))):
        if not masked[annotated_by[0][i]]:
            sub_edges.append(i)
            
    edges = {
        'annotated_by': sub_edges,
        'similar_with': G.edges(etype = 'similar_with', form = 'all')[2],
        'annotate': sub_edges,
        'son_of': G.edges(etype = 'son_of', form = 'all')[2]
    }
    
    g = dgl.edge_subgraph(G, edges, preserve_nodes=True)
    
    tag = True
    for protein_id in tqdm(range(len(masked))):
        if masked[protein_id]:
            if protein_id in g.edges(etype = 'annotated_by', form = 'all')[0]:
                print(protein_id)
                tag = False
    assert(tag)
    print("build sub_graph succ")

    return g


class HNetGOLayer(nn.Module):
    
    def __init__(self,
                 in_dim,
                 out_dim,
                 node_dict,
                 edge_dict,
                 n_heads,
                 dropout = 0.2,
                 use_norm = False):
        super(HNetGOLayer, self).__init__()

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.node_dict     = node_dict
        self.edge_dict     = edge_dict
        self.num_types     = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel     = self.num_types * self.num_relations * self.num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.att           = None

        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        self.use_norm    = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri   = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(self.num_types))
        self.drop           = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

        
    def forward(self, G, h):
        
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[etype]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata['k'] = k
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata['v'] = v

                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')

                sub_graph.edata['t'] = attn_score.unsqueeze(-1)

            G.multi_update_all({etype : (fn.u_mul_e('v', 't', 'm'), fn.sum('m', 't')) \
                                for etype in edge_dict}, cross_reducer = 'mean')

            new_h = {}
            for ntype in G.ntypes:
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                t = G.nodes[ntype].data['t'].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + h[ntype] * (1-alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h

        
class HNetGO(nn.Module):
    
    def __init__(self, G, node_dict, edge_dict, n_inp, n_hid, n_out, n_layers, n_heads, use_norm = True):
        
        super(HNetGO, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.gcs = nn.ModuleList()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        self.adapt_ws  = nn.ModuleList()
        for t in range(len(node_dict)):
            self.adapt_ws.append(nn.Linear(n_inp,  n_hid))
        for _ in range(n_layers):
            self.gcs.append(HNetGOLayer(n_hid, n_hid, node_dict, edge_dict, n_heads, use_norm = use_norm))
        self.out = nn.Linear(n_hid, n_out)

        
    def forward(self, G):
        h = {}
        for ntype in G.ntypes:
            n_id = self.node_dict[ntype]
            h[ntype] = F.gelu(self.adapt_ws[n_id](G.nodes[ntype].data['inp']))
        for i in range(self.n_layers):
            h = self.gcs[i](G, h)
#         return torch.sigmoid(self.out(h[out_key]))
        return h
    

# Link Prediction
class HeteroDotProductPredictor(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        self.f = nn.Linear(1024, 1024)
        
    def forward(self, graph, h, etype):
        
        # h contains the node representations for each node type computed from
        # the GNN defined in the previous section (Section 5.1).
        return torch.sigmoid(torch.mm(self.f(h['protein']), h['term'].t()))

    
def construct_negative_graph(graph, k, etype, device):
    
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.number_of_nodes(vtype), (len(src) * k,)).int().to(device)
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes})


class Model(nn.Module):
    
    def __init__(self, model):
        
        super().__init__()
        self.hnetgo = model
        self.pred = HeteroDotProductPredictor()
        
        
    def forward(self, g, neg_g, etype):
        
        global h
        h = self.hnetgo(g)
#         print(h)
        return self.pred(g, h, etype), self.pred(neg_g, h, etype)


def calculate_performance(actual, pred_prob, threshold=0.4, average='micro'):
    
    pred_lable = []
    for l in range(len(pred_prob)):
        eachline = (pred_prob[l].cpu().detach().numpy() > threshold).astype(np.int)
        eachline = eachline.tolist()
        pred_lable.append(eachline)
    f_score = f1_score(actual.cpu().detach().numpy(), np.array(pred_lable), average=average)
    recall = recall_score(actual.cpu().detach().numpy(), np.array(pred_lable), average=average)
    precision = precision_score(actual.cpu().detach().numpy(), np.array(pred_lable), average=average)
    fpr, tpr, th = roc_curve(actual.cpu().detach().numpy().flatten(),pred_prob.cpu().detach().numpy().flatten(), pos_label=1)
    auc_score = auc(fpr, tpr)
    aupr=cacul_aupr(actual.cpu().detach().numpy().flatten(),pred_prob.cpu().detach().numpy().flatten())
    return f_score, precision, recall, auc_score, aupr


def cacul_aupr(lables, pred):
    
    precision, recall, _thresholds = metrics.precision_recall_curve(lables, pred)
    aupr = metrics.auc(recall, precision)
    return aupr


def get_n_params(model):
    
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def train(model, G, batch, device):
    
    best_val_acc = 0
    best_test_acc = 0
    train_step = 0
    lossF = nn.BCELoss()
    fmax = 0
    p_r_curve = []
    
    labels = G.nodes['protein'].data['labels'].float()
    train_mask = G.nodes['protein'].data['train_mask']
    test_mask = G.nodes['protein'].data['test_mask']
    
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=10000, max_lr = 0.001)
    
    for epoch in np.arange(batch) + 1:
        print("Epoch: %d" % (epoch))
        model.train()
        print("train succ")
        negative_graph = construct_negative_graph(G, 1, ('protein', 'annotated_by', 'term'), device)
        global pos_score, logits, neg_score
        pos_score, neg_score = model(G, negative_graph, ('protein', 'annotated_by', 'term'))
        # The loss is computed only for labeled nodes.
        logits = pos_score
        loss = lossF(logits[train_mask], labels[train_mask])
#         loss = - torch.sum(torch.log(pos_score+0.01)) - torch.sum(torch.log(1-neg_score+0.01))
#         loss = (1 - pos_score + neg_score).clamp(min=0).mean()
        if epoch % 1 == 0:
            model.eval()
#             logits, _ = model(G, negative_graph, ('protein', 'annotated_by', 'term'))
            with torch.no_grad():
                f1, recall, prescision, auc_score, aupr = calculate_performance(labels[test_mask], logits[test_mask])
                p_r_curve.append([f1, recall, prescision, auc_score, aupr])
                if f1>fmax:
                    fmax = f1
                    torch.save(model.state_dict(), '{}Model_Bak_HNetGO_Link_{}_{}.pkl'.format(TEMP_DIR, f1, epoch))
#                     with open( "./prCurve_Model1", 'bw') as f:
#                         pickle.dump(p_r_curve, f)

                # pred   = logits.argmax(1).cpu()
                # train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
                # val_acc   = (pred[val_idx]   == labels[val_idx]).float().mean()
                # test_acc  = (pred[test_idx]  == labels[test_idx]).float().mean()
                # if best_val_acc < val_acc:
                #     best_val_acc = val_acc
                #     best_test_acc = test_acc
                print('Epoch: %d LR: %.5f Loss %.4f, f1: %.4f, racall: %.4f, prescision: %.4f, auc: %.4f, aupr: %.4f' % (
                    epoch,
                    optimizer.param_groups[0]['lr'], 
                    loss.item(),
                    f1, recall, prescision, auc_score, aupr
                ))
            
        model.train()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_step += 1
        scheduler.step(train_step)
        

def main():
    
    parser = create_arg_parser()
    args = parser.parse_args()
    
    VERBOSE = args.verbose
    FEATURE_DIMENSION = args.feature_dimension
    print_graph = args.print_graph
    species = args.species
    branches = args.branch.split(' ')
    frequencies = args.frequencies.split(' ')
    fold = args.fold
    random_seq_features = args.random_seq_features
    batch = args.batch
    device = args.device
    
    # load and build HNet
    print("*"*35 + species + "*"*35)
    graphs = load_graphs(species, print_graph)
    
    for branch in branches:
        for freq in frequencies:
            print("*"*35 + "handling " + "[" + branch + "]" + "[" + freq + "]" + "*"*35)
            G = graphs[branch][freq]
            res = []
            for i in range(0, fold):
                print("-"*35 + "handling fold " + str(fold) + "-"*35)
                split_k_fold(G, fold)
                masked = G.nodes['protein'].data['test_mask']
                labels = build_labels(G)
                node_dict, edge_dict = build_node_edge_features(G, random_seq_features)
                G, graph = build_subgraph(G), G
                device = torch.device(device)
                model = HNetGO(G,
                            node_dict, edge_dict,
                            n_inp=1024,
                            n_hid=1024,
                            n_out=labels.shape[1],
                            n_layers=2,
                            n_heads=4,
                            use_norm = True).to(device)
                model = Model(model).to(device)
                print('Training HNetGO with #param: %d' % (get_n_params(model)))
                G = G.to(device)
                train(model, G, batch, device)
                

def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""

    # Instantiate the parser
    parser = argparse.ArgumentParser(description=( 
            'embedder.py creates ELMo embeddings for a given text '+
            ' file containing sequence(s) in FASTA-format.') )
    
    parser.add_argument( '-v', '--verbose', action='store_true',
                    default=False,
                    help='Output some information while processing. Default: False')
    
    parser.add_argument( '--print_graph', action='store_true',
                    default=False,
                    help='Output detailed information of heterogeneous graph. Default: False')
    
    parser.add_argument( '-s', '--species',  type=str,
                    default='human',
                    help='Output detailed information of heterogeneous graph. Default: False')
    
    parser.add_argument( '-b', '--branch', type=str,
                    default='bp',
                    help='Output detailed information of heterogeneous graph. Default: False')
    
    parser.add_argument( '-f', '--frequencies',  type=str,
                    default='default',
                    help='Output detailed information of heterogeneous graph. Default: False')
    
    parser.add_argument( '-d', '--device',  type=str,
                    default='cuda:0',
                    help='Device usred for training process. Default: cuda:0')
    
    parser.add_argument( '--fold',  type=int,
                    default='5',
                    help='Output detailed information of heterogeneous graph. Default: False')
    
    parser.add_argument( '--batch',  type=int,
                    default='2000',
                    help='Train batch of HNetGO model. Default: 2000')
    
    parser.add_argument( '--feature_dimension',  type=int,
                    default='1024',
                    help='Input node feature dimension. Default: 1024')
    
    parser.add_argument( '--random_seq_features', action='store_true',
                    default=False,
                    help='Replace features extracted from the pretrained model with randomly generated vectors. Default: False')
    
    return parser


if __name__ == '__main__':
    main()
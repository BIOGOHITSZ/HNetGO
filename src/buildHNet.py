#-*- encoding:utf8 -*-
#! /usr/bin/python

import pandas as pd
import numpy as np
import os
import copy
import pickle
import argparse
import collections
import dgl
import dgl.function as fn
import dgl.nn as dglnn
import torch as th
import torch

from dgl import save_graphs, load_graphs
from dgl.heterograph import DGLHeteroGraph
from tqdm import tqdm
from pandas.core.frame import DataFrame, Series
from typing import Dict, Tuple, Sequence


BASE_DIR = "../data/build_graphs/"
OUT_DIR = "../data/models/"
TERM_FILE = "terms.pkl"

INPUT_DATASET = {
    "human": {
        "id": "9606",
        "ppsn-min": "9606-ppsn-min.csv",
        "uniprot-min": "9606-uniprot-min.csv"
    },
    "mouse": {
        "id": "10090",
        "ppsn-min": "10090-ppsn-min.csv",
        "uniprot-min": "10090-uniprot-min.csv"
    }
}

sub_ontologies = {
    "bp": "GO:0008150",
    "cc": "GO:0005575",
    "mf": "GO:0003674"
}


# Function definition for building graphs
def load_files(species: str) -> Tuple[DataFrame, DataFrame, dict]:
    
    # file path
    ppsn = BASE_DIR + INPUT_DATASET[species]['ppsn-min']
    uniprot = BASE_DIR + INPUT_DATASET[species]['uniprot-min']
    term_path = BASE_DIR + TERM_FILE

    # load files
    df_ppsn = pd.read_csv(ppsn, sep = '\t')
    df_uniprot = pd.read_csv(uniprot, sep = '\t')
    with open(term_path, "rb") as f:
        terms, namespace = pickle.load(f)
    
    return df_ppsn, df_uniprot, terms, namespace


def build_ppsn_net(df_uniprot: DataFrame, df_ppsn: DataFrame) -> dict:
    
    # node: 18560 edge:11154322 node_without_annotation: 687 node_with_annotation: 17873
    ppsn_net = {}
#     ppsn_net['node2id'] = dict(zip(df_uniprot['uniprot_id'], df_uniprot['id']))
    ppsn_net['id2node'] = df_uniprot['uniprot_id'].to_numpy()
    ppsn_net['edge_src'] = df_ppsn['protein1'].to_numpy()
    ppsn_net['edge_dst'] = df_ppsn['protein2'].to_numpy()
    ppsn_net['edge_score'] = df_ppsn['score'].to_numpy()
    ppsn_net['node_seq'] = df_uniprot['sequence'].to_numpy()

    # data validation
    assert(len(ppsn_net['edge_src']) == len(ppsn_net['edge_dst']))
    assert(len(ppsn_net['id2node']) == len(ppsn_net['node_seq']))

#     for i in range(len(ppsn_net['node2id'])):
#         assert(ppsn_net['node2id'][ppsn_net['id2node'][i]] == i)
    
    return ppsn_net


def propagate(annotation: set, terms: dict) -> set:
    """propagate annotations with True Path Rule
    
    Args:
        annotation: a annotation set of certain protein
        terms: adjacency list of gene ontology directed acyclic graph, for example
            key -> 'GO:0000001'
            value -> ['GO:0048308', 'GO:0048311']
    """
    while True:
        length=len(annotation)
        temp=[]
        for i in annotation:
            if i not in terms or terms[i] is None:
                continue
            temp.extend(terms[i])
        annotation.update(temp)
        if len(annotation)==length: # 本轮未更新标签
            return annotation


def resolve_annotations(df_uniprot: DataFrame, terms: dict) -> dict:
    
    # handle labels
    annotations = dict(zip(df_uniprot['id'], df_uniprot['labels']))

    cnt = 0
    for i in annotations:
        if not isinstance(annotations[i],float):
            annotations[i] = set(map(lambda x: x.strip(), annotations[i].split(';')))
            cnt += len(annotations[i])
    print("\nbefore propagate: ", cnt)
    cnt = 0
    for i in annotations:
        if not isinstance(annotations[i],float):
            annotations[i] = propagate(annotations[i], terms)
            cnt += len(annotations[i])
    print("after propagate: ", cnt)
    
    return annotations


def split_branches(annotations: dict, namespace: dict, terms: dict) -> dict:
    """split terms into three branches
    """

    print("-"*30, "Total proteins/terms", "-"*30)
    ##划分子空间，每个子空间是一个集合
    bp,mf,cc=set(),set(),set()
    for i in terms:
        if namespace[i]=='biological_process':
            bp.add(i)
        elif namespace[i]=='molecular_function':
            mf.add(i)
        elif namespace[i]=='cellular_component':
            cc.add(i)
            
    print("Total proteins:\t", len(annotations))
    print("Total terms:\t", len(terms))
    print("Total terms in bp branch:\t", len(bp))
    print("Total terms in mf branch:\t", len(mf))
    print("Total terms in cc branch:\t", len(cc))
    assert(len(bp) +
          len(mf) +
          len(cc) == len(terms))

    labels = copy.deepcopy(annotations)
    labels_with_go={}
    for i in labels:
        if not isinstance(labels[i],float):
            labels_with_go[i] = set()
            for j in labels[i]:
                if j in terms:
                    labels_with_go[i].add(j)
    len(labels),len(labels_with_go)### some items has no label are discarded

    #按照子本体分开
    label_bp,label_cc,label_mf=collections.defaultdict(list),collections.defaultdict(list),\
    collections.defaultdict(list)
    for i in labels_with_go:
        for j in labels_with_go[i]:
            if j in bp:
                label_bp[i].append(j)
            elif j in cc:
                label_cc[i].append(j)
            elif j in mf:
                label_mf[i].append(j)


    print("-"*30, "proteins/terms used", "-"*30)

    fre_counter = collections.Counter()
    edge_counter = 0
    for i in labels_with_go:
        fre_counter.update(labels_with_go[i])
        edge_counter += len(labels_with_go[i])
    print("full ontology: ", 
          "\n\tTotal proteins: ", len(labels_with_go),
          "\n\tTotal terms: ", len(fre_counter),
          "\n\tTotal edges: ", edge_counter)


    bp_counter=collections.Counter()
    edge_counter = 0
    for i in label_bp:
        bp_counter.update(label_bp[i])
        edge_counter += label_bp[i].__len__()
    print("bp: \n",
          "\n\tproteins: ", len(label_bp),
          "\n\tterms: ", len(bp_counter),
          "\n\tedges: ", edge_counter)


    cc_counter=collections.Counter()
    edge_counter = 0
    for i in label_cc:
        cc_counter.update(label_cc[i])
        edge_counter += label_cc[i].__len__()
    print("cc: \n",
          "\n\tproteins: ", len(label_cc),
          "\n\tterms: ", len(cc_counter),
          "\n\tedges: ", edge_counter)


    mf_counter=collections.Counter()
    edge_counter = 0
    for i in label_mf:
        mf_counter.update(label_mf[i])
        edge_counter += label_mf[i].__len__()
    print("mf: \n",
          "\n\tproteins: ", len(label_mf),
          "\n\tterms: ", len(mf_counter),
          "\n\tedges: ", edge_counter)

#     print("-"*30, "-"*len("proteins/terms used"), "-"*30)

    term_branches = {
        "full" : {
            "labels":  labels_with_go,
            "counter": fre_counter
        },
        "bp" : {
            "labels":  label_bp,
            "counter": bp_counter
        },
        "cc" :  {
            "labels":  label_cc,
            "counter": cc_counter
        },
        "mf" : {
            "labels":  label_mf,
            "counter": mf_counter
        }
    }

    # for branch in term_branches:
    #     print(branch + ": \n",
    #       "\n\tproteins: ", len(term_branches[branch]["labels"]),
    #       "\n\tterms: ", len(term_branches[branch]["counter"]))
    return term_branches


def build_single_net(id2term: list, annotations: dict, son_of: dict) -> tuple:
    """build term_network upon the given term set and son_of relations
       build annotation_network upon the given annotaions connectons
       
    Args:
        id2term: term list with the index is term_id and the value is term's gene ontology ID
        annotations: {'protein_id': ['term_GO_ID1', 'term_GO_ID2'...]}
        son_of: {'son_term_GO_ID': ['father_GO_ID1', 'father_GO_ID2']}
    """
    
#     print("*"*30, str(len(id2term)))
    term2id = {id2term[i]:i for i in range(len(id2term))}
    # term2id
    for term in term2id:
        assert(term == id2term[term2id[term]])

    # build annotation net
    protein2term_src = []
    protein2term_dst = []

    for protein_id, annotation in annotations.items():
        if isinstance(annotation, float):
            continue
        for term in annotation:
            if term in term2id:
                protein2term_src.append(protein_id)
                protein2term_dst.append(term2id[term])

    assert(len(protein2term_src) == len(protein2term_dst))
    assert(max(protein2term_dst) < len(term2id))

    # build term net
    son_of_src = []
    son_of_dst = []

    for son, son_id in term2id.items():
#         if (son is "GO:0050444"):
#             print(son, son_id)
#             global temp
#             temp = term2id
        fathers = son_of[son]
        if fathers is None:
            if son not in sub_ontologies.values():
                print(son)
            continue
        for father in fathers:
            if father in term2id:
                son_of_src.append(son_id)
                son_of_dst.append(term2id[father])
                
    assert(max(son_of_src) < len(term2id))
    assert(max(son_of_dst) < len(term2id))
    occurence = set(son_of_src).union(set(son_of_dst))
    for i in range(len(term2id)):
        if i not in occurence:
            print("Outlier error: ", id2term[i])
    
    net = {
        'term_net': {
            "id2node":  id2term,
            "edge_src": son_of_src,
            "edge_dst": son_of_dst
        },
        'annotation_net': {
            "edge_src": protein2term_src,
            "edge_dst": protein2term_dst
        }
    }
    
    return net


def build_term_and_annotation_nets(branch: dict, annotations: dict, son_of: dict, default: bool) -> dict:

    nets = {}
    if default:
        return nets
    sorted_branch = list(branch.items())
    sorted_branch.sort(key = lambda x: x[1], reverse = True)

    term_frequence_list = {}

    n = len(sorted_branch)
    span = int(n*0.05)
    for cnt in tqdm(range(1, 21)):
        end = span * cnt
        if cnt == 20:
            end = len(sorted_branch)
        cur_terms = [x[0] for x in sorted_branch[0: end]]
        term_frequence_list[str(0.05*cnt)[:4]] = cur_terms

#     for k, v in term_frequence_list.items():
#         print(k + ": " + str(len(v)))


    for freq, id2term in tqdm(term_frequence_list.items()):
        nets[freq] = build_single_net(id2term, annotations, son_of)
    # net = build_single_net(term_frequence_list['1.0'], annotations, son_of)
#     for freq in nets:
#         print(freq)
#         for net_name, net in nets[freq].items():
#             print(net_name + ": ")
#             for k, v in net.items():
#                 print("\t", k, len(v))
                
    return nets


def build_default_term_net(branches: dict, annotations: dict, son_of: dict) -> dict:
    
    sum_length = 0
    counters = {}
    counter_full = set()

    for k, v in branches.items():
        if k == "full":
            continue
        cnt = 100 if k!="bp" else 300
#         cnt = 25 if k!="bp" else 150
        counters[k] = set(k for k, v in v["counter"].items() if v>=cnt)
        counter_full = counter_full.union(counters[k])

        sum_length += len(counters[k])
#         print(k + ":", len(counters[k]))

#     print("full" + ":", len(counter_full))
    assert(len(counter_full) == sum_length)
    counters["full"] = counter_full

    default = {}
    for k, id2term in counters.items():
#         print(k)
        default[k] = build_single_net(list(id2term), annotations, son_of)
    
    return default


# graph building function
def build_network_from_path(ppi_path, term_path, graph_path='./temp_graph.bin', save=False):
    
    warnings.warn("build_network_from_path is deprecated, use build_network instead", DeprecationWarning)
    print("build network from sub_net...")
    with open(ppi_path, "rb") as f:
        ppi_net = pickle.load(f)

    with open(term_path, "rb") as f:
        term_net = pickle.load(f)

    ppi_net.keys()
    term_net.keys()

    # 无向
    interaction_src = np.concatenate([ppi_net['edge_src'], ppi_net['edge_dst']])
    interaction_dst = np.concatenate([ppi_net['edge_dst'], ppi_net['edge_src']])
    ppi_net['edge_score'] =  np.concatenate([ppi_net['edge_score'], ppi_net['edge_score']]) #更新权重

    # 有向
    is_a_src = term_net['edge_src']
    is_a_dst = term_net['edge_dst']

    # 无向，非对称关联
    annotated_by_src = term_net['protein2term_src']
    annotated_by_dst = term_net['protein2term_dst']
    annotate_src = term_net['protein2term_dst']
    annotate_dst = term_net['protein2term_src']


    hetero_graph = dgl.heterograph({
        ('protein', 'interaction', 'protein'): (interaction_src, interaction_dst),
        ('term', 'is_a', 'term'): (is_a_src, is_a_dst),
        ('protein', 'annotated_by', 'term'): (annotated_by_src, annotated_by_dst),
        ('term', 'annotate', 'protein'): (annotate_src, annotate_dst)})

    # 属性：数据集成
    hetero_graph.nodes['protein'].data['node_vertex_embedding'] = torch.from_numpy(ppi_net['node_vertex_embedding'])    #属性特征
#     hetero_graph.nodes['protein'].data['node_seq_embedding'] = torch.Tensor(ppi_net['node_seq_embedding'])
    hetero_graph.nodes['term'].data['feature'] = torch.randn(len(term_net['id2term']), 128)
    # hetero_graph.nodes['user'].data['label'] = torch.randint(0, n_user_classes, (n_users,)) #分类特征
    hetero_graph.edges['interaction'].data['weight'] = torch.from_numpy(ppi_net['edge_score'])
    # randomly generate training masks on user nodes and click edges
    hetero_graph.nodes['protein'].data['train_mask'] = torch.zeros(len(ppi_net['id2node']), dtype=torch.bool).bernoulli(0.7)   #节点分类
    hetero_graph.edges['annotated_by'].data['train_mask'] = torch.zeros(len(term_net['protein2term_dst']), dtype=torch.bool).bernoulli(0.7)#链接预测 
    
    # 序列化，减少网络预处理次数
    if save:
        save_graphs(graph_path, hetero_graph)
    
    hetero_graph = hetero_graph.to("cuda:1")
    return hetero_graph, ppi_net, term_net
    
    
def load_network_from_path(graph_path):
    
    warnings.warn("load_network_from_path is deprecated, use build_network instead", DeprecationWarning)
    print("load network from {} file...".format(graph_path))
    hetero_graph, _ = load_graphs(graph_path)
    hetero_graph = hetero_graph[0].to("cuda:1")
    return hetero_graph


def build_single_network(ppsn_net: dict, term_net: dict, annotation_net: dict, branch_name: str, namespace: dict) -> DGLHeteroGraph:
    
    # build dgl graph
    # 无向
    # interaction_src = np.concatenate([ppi_net['edge_src'], ppi_net['edge_dst']])
    # interaction_dst = np.concatenate([ppi_net['edge_dst'], ppi_net['edge_src']])
    interaction_src = ppsn_net['edge_src']
    interaction_dst = ppsn_net['edge_dst']

    # 有向
    son_of_src = term_net['edge_src']
    son_of_dst = term_net['edge_dst']

    # 无向，非对称关联
    annotated_by_src = annotation_net['edge_src']
    annotated_by_dst = annotation_net['edge_dst']
    annotate_src = annotation_net['edge_dst']
    annotate_dst = annotation_net['edge_src']


    hetero_graph = dgl.heterograph({
        ('protein', 'similar_with', 'protein'): (interaction_src, interaction_dst),
        ('term', 'son_of', 'term'): (son_of_src, son_of_dst),
        ('protein', 'annotated_by', 'term'): (annotated_by_src, annotated_by_dst),
        ('term', 'annotate', 'protein'): (annotate_src, annotate_dst)}, idtype=th.int32)
    
    # handle properties
    if branch_name == "full":
        id2term = term_net['id2node']
        branch_map = {
            'biological_process': [0 for i in range(len(id2term))],
            'molecular_function': [0 for i in range(len(id2term))],
            'cellular_component': [0 for i in range(len(id2term))]
        }
        
        for term_id in range(len(id2term)):
            if (branch_name != "full"):
                print("x"*30, branch_name)
            term = id2term[term_id]
            branch_map[namespace[term]][term_id] = 1
        
        cnt = 0
        for _, masks in branch_map.items():
            cnt += sum(masks)
        assert(cnt == len(id2term))
        
        hetero_graph.nodes['term'].data['bp_mask'] = torch.from_numpy(np.array(branch_map['biological_process']))
        hetero_graph.nodes['term'].data['mf_mask'] = torch.from_numpy(np.array(branch_map['molecular_function']))
        hetero_graph.nodes['term'].data['cc_mask'] = torch.from_numpy(np.array(branch_map['cellular_component']))
    
    return hetero_graph


def build_networks(ppsn_net: dict, term_nets: dict, namespace: dict) -> dict:
    # echo some base information of input nets
    print("ppsn_net: ")
    for k, v in ppsn_net.items():
        print("\t" + k + "->" + str(len(v)))

    print("term_nets: ")
    for branch, cur_nets in term_nets.items():
        print("\n" + "-"*30, branch, "-"*30)
        for freq in cur_nets:
            print(freq)
            for net_name, net in cur_nets[freq].items():
                print(net_name + ":\t", end = "")
                for k, v in net.items():
                    print(k + "->" + str(len(v)), " ", end = "")
                print()

    graphs = {}
    for branch_name, cur_nets in term_nets.items():
        cur_graphs = {}
        for freq, cur_net in tqdm(cur_nets.items()):
             cur_graphs[freq] = build_single_network(ppsn_net, cur_net['term_net'], cur_net['annotation_net'], branch_name, namespace)
        graphs[branch_name] = cur_graphs

    print("graphs: ")
    for branch, cur_graphs in graphs.items():
        print("\n" + "-"*30, branch, "-"*30)
        for freq, graph in cur_graphs.items():
            print('\033[1;36m' + branch + ": " + freq + '\033[0m')
            print(graph)
    return graphs


def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""

    # Instantiate the parser
    parser = argparse.ArgumentParser(description=( 
            'Build heterogeneous networks.') )
    
    # build protein protein similarity network, output as csv format
    parser.add_argument('-d', '--default', action='store_true', 
                    default=False,
                    help='Build default heterogeneous networks only. Default: False')
    
    parser.add_argument('-g', '--graph_mode', action='store_true', 
                    default=False,
                    help='Output heterogeneous graph. Default: False')
    
    parser.add_argument('-n', '--net_mode', action='store_true', 
                    default=False,
                    help='Output ppsn and term net. Default: False')
    
    parser.add_argument('--debug', action='store_true', 
                    default=False,
                    help='Debug mode. Default: False')
    return parser



def main():
    
    parser = create_arg_parser()
    args = parser.parse_args()
    default = args.default
    assert(args.net_mode or args.graph_mode)
    
    # check dir
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    
    for species in INPUT_DATASET:
        nets_out_path = OUT_DIR + INPUT_DATASET[species]['id'] + "-nets-min.pkl"
        graphs_out_path = OUT_DIR + INPUT_DATASET[species]['id'] + "-graphs-min.pkl"
        df_ppsn, df_uniprot, terms, term_namespace = load_files(species)

        print("\n" + "*"*35, species, "*"*35, ": ")
        df_ppsn.head()
        df_uniprot.head()
        print("terms:")
        list(terms.items())[0:10]

        # print("\nppsn_net:")
        ppsn_net = build_ppsn_net(df_uniprot, df_ppsn)
        # ppsn_net

        annotations = resolve_annotations(df_uniprot, terms)
        branches = split_branches(annotations, term_namespace, terms)

        branches.keys()
        print("-"*30, "build term and annotation nets", "-"*30)

        # build default net
        default_nets = build_default_term_net(branches, annotations, terms)
        # build freq nets
        term_nets = {}
        for branch_name, branch in branches.items():
            print("-"*30, branch_name, "-"*30)
            term_nets[branch_name] = build_term_and_annotation_nets(branch['counter'], annotations, terms, default)
            term_nets[branch_name]['default'] = default_nets[branch_name]

            # output statistical information of nets
    #         cur_nets = term_nets[branch_name]
    #         for freq in cur_nets:
    #             print(freq)
    #             for net_name, net in cur_nets[freq].items():
    #                 print(net_name + ": ")
    #                 for k, v in net.items():
    #                     print("\t", k, len(v))

        #svae nets
        if args.net_mode:
            with open(nets_out_path, "wb+") as f:
                pickle.dump(ppsn_net, f)

        if args.graph_mode:
            print("-"*30, "build term and annotation graphs", "-"*30)
            # build and save graphs
            graphs = build_networks(ppsn_net, term_nets, term_namespace)
            with open(graphs_out_path, "wb+") as f:
                pickle.dump(graphs, f)
        
        print("done")
        
        
if __name__ == '__main__':
    main()
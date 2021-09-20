#-*- encoding:utf8 -*-
#! /usr/bin/python

import os
import pickle
import collections
import argparse
import logging as log
import pandas as pd
import numpy as np

from tqdm import tqdm
from pandas.core.frame import DataFrame, Series
from typing import Dict, Tuple, Sequence

BASE_DIR = "../data/prepare_data/"
OUT_DIR = "../data/build_graphs/"
TEMP_DIR = "../data/temp/"
OBO_NAME = "../data/dataset/go-basic.obo"
RESERVE_PPI = False
ALPHA_FACTOR = 1500

INPUT_DATASET = {
    "human": {
        "id": "9606",
        "ppi": "9606-ppi.txt",
        "uniprot": "9606-uniprot.xlsx",
        "diamond": "9606-diamond.tsv"
    },
    "mouse": {
        "id": "10090",
        "ppi": "10090-ppi.txt",
        "uniprot": "10090-uniprot.xlsx",
        "diamond": "10090-diamond.tsv"
    }
}

# handle single type of input
def handle_uniprot(df_uniprot: DataFrame, reserve_ppi: bool=False) -> DataFrame:
    df_uniprot.columns = ['uniprot_id', 'string_id', 'sequence', 'labels']
    # 清洗 string_id 字段
    df_uniprot['string_id'] = df_uniprot['string_id'].map(lambda x: x.strip(';') if isinstance(x, str) else x)
    
    # 由于部分对比方法无法处理无PPI信息的数据，因此去除空字段
    # 注意本方法支持无PPI信息的数据，注释掉此句可以保留这些节点
    if not reserve_ppi:
        df_uniprot = df_uniprot.drop(df_uniprot[df_uniprot['string_id'].map(lambda x: False if isinstance(x, str) else True)].index).reset_index(drop=True)
    
    return df_uniprot


def handle_ppi(df_ppi: DataFrame, string2uniprot: dict, reserve_ppi: bool=False) -> DataFrame:
    # 映射函数
    get_uniprot = lambda x: string2uniprot.get(x, float('nan'))
    # 保留所有PPI数据时使用此函数，对于存在映射的，优先使用uniprotid，否则使用stringid
    get_uniprot_reserve_x = lambda x: string2uniprot.get(x, x)
    get_fun = get_uniprot_reserve_x if reserve_ppi else get_uniprot
    
    # 清洗 ppi 数据
    df_ppi['protein1'] = df_ppi['protein1'].map(lambda x: get_fun(x))
    df_ppi['protein2'] = df_ppi['protein2'].map(lambda x: get_fun(x))
    
    # df_ppi = df_ppi.drop(df_ppi[df_ppi['protein1'].map(lambda x: False if isinstance(x, str) else True)].index).reset_index(drop=True)
    # df_ppi = df_ppi.drop(df_ppi[df_ppi['protein2'].map(lambda x: False if isinstance(x, str) else True)].index).reset_index(drop=True)
    df_ppi = df_ppi.dropna().reset_index(drop=True)
    
    return df_ppi


def handle_diamond(df_diamond: DataFrame, uniprot_ids: set, reserve_ppi: bool=False) -> DataFrame:
    df_diamond = df_diamond[['protein1', 'protein2', 'bit_score']].copy(deep=True)
    # 字段映射
    df_diamond['protein1'] = df_diamond['protein1'].map(lambda x: x.split("|")[1].split("-")[0])
    df_diamond['protein2'] = df_diamond['protein2'].map(lambda x: x.split("|")[1].split("-")[0])
    
    # 字段清洗
    df_diamond['protein1'] = df_diamond['protein1'].map(lambda x: x if x in uniprot_ids else float('nan'))
    df_diamond['protein2'] = df_diamond['protein2'].map(lambda x: x if x in uniprot_ids else float('nan'))
    
    df_diamond = df_diamond.dropna().reset_index(drop=True)
    return df_diamond


def preprocessing_data(species: str="human") -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Read source file into memory 
    """
    uniprot = BASE_DIR + INPUT_DATASET[species]['uniprot']
    ppi = BASE_DIR + INPUT_DATASET[species]['ppi']
    diamond = BASE_DIR + INPUT_DATASET[species]['diamond']

    df_uniprot = pd.read_excel(uniprot)[['Entry', 'Cross-reference (STRING)', 'Sequence', 'Gene ontology IDs']]
    df_ppi = pd.read_csv(ppi, sep=' ')
    df_diamond = pd.read_csv(diamond, sep='\t', header=None, names=['protein1', 'protein2', 'sequence identity percentage', 'length', 'mismatches', 'gap openings', 'start1', 'end1', 'start2', 'end2', 'e-value', 'bit_score'])

    df_uniprot = handle_uniprot(df_uniprot, RESERVE_PPI)

    # 映射字典
    string2uniprot = dict(zip(df_uniprot['string_id'], df_uniprot['uniprot_id']))
    df_ppi = handle_ppi(df_ppi, string2uniprot, RESERVE_PPI)

    # 保留 id
    uniprot_ids  = set(df_uniprot['uniprot_id'])
    df_diamond = handle_diamond(df_diamond, uniprot_ids, RESERVE_PPI)
    return df_uniprot, df_ppi, df_diamond


def build_similar_matrix(df_ppi: DataFrame, uniprot2id: dict, n: int, out_prefix: str) -> str:
    
    cnt = 0
    path = TEMP_DIR + out_prefix + "-ppi-mat.pkl"
    
    if os.path.exists(path):
        return path
    similar_matrix = [[0.]*n for i in range(n)]

    for _, row in tqdm(df_ppi.iterrows(), total = len(df_ppi)):
        cnt += 1
        if (cnt % 1000000 == 0):
            print(row)
        x = uniprot2id[row['protein1']]
        y = uniprot2id[row['protein2']]
        if (similar_matrix[x][y] != 0):
            print(x, " ", y)
        similar_matrix[x][y] += row['combined_score']

    cnt
    with open(path, "wb+") as f:
        pickle.dump(similar_matrix, f)
        
    return path


def build_diamond(df_uniprot: DataFrame, df_diamond: DataFrame, uniprot2id: dict, path: str) -> list:
    with open(path, "rb") as f:
        similar_matrix = pickle.load(f)

    for protein1 in tqdm(df_uniprot['uniprot_id']):
    #     print(protein1)
        df = df_diamond[df_diamond['protein1']==protein1]
        x = uniprot2id[protein1]

        for score in df[df['protein2'] == protein1]['bit_score']:
            similar_matrix[x][x] = max(similar_matrix[x][x], score)

        for _, row in df.iterrows():
            y = uniprot2id[row['protein2']]
            if x != y:
                similar_matrix[x][y] = row['bit_score'] / similar_matrix[x][x] * ALPHA_FACTOR

        similar_matrix[x][x] = 0

#     with open("./9606-similar-mat.pkl", "wb+") as f:
#         pickle.dump(similar_matrix, f)
    return similar_matrix


def build_ppsn(df_uniprot: DataFrame, df_ppi: DataFrame, df_diamond: DataFrame, species: str="human") -> DataFrame:
    
    out_prefix = INPUT_DATASET[species]['id']
    df_uniprot = df_uniprot.reset_index().rename(columns={'index': 'id'})
    uniprot2id = dict(zip(df_uniprot['uniprot_id'], df_uniprot['id']))
    path = build_similar_matrix(df_ppi, uniprot2id, len(df_uniprot), out_prefix)
    similar_matrix = build_diamond(df_uniprot, df_diamond, uniprot2id, path)
    
    protein1 = []
    protein2 = []
    score = []
    n = len(df_uniprot)

    for i in tqdm(range(n)):
        for j in range(n):
            if similar_matrix[i][j] != 0:
                if i == j:
                    print("error")
                    break;
                protein1.append(i)
                protein2.append(j)
                score.append(similar_matrix[i][j])
                
    # protein-protein similarity networks
    ppsn = DataFrame({
        'protein1': protein1,
        'protein2': protein2,
        'score': score
    })
    
    # output result
    ppsn.to_csv(OUT_DIR + out_prefix + "-ppsn-min.csv", index=False, sep='\t')
    df_uniprot.to_csv(OUT_DIR + out_prefix +"-uniprot-min.csv", index=False, sep='\t')
    return ppsn


def resolve_terms():
    
    go_path = OBO_NAME
    out_path = OUT_DIR + "terms.pkl"
    gos = []
    global namespace
    namespace = collections.defaultdict(str)
    is_a = collections.defaultdict(list)
    part_of = collections.defaultdict(list)

    # 根据规则来提取go term ，并依据其之间的依赖关系构建图谱
    with open(go_path,'r')as f:
        for line in f:
            if '[Typedef]' in line:
                break
            if line[:5]=='id: G':                       # 构建 gos
                line=line.strip().split()
                gos.append(line[1])
            elif line[:4]=='is_a':                      # 构建 is_a 关系
                line=line.strip().split()
                is_a[gos[-1]].append(line[1])
            elif line[:4]=='rela' and 'part' in line:   # 构建 partof 关系
                line=line.strip().split()
                part_of[gos[-1]].append(line[2])
            elif line[:5]=='names':                     # 统计子本体
                line=line.strip().split()
                namespace[gos[-1]]=line[1]

    son_of = {
        "GO:0008150": None,
        "GO:0005575": None,
        "GO:0003674": None
    }

    son_of = {**son_of, **is_a}

    for i in part_of:
        son_of[i].extend(part_of[i])

    cross_ontology = 0
    for k in son_of:
        tag = False
        if son_of[k] is None:
            continue
        for j in son_of[k]:
            if namespace[k]!=namespace[j]:
    #             print("error: child->{}, father->{}".format(k, j))
                tag = True
        if tag:
            cross_ontology += 1
            
    with open(out_path, "wb+") as f:
        pickle.dump([son_of, namespace], f)
    
    print("Cross-ontology connection in prat_of: ", cross_ontology)
    print("Total terms: ", len(gos))         # 47210
    print("is_a relation: ", len(is_a))        # 44082
    print("part_of relation: ", len(part_of))     # 8295
    print(len(namespace))   # 47210
    print("output terms: ", len(son_of))
    
    
def main():
    
    log.info("start preprocessing...")
    
    # arguments
    parser = create_arg_parser()
    args = parser.parse_args()
    
    if args.build_ppsn:
        # check dir
        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR)

        # TODO: build ppsn which retains all information of ppi 
        for species in INPUT_DATASET.keys():
            print(species)
            df_uniprot, df_ppi, df_diamond = preprocessing_data(species)

            print("*"*35, species, "*"*35, ": ")
            df_uniprot.shape
            df_ppi.shape
            df_diamond.shape

            df_uniprot.head()
            df_ppi.head()
            df_diamond.head()

            ppsn = build_ppsn(df_uniprot, df_ppi, df_diamond, species)
            
    if args.resolve_terms:
        resolve_terms()
        
        
def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""

    # Instantiate the parser
    parser = argparse.ArgumentParser(description=( 
            'filter and preprocessing dataset.') )
    
    # build protein protein similarity network, output as csv format
    parser.add_argument('--build_ppsn', action='store_true', 
                    default=False,
                    help='Build protein protein similarity network, output as csv format. Default: False')
    
    # filter gene ontology terms and build Directed Acyclic Graph
    parser.add_argument('--resolve_terms', action='store_true', 
                    default=False,
                    help='Filter gene ontology terms and build Directed Acyclic Graph. Default: False')
    
    parser.add_argument('--debug', action='store_true', 
                    default=False,
                    help='Debug mode. Default: False')
    return parser
        
        
if __name__ == "__main__":
    main()
    
    
    
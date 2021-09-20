# HNetGO: protein function prediction via heterogeneous network transformer

# Abstract
Protein function annotation is one of the most important research topics for revealing the essence of life at molecular level in the post-genome era. Current research shows that integrating multi-source data can effectively improve the performance of protein function prediction models. However, the heavy reliance on complex feature engineering and model integration methods limits the development of existing methods. Besides, models based on deep learning only use labeled data in a certain dataset to extract sequence features, thus ignoring a large amount of existing unlabeled sequence data. Here, we propose an end-to-end protein function annotation model named HNetGO, which innovatively uses heterogeneous network to integrate protein sequence similarity and protein-protein interaction (PPI) network information, and combines the pre-training model to extract the semantic features of the protein sequence. In addition, we design an attention-based graph neural network model, which can effectively extract node-level features from heterogeneous networks, and predict protein function by measuring the similarity between protein nodes and Gene Ontology (GO) term nodes. Comparative experiments on the human dataset show that HNetGO achieves state-of-the-art performance on Cellular Component and Molecular Function branches.

# Dependencies
* The code was developed and tested using python 3.7.
* To install python dependencies run: `pip install -r requirements.txt`.
* Follow the [instructions](https://github.com/bbuchfink/diamond/wiki/2.-Installation) to install [diamond](https://github.com/bbuchfink/diamond) program.

# Data
* Protein protein interaction data: download human(9606) and mouse(10090) data from [STRING database](https://string-db.org/cgi/download?sessionId=inputgtsessionId).
* Sequence and annotation data: download  sequence dataset (**fasta format**, both canonical and canonical&isoform) and  reviewed annotation dataset (**excel xlsx format**, include Entry, Sequence, Gene ontology IDs and Cross-reference (STRING) columns) from [Swiss-Prot database](https://www.uniprot.org/uniprot/?query=reviewed:yes).
* Gene Ontology data: download the latest released gene ontology data from the [official website](http://purl.obolibrary.org/obo/go/go-basic.obo).

# Usage
## Download data
* `mkdir ./data/dataset` and extract dataset to this directory.
* unzip the dataset 
```bash
cd data/dataset
mkdir ../prepare_data

# human 9606
gunzip -c 9606.protein.links.v11.0.txt.gz > ../prepare_data/9606-ppi.txt
gunzip -c uniprot-filtered-organism__Homo+sapiens+\(Human\)+\[9606\]_+AND+review--.xlsx.gz > ../prepare_data/9606-uniprot.xlsx

# mouse 10090
gunzip -c 10090.protein.links.v11.0.txt.gz > ../prepare_data/10090-ppi.txt
gunzip -c uniprot-filtered-organism__Mus+musculus+\(Mouse\)+\[10090\]_+AND+revie--.xlsx.gz > ../prepare_data/10090-uniprot.xlsx

# fasta data
gunzip -c uniprot-filtered-organism__Homo+sapiens+\(Human\)+\[9606\]_+AND+review--isoform.fasta.gz > ../prepare_data/9606.fa
gunzip -c uniprot-filtered-organism__Mus+musculus+\(Mouse\)+\[10090\]_+AND+revie--isoform.fasta.gz > ../prepare_data/10090.fa
gunzip -c uniprot-filtered-organism__Homo+sapiens+\(Human\)+\[9606\]_+AND+review--.fasta.gz > ../prepare_data/9606-query.fa
gunzip -c uniprot-filtered-organism__Mus+musculus+\(Mouse\)+\[10090\]_+AND+revie--.fasta.gz > ../prepare_data/10090-query.fa
```
## Preprocessing

* Multiple sequence alignment via Diamond

```bash
cd data/prepare_data

# generate diamond database 
diamond makedb --in ./9606.fa  -d 9606 && rm ./9606.fa
diamond makedb --in ./10090.fa -d 10090 && rm ./10090.fa

# MSA
diamond blastp -q ./9606-query.fa -d 9606.dmnd  -o 9606-diamond.tsv --very-sensitive -b8 -c1
diamond blastp -q ./10090-query.fa -d 10090.dmnd  -o 10090-diamond.tsv --very-sensitive -b8 -c1
```

* Extract protein-level sequence features through SeqVec

```bash
mkdir ./data/models
./src/seq2vec.py  -i ./data/prepare_data/9606-query.fa -o ./data/models/9606-avg-emb.pkl --split_char " " --protein  -g 0
./src/seq2vec.py  -i ./data/prepare_data/10090-query.fa -o ./data/models/9606-avg-emb.pkl --split_char " " --protein -g 0

# using CPU to compute embeddings
./src/seq2vec.py  -i ./data/prepare_data/9606-query.fa -o ./data/models/9606-avg-emb.pkl --split_char " " --protein  --cpu
./src/seq2vec.py  -i ./data/prepare_data/10090-query.fa -o ./data/models/9606-avg-emb.pkl --split_char " " --protein --cpu
```

​	In addition, we provide pretrained sequence features in the data directory:

```bash
mkdir ./data/models
mv ./data/*.pkl /data/models
```


* Data Preprocessing
```bash
cd ./src
python ./preprocessing.py --resolve_terms --build_ppsn
```

* Construct heterogeneous information network

```bash
buildHNet.py --default --graph_mode --net_mode
```

## Train and Evaluation

```bash
HNetGO.py -s human -b bp -f default
```


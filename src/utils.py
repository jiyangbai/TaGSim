"""Data processing utilities."""
import json
import math
import random
import time
import scipy.stats as stats
from scipy.stats import kendalltau
import torch
import pickle
from glob import glob
import networkx as nx
import numpy as np
from os.path import basename
from sklearn.preprocessing import OneHotEncoder
from texttable import Texttable

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def process_pair(path):
    """
    Reading a json file with a pair of graphs.
    :param path: Path to a JSON file.
    :return data: Dictionary with data.
    """
    data = json.load(open(path))
    return data

def calculate_loss(prediction, target):
    """
    Calculating the squared loss on the normalized GED.
    :param prediction: Predicted log value of GED.
    :param target: Factual log transofmed GED.
    :return score: Squared error.
    """
    prediction = -math.log(prediction)
    target = -math.log(target)
    score = (prediction-target)**2
    return score

def calculate_normalized_ged(data):
    """
    Calculating the normalized GED for a pair of graphs.
    """
    graph1, graph2 = data['graph_pair'][0], data['graph_pair'][1]
    norm_ged = data["ged"]/(0.5*(len(data["labels_1"])+len(data["labels_2"])))
    norm_ged = (data['ged']['nc'] + data['ged']['in'] + data['ged']['ie']) / (0.5 * (graph1.number_of_nodes() + graph2.number_of_nodes()))
    return norm_ged

def return_eq(node1, node2):
    return node1['type'] == node2['type']

def edge_eq(e1, e2):
    return e1['type'] == e2['type']

def sorted_nicely(l):
    def tryint(s):
        try:
            return int(s)
        except:
            return s

    import re
    def alphanum_key(s):
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    return sorted(l, key=alphanum_key)


def load_gexf_file(dataset_name, train_or_test='train', node_featue_name='type'):
    graphs = []
    dir = '/home/jiyang/SimGNN_pytorch/dataset/' + dataset_name + '/' + train_or_test

    for file in sorted_nicely(glob(dir + '/*.gexf')):
        gid = int(basename(file).split('.')[0])
        g = nx.read_gexf(file)
        g.graph['gid'] = gid
        graphs.append(g)
        if not nx.is_connected(g):
            raise RuntimeError('{} not connected'.format(gid))

    for file in sorted_nicely(glob('/home/jiyang/SimGNN_pytorch/dataset/' + dataset_name + '/' + 'test' + '/*.gexf')):
        gid = int(basename(file).split('.')[0])
        g = nx.read_gexf(file)
        g.graph['gid'] = gid
        graphs.append(g)
        if not nx.is_connected(g):
            raise RuntimeError('{} not connected'.format(gid))

    inputs_set = set()

    return graphs

def load_graphs(dataset_name, train_or_test='train'):
    graphs = []
    dir = './dataset/' + dataset_name + '/' + train_or_test
    for file in sorted_nicely(glob(dir + '/*.gexf')):
        gid = int(basename(file).split('.')[0])
        g = nx.read_gexf(file)
        g.graph['gid'] = gid
        graphs.append(g)
        if not nx.is_connected(g):
            raise RuntimeError('{} not connected'.format(gid))
    return graphs

def load_generated_graphs(dataset_name, file_name='generated_graph_500'):

    dir = './dataset/' + dataset_name + '/' + file_name
    g = open(dir, 'rb')
    generated_graphs = pickle.load(g)
    g.close()
    return generated_graphs

# def load_sdz_dataset(dataset_name='AID2DA99.sdz'):
#     dir = '/home/jiyang/SimGNN_pytorch/dataset/' + dataset_name

#     count, graph_list, global_node_label, global_edge_label = 0, [], [], []
#     with open(dir, 'r', encoding='UTF-8') as file:
#         line = file.readline()

#         while(line is not None and line != ''):
#             # line_list = line.split(' ')    
#             if(line == '$$$$\n'):
#                 line = file.readline()
#                 line = file.readline()
#                 line = file.readline()
#                 line = file.readline()
#                 vertexSize, edgeSize = int(line[0:3]), int(line[3:6])
#                 print('vesize', vertexSize, edgeSize)
#                 G = nx.Graph()
#                 for i in range(vertexSize):
#                     line = file.readline()
#                     line_list = line.split(' ')
#                     while('' in line_list):
#                         line_list.remove('')
#                     G.add_node(str(i), label=str(i), type=line_list[3])
#                     if(line_list[3] not in global_node_label):
#                         global_node_label.append(line_list[3])

#                 for j in range(edgeSize):
#                     line = file.readline()
#                     G.add_edge(str(int(line[0:3]) - 1), str(int(line[3:6]) - 1), type = line[8:9])
#                     if(line[8:9] not in global_edge_label):
#                         global_edge_label.append(line[8:9])
#                 graph_list.append(G)
#                 # print(nx.get_edge_attributes(G, 'type'))

#             # if(count == 200):
#             #     break
#             line = file.readline()
#             count+=1

#     return graph_list


# def load_PubChem(dataset_name='PubChem'):
#     dir = '/home/jiyang/SimGNN_pytorch/dataset/' + dataset_name + '/PubChem.txt'

#     graph_list, global_node_label, global_edge_label, count = [], [], [], 0
#     with open(dir, 'r', encoding='UTF-8') as file:
#         line = file.readline()
#         G = nx.Graph()

#         while(line is not None and line != ''):
#             line_list = line.split(' ')
#             if(line_list[0] == 't'):
#                 count += 1
#                 graph_list.append(G)
#                 G = nx.Graph()
#                 print(count)
#             elif(line_list[0] == 'v'):
#                 G.add_node(line_list[1], label=line_list[1], type=line_list[2])#[:-1])
#                 if(line_list[2] not in global_node_label):
#                     global_node_label.append(line_list[2])#[:-1])
#             elif(line_list[0] == 'e'):
#                 G.add_edge(line_list[1], line_list[2], type=line_list[3])#[:-1])
#                 if(line_list[3] not in global_edge_label):
#                     global_edge_label.append(line_list[3])#[:-1])

#             line = file.readline()
        
#     return graph_list, global_node_label, global_edge_label


# if __name__ == "__main__":





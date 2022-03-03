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

    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def process_pair(path):

    data = json.load(open(path))
    return data

def calculate_loss(prediction, target):

    prediction = -math.log(prediction)
    target = -math.log(target)
    score = (prediction-target)**2
    return score

def calculate_normalized_ged(data):

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
    dir = './dataset/' + dataset_name + '/' + train_or_test

    for file in sorted_nicely(glob(dir + '/*.gexf')):
        gid = int(basename(file).split('.')[0])
        g = nx.read_gexf(file)
        g.graph['gid'] = gid
        graphs.append(g)
        if not nx.is_connected(g):
            raise RuntimeError('{} not connected'.format(gid))

    for file in sorted_nicely(glob('./dataset/' + dataset_name + '/' + 'test' + '/*.gexf')):
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



# if __name__ == "__main__":





import numpy as np
import networkx as nx
from copy import deepcopy
import pickle
import random

from utils import load_generated_graphs

# def return_eq(node1, node2):
#     return node1['type']==node2['type']

def graph_generation(graph, global_labels, global_edge_labels, total_ged=0):

    new_g =  deepcopy(graph)

    target_ged = {}
    while(True):
        target_ged['nc'] = np.random.randint(0, max(int(new_g.number_of_nodes()/3), 1))
        target_ged['ec'] = np.random.randint(0, max(int(new_g.number_of_edges()/3), 1))
        target_ged['in'] = np.random.randint(1, 5)
        max_add_edge = min(int(((new_g.number_of_nodes()+target_ged['in'])*(new_g.number_of_nodes()+target_ged['in'] - 1)/2 - new_g.number_of_edges())/2), 4)
        if(max_add_edge <= (target_ged['in'] + 1)):
            max_add_edge = target_ged['in'] + 1
        target_ged['ie'] = np.random.randint(target_ged['in'], max_add_edge)

        temp_ged = target_ged['nc'] + target_ged['in'] + target_ged['ie'] + target_ged['ec']
        if(temp_ged != 0):
            break
    target_ged['nc'] = round(target_ged['nc']*total_ged/temp_ged)
    target_ged['in'] = round(target_ged['in']*total_ged/temp_ged)
    target_ged['ie'] = round(target_ged['ie']*total_ged/temp_ged)
    target_ged['ec'] = round(target_ged['ec']*total_ged/temp_ged)
    if(target_ged['ie'] < target_ged['in']):
        target_ged['ie'] = target_ged['in']

    print('target ged ', target_ged)

    ## edit node labels
    to_edit_idx_newg = random.sample(new_g.nodes(), target_ged['nc'])
    for idx in to_edit_idx_newg:
        while(True):
            toassigned_new_nodetype = random.choice(list(global_labels))
            if(toassigned_new_nodetype != new_g.nodes()[idx]['type']):
                break

        new_g.nodes()[idx]['type'] = toassigned_new_nodetype



    ## edit edge deletion
    if((target_ged['ie'] - target_ged['in']) == 0):
        to_ins, to_del = 0, 0
    else:
        to_del = min(int(new_g.number_of_edges()/3), np.random.randint(0, (target_ged['ie'] - target_ged['in'])))
        to_ins = target_ged['ie'] - target_ged['in'] - to_del

    deleted_edges = []
    for num in range(to_del):
        curr_num_egde = new_g.number_of_edges()
        to_del_edge = random.sample(new_g.edges(), 1)
        deleted_edges.append(to_del_edge[0])
        deleted_edges.append((to_del_edge[0][1], to_del_edge[0][0]))
        new_g.remove_edges_from(to_del_edge)
        assert((curr_num_egde - new_g.number_of_edges()) == 1)

    ## edit edge labels
    to_edit_idx_edge = random.sample(new_g.edges(), target_ged['ec'])
    for idx in to_edit_idx_edge:
        while(True):
            toassigned_new_edgetype = random.choice(global_edge_labels)
            if(toassigned_new_edgetype != new_g.edges()[idx]['type']):
                break
        new_g.edges()[idx]['type'] = toassigned_new_edgetype


    ## edit node insertions    
    for num in range(target_ged['in']):
        curr_num_node = new_g.number_of_nodes()
        to_insert_edge = random.sample(new_g.nodes(), 1)[0]
        new_g.add_node(str(curr_num_node), label=str(curr_num_node), type=random.choice(global_labels))
        # add edge to the newly inserted ndoe
        new_g.add_edge(str(curr_num_node), to_insert_edge, type=random.choice(global_edge_labels))


    ## edit edge insertions
    for num in range(to_ins):
        curr_num_egde = new_g.number_of_edges()
        while(True):
            curr_pair = random.sample(new_g.nodes(), 2)
            if((curr_pair[0], curr_pair[1]) not in deleted_edges):
                # print('poten edge', curr_pair[0], curr_pair[1])
                if((curr_pair[0], curr_pair[1]) not in new_g.edges()):
                    # print('added adge', curr_pair[0], curr_pair[1])
                    new_g.add_edge(curr_pair[0], curr_pair[1], type=random.choice(global_edge_labels))
                    break

   
    # print('Total target ged', target_ged['nc'] + target_ged['ec'] + target_ged['in'] + target_ged['ie'], total_ged)
    # print('----------------------------------------------------------------------------')
    
    return target_ged, new_g


if __name__ == "__main__":

    ## Load graphs and the global labels, which can be accessed in dataset file;
    dataset_name = 'AIDS'#'PubChem'#

    graphs = load_generated_graphs(dataset_name, 'AIDS')
    
    g = open('./dataset/' + dataset_name + '/global_node_label', 'rb')
    global_node_labels = pickle.load(g)
    g.close()

    g = open('./dataset/' + dataset_name + '/global_edge_label', 'rb')
    global_edge_labels = pickle.load(g)
    g.close()

    ## Here is an example of generating graph pair (graphs[0], new_g) and their corresponding target_ged;
    ## Note that the input "total_ged" here is a randomly sampled value, it is only for reference and may not exactly equal to the returned final target_ged
    target_ged, new_g = graph_generation(graphs[0], global_node_labels, global_edge_labels, total_ged=random.randint(18))




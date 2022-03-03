import glob
import torch
import random
import pickle
import numpy as np
import scipy.sparse as sp
import networkx as nx
from layers import TensorNetworkModule, GraphAggregationLayer
from utils import load_graphs, load_generated_graphs, process_pair


class TaGSim(torch.nn.Module):

    def __init__(self, args, number_of_node_labels, number_of_edge_labels):

        super(TaGSim, self).__init__()
        self.args = args
        self.number_of_node_labels = number_of_node_labels
        self.number_of_edge_labels = number_of_edge_labels
        self.setup_layers()



    def setup_layers(self):

        self.gal1 = GraphAggregationLayer()
        self.gal2 = GraphAggregationLayer()
        self.feature_count = self.args.tensor_neurons

        self.tensor_network_nc = TensorNetworkModule(self.args, 2*self.number_of_node_labels)
        self.tensor_network_in = TensorNetworkModule(self.args, 2*self.number_of_node_labels)
        self.tensor_network_ie = TensorNetworkModule(self.args, 2*self.number_of_node_labels)
        self.tensor_network_ec = TensorNetworkModule(self.args, 2*self.number_of_edge_labels)

        self.fully_connected_first_nc = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.fully_connected_second_nc = torch.nn.Linear(self.args.bottle_neck_neurons, 8)
        self.fully_connected_third_nc = torch.nn.Linear(8, 4)
        self.scoring_layer_nc = torch.nn.Linear(4, 1)

        self.fully_connected_first_in = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.fully_connected_second_in = torch.nn.Linear(self.args.bottle_neck_neurons, 8)
        self.fully_connected_third_in = torch.nn.Linear(8, 4)
        self.scoring_layer_in = torch.nn.Linear(4, 1)

        self.fully_connected_first_ie = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.fully_connected_second_ie = torch.nn.Linear(self.args.bottle_neck_neurons, 8)
        self.fully_connected_third_ie = torch.nn.Linear(8, 4)
        self.scoring_layer_ie = torch.nn.Linear(4, 1)

        self.fully_connected_first_ec = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.fully_connected_second_ec = torch.nn.Linear(self.args.bottle_neck_neurons, 8)
        self.fully_connected_third_ec = torch.nn.Linear(8, 4)
        self.scoring_layer_ec = torch.nn.Linear(4, 1)


    def gal_pass(self, edge_index, features):

        hidden1 = self.gal1(features, edge_index)
        hidden2 = self.gal2(hidden1, edge_index)

        return hidden1, hidden2


    def forward(self, data):

        adj_1, adj_2 = torch.FloatTensor(np.array(data["edge_index_1"].todense())), torch.FloatTensor(np.array(data["edge_index_2"].todense()))
        edge_adj_1, edge_adj_2 =  data["edge_adj_1"], data["edge_adj_2"]
        features_1, features_2 = data["features_1"], data["features_2"]
        edge_features_1, edge_features_2 = data["edge_features_1"], data["edge_features_2"]

        graph1_hidden1, graph1_hidden2 = self.gal_pass(adj_1, features_1)#
        graph2_hidden1, graph2_hidden2 = self.gal_pass(adj_2, features_2)#
        edge1_hidden1, edge1_hidden2 = self.gal_pass(edge_adj_1, edge_features_1)
        edge2_hidden1, edge2_hidden2 = self.gal_pass(edge_adj_2, edge_features_2)


        graph1_01concat = torch.cat([features_1, graph1_hidden1], dim=1)
        graph2_01concat = torch.cat([features_2, graph2_hidden1], dim=1)
        graph1_12concat = torch.cat([graph1_hidden1, graph1_hidden2], dim=1)
        graph2_12concat = torch.cat([graph2_hidden1, graph2_hidden2], dim=1)

        graph1_01pooled = torch.sum(graph1_01concat, dim=0).unsqueeze(1)#sum
        graph2_01pooled = torch.sum(graph2_01concat, dim=0).unsqueeze(1)
        graph1_12pooled = torch.sum(graph1_12concat, dim=0).unsqueeze(1)
        graph2_12pooled = torch.sum(graph2_12concat, dim=0).unsqueeze(1)
        

        edge1_01concat = torch.cat([edge_features_1, edge1_hidden1], dim=1)
        edge2_01concat = torch.cat([edge_features_2, edge2_hidden1], dim=1)

        edge1_01pooled = torch.sum(edge1_01concat, dim=0).unsqueeze(1)#sum
        edge2_01pooled = torch.sum(edge2_01concat, dim=0).unsqueeze(1)


        scores_nc = self.tensor_network_nc(graph1_01pooled, graph2_01pooled)
        scores_nc = torch.t(scores_nc)

        scores_nc = torch.nn.functional.relu(self.fully_connected_first_nc(scores_nc))
        scores_nc = torch.nn.functional.relu(self.fully_connected_second_nc(scores_nc))
        scores_nc = torch.nn.functional.relu(self.fully_connected_third_nc(scores_nc))
        score_nc = torch.sigmoid(self.scoring_layer_nc(scores_nc))

        scores_in = self.tensor_network_in(graph1_01pooled, graph2_01pooled)
        scores_in = torch.t(scores_in)

        scores_in = torch.nn.functional.relu(self.fully_connected_first_in(scores_in))
        scores_in = torch.nn.functional.relu(self.fully_connected_second_in(scores_in))
        scores_in = torch.nn.functional.relu(self.fully_connected_third_in(scores_in))
        score_in = torch.sigmoid(self.scoring_layer_in(scores_in))

        scores_ie = self.tensor_network_ie(graph1_12pooled, graph2_12pooled)
        scores_ie = torch.t(scores_ie)

        scores_ie = torch.nn.functional.relu(self.fully_connected_first_ie(scores_ie))
        scores_ie = torch.nn.functional.relu(self.fully_connected_second_ie(scores_ie))
        scores_ie = torch.nn.functional.relu(self.fully_connected_third_ie(scores_ie))
        score_ie = torch.sigmoid(self.scoring_layer_ie(scores_ie))

        scores_ec = self.tensor_network_ec(edge1_01pooled, edge2_01pooled)
        scores_ec = torch.t(scores_ec)

        scores_ec = torch.nn.functional.relu(self.fully_connected_first_ec(scores_ec))
        scores_ec = torch.nn.functional.relu(self.fully_connected_second_ec(scores_ec))
        scores_ec = torch.nn.functional.relu(self.fully_connected_third_ec(scores_ec))
        score_ec = torch.sigmoid(self.scoring_layer_ec(scores_ec))

        return torch.cat([score_nc, score_in, score_ie, score_ec], dim=1)


class TaGSimTrainer(object):

    def __init__(self, args):

        self.args = args
        self.initial_label_enumeration()
        self.model = TaGSim(self.args, self.number_of_node_labels, self.number_of_edge_labels)

    def initial_label_enumeration(self):

        self.training_pairs = load_generated_graphs(self.args.dataset, file_name='generated_graph_pairs')
        g = open('./dataset/' + self.args.dataset + '/AIDS', 'rb')
        self.all_graphs = pickle.load(g)
        g.close()
        
        g = open('./dataset/' + self.args.dataset + '/global_node_label', 'rb')
        self.global_node_labels = pickle.load(g)
        g.close()

        g = open('./dataset/' + self.args.dataset + '/global_edge_label', 'rb')
        self.global_edge_labels = pickle.load(g)
        g.close()

        self.number_of_node_labels = len(self.global_node_labels)
        self.number_of_edge_labels = len(self.global_edge_labels)


    def transfer_to_torch(self, data, type_specified=True):
 
        new_data = dict()
        graph1, graph2 = data['graph_pair'][0], data['graph_pair'][1]
        nodes1, nodes2 = list(graph1.nodes()), list(graph2.nodes())

        features_1, features_2, edge_features_1, edge_features_2, edge_adj_1, edge_adj_2 = [], [], [], [], [], []

        for n in graph1.nodes():
            features_1.append([1.0 if graph1.nodes()[n]['type'] == ele else 0.0 for ele in self.global_node_labels])

        for n in graph2.nodes():
            features_2.append([1.0 if graph2.nodes()[n]['type'] == ele else 0.0 for ele in self.global_node_labels])
        features_1, features_2 = torch.FloatTensor(np.array(features_1)), torch.FloatTensor(np.array(features_2))

        for e in graph1.edges():
            edge_features_1.append([1.0 if graph1.edges()[e]['type'] == ele else 0.0 for ele in self.global_edge_labels])
            adj_row = []
            for d in graph1.edges():
                if(e == d):
                    adj_row.append(0.0)
                    continue
                if((e[0] in d) | (e[1] in d)):
                    adj_row.append(1.0)
                else:
                    adj_row.append(0.0)
            edge_adj_1.append(adj_row)

        for e in graph2.edges():
            edge_features_2.append([1.0 if graph2.edges()[e]['type'] == ele else 0.0 for ele in self.global_edge_labels])
            adj_row = []
            for d in graph2.edges():
                if(e == d):
                    adj_row.append(0.0)
                    continue
                if((e[0] in d) | (e[1] in d)):
                    adj_row.append(1.0)
                else:
                    adj_row.append(0.0)
            edge_adj_2.append(adj_row)

        edge_features_1, edge_features_2 = torch.FloatTensor(np.array(edge_features_1)), torch.FloatTensor(np.array(edge_features_2))
        edge_adj_1, edge_adj_2 = torch.FloatTensor(np.array(edge_adj_1)), torch.FloatTensor(np.array(edge_adj_2))

        new_data["edge_index_1"], new_data["edge_index_2"] = nx.adjacency_matrix(graph1), nx.adjacency_matrix(graph2)
        new_data["features_1"], new_data["features_2"] = features_1, features_2
        new_data["edge_features_1"], new_data["edge_features_2"] = edge_features_1, edge_features_2
        new_data["edge_adj_1"], new_data["edge_adj_2"] = edge_adj_1, edge_adj_2


        if(type_specified):
            norm_ged = [data['ged'][key] / (0.5 * (graph1.number_of_nodes() + graph2.number_of_nodes())) for key in ['nc', 'in', 'ie', 'ec']]
            norm_ged = np.array(norm_ged)
            new_data["target"] = torch.from_numpy(np.exp(-norm_ged)).view(1,-1).float()
            
            norm_gt_ged = (data['ged']['nc'] + data['ged']['in'] + data['ged']['ie'] + data['ged']['ec']) / (0.5 * (graph1.number_of_nodes() + graph2.number_of_nodes()))
            new_data["gt_ged"] = torch.from_numpy(np.exp(-norm_gt_ged).reshape(1, 1)).view(1, -1).float()
        else:
            norm_gt_ged = data['ged'] / (0.5 * (graph1.number_of_nodes() + graph2.number_of_nodes()))
            new_data["gt_ged"] = torch.from_numpy(np.exp(-norm_gt_ged).reshape(1, 1)).view(1, -1).float()

        return new_data

#----------------------------------------------------------------------------------------------------------
    def fit(self):
        print("\n-------Model training---------.\n")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        iteration = 0

        for epoch in range(self.args.epochs):
            random.shuffle(self.training_pairs)
            batches = []
            for graph in range(0, len(self.training_pairs), self.args.batch_size):
                batches.append(self.training_pairs[graph:graph+self.args.batch_size])

            for batch in batches:
                self.model.train()
                self.optimizer.zero_grad()
                losses = 0
                for graph_pair in batch:
                    data = self.transfer_to_torch(graph_pair)
                    prediction = self.model(data)
                    losses += torch.nn.functional.mse_loss(data["target"], prediction)
                losses.backward(retain_graph=True)
                self.optimizer.step()
                loss = losses.item()

                print('Iteration', iteration, 'loss: ', loss/len(batch))
                
                iteration += 1
               
#-------------------------------------------------------------------------------------------------------
    def test(self):

        print("\n\nModel testing.\n")
       
        self.model.eval()
        self.test_scores = []

        test_gt_ged = load_generated_graphs(self.args.dataset, file_name='ged_matrix_test')
        for i in range(40200, len(self.all_graphs)):
            for j in range(len(self.all_graphs)):
                if((i, j) in test_gt_ged):
                    curr_graph_pair = {'graph_pair': [self.all_graphs[i], self.all_graphs[j]], 'ged':test_gt_ged[(i, j)]}
                    data = self.transfer_to_torch(curr_graph_pair, type_specified=False)
                    prediction = self.model(data)
                    prediction = torch.exp(torch.sum(torch.log(prediction))).view(1, -1)
                    current_error = torch.nn.functional.mse_loss(prediction, data["gt_ged"])

                    self.test_scores.append(current_error.data.item())


        model_error = sum(self.test_scores) / len(self.test_scores)
        print("\nModel test error: " + str(model_error))

        

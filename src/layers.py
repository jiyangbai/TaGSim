import torch
import torch.nn as nn
import torch.nn.functional as F


class TensorNetworkModule(torch.nn.Module):

    def __init__(self, args, input_dim):

        super(TensorNetworkModule, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):

        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.input_dim, self.input_dim, self.args.tensor_neurons))

        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 2*self.input_dim))
        self.bias = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 1))

    def init_parameters(self):

        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):

        scoring = torch.mm(torch.t(embedding_1), self.weight_matrix.view(self.input_dim, -1))
        scoring = scoring.view(self.input_dim, self.args.tensor_neurons)
        scoring = torch.mm(torch.t(scoring), embedding_2)
        combined_representation = torch.cat((embedding_1, embedding_2))
        block_scoring = torch.mm(self.weight_matrix_block, combined_representation)
        scores = torch.nn.functional.relu(scoring + block_scoring + self.bias)
        return scores


class GraphAggregationLayer(nn.Module):

    def __init__(self, in_features=10, out_features=10):
        super(GraphAggregationLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features


    def forward(self, input, adj):
        h_prime = torch.mm(adj, input)

        return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



        
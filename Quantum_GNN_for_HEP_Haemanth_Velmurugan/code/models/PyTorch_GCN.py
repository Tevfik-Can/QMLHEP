import torch
from torch.nn import Linear, ModuleList
from torch_geometric.nn import GCNConv, global_mean_pool


class GNN(torch.nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, activ_fn):
        super().__init__()
        layers = []
        layers.append(GCNConv(input_dims, hidden_dims[0]))

        for i in range(len(hidden_dims)-1):
            layers.append(GCNConv(hidden_dims[i], hidden_dims[i+1]))

        self.layers = ModuleList(layers)
        self.activ_fn = activ_fn
        self.classifier = Linear(hidden_dims[-1], output_dims)

    def forward(self, x, edge_index, batch):
        h = x
        for i in range(len(self.layers)):
            h = self.layers[i](h, edge_index)
            h = self.activ_fn(h)

        # readout layer to get the embedding for each graph in batch
        h = global_mean_pool(h, batch)
        h = self.classifier(h)
        return h

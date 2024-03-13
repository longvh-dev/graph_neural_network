import time
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from pprint import pprint
# from __future__ import division
# from __future__ import print_function

from GCN import GCN
from graph import Graph
from utils import load_data


# Training settings
def get_args():
    parser = argparse.ArgumentParser()

    # training hyperparameters
    parser.add_argument('--no-cuda', action = 'store_true', default = False,
                        help = 'Disables CUDA training.')
    parser.add_argument('--fastmode', action = 'store_true', default = False,
                        help = 'Validate during training pass.')
    parser.add_argument('--seed', type = int, default = 42,
                        help = 'Random seed.')
    parser.add_argument('--epochs', type = int, default = 200,
                        help = 'Number of epochs to train.')
    parser.add_argument('--lr', type = float, default = 0.01,
                        help = 'Initial learning rate.')
    parser.add_argument('--weight_decay', type = float, default = 5e-4,
                        help = 'Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type = int, default = 16,
                        help = 'Number of hidden units.')
    parser.add_argument('--dropout', type = float, default = 0.5,
                        help = 'Dropout rate (1 - keep probability).')

    args = parser.parse_args()
    return args


# args = get
# args.cuda = not args.no_cuda and torch.cuda.is_available()
#
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)
#
# # Load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data('1')
#
# num_features = features.shape[1]
# num_classes = labels.max().item() + 1
# # Model and optimizer
# model = GCN(num_features = features.shape[1],
#             num_classes = labels.max().item() + 1,
#             dropout = args.dropout
#             )
# optimizer = optim.Adam(model.parameters(),
#                        lr = args.lr, weight_decay = args.weight_decay)
#
# if args.cuda:
#     model.cuda()
#     features = features.cuda()
#     adj = adj.cuda()
#     labels = labels.cuda()
#     idx_train = idx_train.cuda()
#     idx_val = idx_val.cuda()
#     idx_test = idx_test.cuda()

# class GraphTrainer:
def main():
    file_path = 'small_ESConv.json'
    conversation = load_data(file_path)
    conversation_graph_instance = Graph(conversation, similarity_threshold =
    0.8)
    conversation_graph_instance.plot_graph()

    adjacency_matrix = conversation_graph_instance.get_adjacency_matrix()
    print(adjacency_matrix)
    print('----------------------------------')

    adjacency_tensor = torch.tensor(adjacency_matrix.todense(),
                                    dtype = torch.float32)
    # pprint(1)
    print(adjacency_tensor)
    print('----------------------------------')

    num_features = 27  # Number of input features for each node
    num_classes = 27  # Number of output classes
    model = GCN(num_features, num_classes, dropout = 0.5)

    # input is adjacency tensor, output is the prediction
    output = model(adjacency_tensor, adjacency_tensor)
    print(output.detach().numpy())
    predicted_classes = output.argmax(axis = 1)
    print(predicted_classes)
    print('----------------------------------')


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.nn.functional as F

from impl_config import FeatureParserConfig as fp
from impl_config import NetworkConfig as ns

class DuelingQNetwork(nn.Module):
    """Dueling Q-network (https://arxiv.org/abs/1511.06581)"""

    def __init__(self, state_size, action_size, hidsize1=128, hidsize2=128):
        super(DuelingQNetwork, self).__init__()

        # value network
        self.fc1_val = nn.Linear(state_size, hidsize1)
        self.fc2_val = nn.Linear(hidsize1, hidsize2)
        self.fc4_val = nn.Linear(hidsize2, 1)

        # advantage network
        self.fc1_adv = nn.Linear(state_size, hidsize1)
        self.fc2_adv = nn.Linear(hidsize1, hidsize2)
        self.fc4_adv = nn.Linear(hidsize2, action_size)

    def forward(self, x):
        val = F.relu(self.fc1_val(x))
        val = F.relu(self.fc2_val(val))
        val = self.fc4_val(val)

        # advantage calculation
        adv = F.relu(self.fc1_adv(x))
        adv = F.relu(self.fc2_adv(adv))
        adv = self.fc4_adv(adv)

        return val + adv - adv.mean()
    
class TreeLSTM(nn.Module):
    """
    https://github.com/unbounce/pytorch-tree-lstm/blob/master/treelstm/tree_lstm.py#L10
    """

    def __init__(self, in_features, out_features) -> None:
        """TreeLSTM class initializer
        Takes in int sizes of in_features and out_features and sets up model Linear network layers.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # bias terms are only on the W layers for efficiency
        self.W_iou = torch.nn.Linear(self.in_features, 3 * self.out_features)
        self.U_iou = torch.nn.Linear(
            3 * self.out_features, 3 * self.out_features, bias=False
        )
        # self.W_h = torch.nn.Linear(3 * self.out_features, self.out_features)
        self.W_c = torch.nn.Linear(3 * self.out_features, self.out_features)

        # f terms are maintained seperate from the iou terms because they involve sums over child nodes
        # while the iou terms do not
        self.W_f = torch.nn.Linear(self.in_features, self.out_features)
        self.U_f = torch.nn.Linear(self.out_features, self.out_features, bias=False)

    def forward(self, forest, adjacency, node_order, edge_order):
        """Run TreeLSTM model on a tree data structure with node features
        Takes Tensors encoding node features, a tree node adjacency_list, and the order in which
        the tree processing should proceed in node_order and edge_order.
        """
        forest = forest.flatten(0, 2)
        adjacency_list = adjacency.flatten(0, 2)
        node_order = node_order.flatten(0, 2)
        edge_order = edge_order.flatten(0, 2)

        # Total number of nodes in every tree in the batch
        batch_size = node_order.shape[0]

        # Retrive device the model is currently loaded on to generate h, c, and h_sum result buffers
        device = next(self.parameters()).device

        # h and c states for every node in the batch
        h = torch.zeros(batch_size, self.out_features, device=device)
        c = torch.zeros(batch_size, self.out_features, device=device)

        # populate the h and c states respecting computation order
        for n in range(node_order.max() + 1):
            self._run_lstm(n, h, c, forest, node_order, adjacency_list, edge_order)
        return h

    # @torchsnooper.snoop()
    def _run_lstm(
        self,
        iteration: int,
        h: torch.Tensor,
        c: torch.Tensor,
        features: torch.Tensor,
        node_order: torch.Tensor,
        adjacency_list: torch.Tensor,
        edge_order: torch.Tensor,
    ):
        """Helper function to evaluate all tree nodes currently able to be evaluated."""
        # N is the number of nodes in the tree
        # n is the number of nodes to be evaluated on in the current iteration
        # E is the number of edges in the tree
        # e is the number of edges to be evaluated on in the current iteration
        # F is the number of features in each node
        # M is the number of hidden neurons in the network

        # node_order is a tensor of size N x 1
        # edge_order is a tensor of size E x 1
        # features is a tensor of size N x F
        # adjacency_list is a tensor of size E x 2

        # node_mask is a tensor of size N x 1
        node_mask = node_order == iteration
        # edge_mask is a tensor of size E x 1
        edge_mask = edge_order == iteration

        # x is a tensor of size n x F
        x = features[node_mask, :]

        # At iteration 0 none of the nodes should have children
        # Otherwise, select the child nodes needed for current iteration
        # and sum over their hidden states
        if iteration == 0:
            iou = self.W_iou(x)
        else:
            # adjacency_list is a tensor of size e x 2
            adjacency_list = adjacency_list[edge_mask, :]

            # parent_indexes and child_indexes are tensors of size e x 1
            # parent_indexes and child_indexes contain the integer indexes needed to index into
            # the feature and hidden state arrays to retrieve the data for those parent/child nodes.
            parent_indexes = adjacency_list[:, 0]
            child_indexes = adjacency_list[:, 1]

            # child_h and child_c are tensors of size e x 1
            child_h = h[child_indexes, :]
            child_c = c[child_indexes, :]

            # # Add child hidden states to parent offset locations
            # _, child_counts = torch.unique_consecutive(parent_indexes, return_counts=True)
            # child_counts = tuple(child_counts)

            # parent_children = torch.split(child_h, child_counts)
            # parent_list = [item.sum(0) for item in parent_children]

            # h_sum = torch.stack(parent_list)

            i_dims = child_h.shape[0] // 3
            child_h_merge = child_h.unflatten(0, (i_dims, 3)).flatten(start_dim=1)
            # h_reduce = self.W_h(child_h_merge)

            iou = self.W_iou(x) + self.U_iou(child_h_merge)

        # i, o and u are tensors of size n x M
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        u = torch.tanh(u)

        # At iteration 0 none of the nodes should have children
        # Otherwise, calculate the forget states for each parent node and child node
        # and sum over the child memory cell states
        if iteration == 0:
            c[node_mask, :] = i * u
        else:
            # f is a tensor of size e x M
            f = self.W_f(features[parent_indexes, :]) + self.U_f(child_h)
            f = torch.sigmoid(f)

            # fc is a tensor of size e x M
            fc = f * child_c

            # Add the calculated f values to the parent's memory cell state
            # parent_children = torch.split(fc, child_counts)
            # parent_list = [item.sum(0) for item in parent_children]

            # c_sum = torch.stack(parent_list)

            fc = fc.unflatten(0, (fc.shape[0] // 3, 3)).flatten(start_dim=1)
            c_reduce = self.W_c(fc)

            c[node_mask, :] = i * u + c_reduce

        h[node_mask, :] = o * torch.tanh(c[node_mask])

class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Transformer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.att_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
        )

    def forward(self, input):
        batch_size, n_agents, embedding_size = input.shape
        input = input.permute(1, 0, 2)  # agents, batch, embedding
        input = input.view(n_agents, -1, embedding_size)
        output, _ = self.attention(input, input, input)

        input = input.view(n_agents, -1, embedding_size)
        input = input.permute(1, 0, 2)
        output = output.view(n_agents, -1, embedding_size)
        output = output.permute(1, 0, 2)
        output = self.att_mlp(torch.cat([input, output], dim=-1))
        return output

class LSTMQNetwork(nn.Module):
    def __init__(self):
        super(TransLSTM, self).__init__()
        self.tree_lstm = TreeLSTM(fp.node_sz, ns.tree_embedding_sz)
        self.attr_embedding = nn.Sequential(
            nn.Linear(fp.agent_attr, 2 * ns.hidden_sz),
            nn.GELU(),
            nn.Linear(2 * ns.hidden_sz, 2 * ns.hidden_sz),
            nn.GELU(),
            nn.Linear(2 * ns.hidden_sz, 2 * ns.hidden_sz),
            nn.GELU(),
            nn.Linear(2 * ns.hidden_sz, ns.hidden_sz),
            nn.GELU(),
        )
        self.actor_net = nn.Sequential(
            nn.Linear(ns.hidden_sz + ns.tree_embedding_sz, 2 * ns.hidden_sz),
            nn.GELU(),
            nn.Linear(ns.hidden_sz * 2, ns.hidden_sz),
            nn.GELU(),
            nn.Linear(ns.hidden_sz, fp.action_sz),
            nn.Softmax(dim=-1),
        )
        self.critic_net = nn.Sequential(
            nn.Linear(ns.hidden_sz + ns.tree_embedding_sz, ns.hidden_sz * 2),
            nn.GELU(),
            nn.Linear(ns.hidden_sz * 2, ns.hidden_sz),
            nn.GELU(),
            nn.Linear(ns.hidden_sz, 1),
        )

    def forward(self, agents_attr, forest, adjacency, node_order, edge_order):
        batch_size, n_agents, num_nodes, _ = forest.shape
        device = next(self.parameters()).device
        adjacency = self.modify_adjacency(adjacency, device)

        tree_embedding = self.tree_lstm(forest, adjacency, node_order, edge_order)
        tree_embedding = tree_embedding.unflatten(0, (batch_size, n_agents, num_nodes))[
            :, :, 0, :
        ]

        agent_attr_embedding = self.attr_embedding(agents_attr)
        embedding = torch.cat([agent_attr_embedding, tree_embedding], dim=2)

        worker_action = self.actor_net(embedding)
        critic_value = self.critic_net(embedding)
        critic_value = critic_value.mean(1).view(-1)
        return [worker_action], critic_value  # (batch size, 1)

class TransLSTM(nn.Module):
    """
    Feature:  cat(agents_attr_embedding, tree_embedding)
    structure: mlp
    """
    def __init__(self):
        super(TransLSTM, self).__init__()
        self.tree_lstm = TreeLSTM(fp.node_sz, ns.tree_embedding_sz)
        self.attr_embedding = nn.Sequential(
            nn.Linear(fp.agent_attr, 2 * ns.hidden_sz),
            nn.GELU(),
            nn.Linear(2 * ns.hidden_sz, 2 * ns.hidden_sz),
            nn.GELU(),
            nn.Linear(2 * ns.hidden_sz, 2 * ns.hidden_sz),
            nn.GELU(),
            nn.Linear(2 * ns.hidden_sz, ns.hidden_sz),
            nn.GELU(),
        )
        self.transformer = nn.Sequential(
            Transformer(ns.hidden_sz + ns.tree_embedding_sz, 4),
            Transformer(ns.hidden_sz + ns.tree_embedding_sz, 4),
            Transformer(ns.hidden_sz + ns.tree_embedding_sz, 4),
        )
        self.actor_net = nn.Sequential(
            nn.Linear(ns.hidden_sz * 2 + ns.tree_embedding_sz * 2, 2 * ns.hidden_sz),
            nn.GELU(),
            nn.Linear(ns.hidden_sz * 2, ns.hidden_sz),
            nn.GELU(),
            nn.Linear(ns.hidden_sz, fp.action_sz),
        )
        self.critic_net = nn.Sequential(
            nn.Linear(ns.hidden_sz * 2 + ns.tree_embedding_sz * 2, ns.hidden_sz * 2),
            nn.GELU(),
            nn.Linear(ns.hidden_sz * 2, ns.hidden_sz),
            nn.GELU(),
            nn.Linear(ns.hidden_sz, 1),
        )

    def forward(self, agents_attr, forest, adjacency, node_order, edge_order):
        batch_size, n_agents, num_nodes, _ = forest.shape
        device = next(self.parameters()).device
        adjacency = self.modify_adjacency(adjacency, device)

        tree_embedding = self.tree_lstm(forest, adjacency, node_order, edge_order)
        tree_embedding = tree_embedding.unflatten(0, (batch_size, n_agents, num_nodes))[
            :, :, 0, :
        ]

        agent_attr_embedding = self.attr_embedding(agents_attr)
        embedding = torch.cat([agent_attr_embedding, tree_embedding], dim=2)

        ## attention
        att_embedding = self.transformer(embedding)

        worker_action = torch.zeros((batch_size, n_agents, 5), device=device)
        worker_action[:, :n_agents, :] = self.actor(embedding, att_embedding)
        critic_value = self.critic(embedding, att_embedding)
        return [worker_action], critic_value  # (batch size, 1)

    def actor(self, embedding, att_embedding):
        worker_action = torch.cat([embedding, att_embedding], dim=-1)
        worker_action = self.actor_net(worker_action)
        return worker_action

    def critic(self, embedding, att_embedding):
        output = torch.cat([embedding, att_embedding], dim=-1)
        critic_value = self.critic_net(output)
        critic_value = critic_value.mean(1).view(-1)
        return critic_value

    def modify_adjacency(self, adjacency, _device):
        batch_size, n_agents, num_edges, _ = adjacency.shape
        num_nodes = num_edges + 1
        id_tree = torch.arange(0, batch_size * n_agents, device=_device)
        id_nodes = id_tree.view(batch_size, n_agents, 1)
        adjacency[adjacency == -2] = (
            -batch_size * n_agents * num_nodes
        )  # node_idx == -2 invalid node
        adjacency[..., 0] += id_nodes * num_nodes
        adjacency[..., 1] += id_nodes * num_nodes
        adjacency[adjacency < 0] = -2
        return adjacency
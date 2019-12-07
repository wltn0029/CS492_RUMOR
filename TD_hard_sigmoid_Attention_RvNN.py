"""Tree GRU aka Recursive Neural Networks."""

import numpy as np
import torch
import torch.nn.functional as F

from model_library import Node_tweet, gen_nn_inputs, _get_tree_path, hard_sigmoid, RvNNPrototype

dtype = 'float32'
torch_dtype = torch.float32


class RvNN(RvNNPrototype):
    """Data is represented in a tree structure.

    Every leaf and internal node has a data (provided by the input)
    and a memory or hidden state.  The hidden state is computed based
    on its own data and the hidden states of its children.  The
    hidden state of leaves is given by a custom init function.

    The entire tree's embedding is represented by the final
    state computed at the root.

    """

    def __init__(self, word_dim, hidden_dim=5, Nclass=4,
                degree=2, momentum=0.9,
                 trainable_embeddings=True,
                 labels_on_nonroot_nodes=False,
                 irregular_tree=True,
                 device='cpu'):
        super().__init__(word_dim, hidden_dim=hidden_dim, Nclass=Nclass,
                degree=degree, momentum=momentum,
                 trainable_embeddings=trainable_embeddings,
                 labels_on_nonroot_nodes=labels_on_nonroot_nodes,
                 irregular_tree=irregular_tree,
                 device=device)
        self.attention_fn, self.output_fn = self.create_output_fn()

    def predict_up(self, x_word, x_index, num_parent, tree):
       # similar to forward function.
       # except loss, gradient part.
        tree_states = self.compute_tree(x_word, x_index, num_parent, tree)
        final_state = self.attention_fn(tree_states)
        pred_y = self.output_fn(final_state)
        return pred_y

    def create_output_fn(self):
        self.W_out = torch.tensor(self.init_matrix([self.Nclass, self.hidden_dim]), requires_grad = True, device=self.device)
        self.b_out = torch.tensor(self.init_vector([self.Nclass]),requires_grad = True, device=self.device)
        self.params.extend([self.W_out, self.b_out])

        self.W_attention = torch.tensor(self.init_matrix([self.hidden_dim, 1]), requires_grad = True, device=self.device)
        self.b_attention = torch.tensor(self.init_vector([1]),requires_grad = True, device=self.device)
        self.params.extend([self.W_attention, self.b_attention])

        def attention_fn(tree_states):
            attention = F.softmax(hard_sigmoid(torch.matmul(tree_states, self.W_attention) + self.b_attention), dim=0)
            return torch.matmul(tree_states.T, attention)

        def fn(final_state):
            return F.softmax(torch.matmul(self.W_out, final_state).reshape(-1) + self.b_out, dim=0)
        return attention_fn, fn


    def create_recursive_unit(self):
        self.E = torch.tensor(self.init_matrix([self.hidden_dim, self.word_dim]),requires_grad = True, device=self.device)
        self.W_z = torch.tensor(self.init_matrix([self.hidden_dim, self.hidden_dim]),requires_grad = True, device=self.device)
        self.U_z = torch.tensor(self.init_matrix([self.hidden_dim, self.hidden_dim]),requires_grad = True, device=self.device)
        self.b_z = torch.tensor(self.init_vector([self.hidden_dim]),requires_grad = True, device=self.device)
        self.W_r = torch.tensor(self.init_matrix([self.hidden_dim, self.hidden_dim]),requires_grad = True, device=self.device)
        self.U_r = torch.tensor(self.init_matrix([self.hidden_dim, self.hidden_dim]),requires_grad = True, device=self.device)
        self.b_r = torch.tensor(self.init_vector([self.hidden_dim]),requires_grad = True, device=self.device)
        self.W_h = torch.tensor(self.init_matrix([self.hidden_dim, self.hidden_dim]),requires_grad = True, device=self.device)
        self.U_h = torch.tensor(self.init_matrix([self.hidden_dim, self.hidden_dim]),requires_grad = True, device=self.device)
        self.b_h = torch.tensor(self.init_vector([self.hidden_dim]),requires_grad = True, device=self.device)
        self.params.extend([self.E, self.W_z, self.U_z, self.b_z, self.W_r, self.U_r, self.b_r, self.W_h, self.U_h, self.b_h])
        def unit(word, index, parent_h):
            child_xe = torch.matmul(self.E[:,index],torch.tensor(word, device=self.device))
            z = hard_sigmoid(torch.matmul(self.W_z, child_xe) +
                             torch.matmul(self.U_z, parent_h) +
                             self.b_z)
            r = hard_sigmoid(torch.matmul(self.W_r, child_xe) +
                             torch.matmul(self.U_r, parent_h) +
                             self.b_r)
            c = torch.tanh(torch.matmul(self.W_h, child_xe) +
                           torch.matmul(self.U_h, parent_h * r) +
                           self.b_h)
            h = z*parent_h + (1-z)*c
            return h
        return unit

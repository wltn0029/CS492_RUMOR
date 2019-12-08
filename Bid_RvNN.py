__doc__ = """Tree GRU aka Recursive Neural Networks."""

import numpy as np
#import theano
#from theano import tensor as T
#from collections import OrderedDict
#from theano.compat.python2x import OrderedDict
#from theano.tensor.signal.pool import pool_2d
import torch
import torch.nn.functional as F

dtype = 'float32'
torch_dtype = torch.float32
torch_dtype_d = torch.float64

def hard_sigmoid(x):
    """
    Computes element-wise hard sigmoid of x.
    See e.g. https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py#L279
    """
    x = (0.2 * x) + 0.5
    x = F.threshold(-x, -1, -1)
    x = F.threshold(-x, 0, 0)
    return x

################################ tree rnn class ######################################
class RvNN(object):
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
        assert word_dim > 1 and hidden_dim > 1
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.Nclass = Nclass
        self.degree = degree
        self.momentum = momentum
        self.irregular_tree = irregular_tree
        self.device=device
        self.params = []

        torch.autograd.set_detect_anomaly(True)

        self.output_fn = self.create_output_fn()
        self.td_recursive_unit = self.create_td_recursive_unit()
        self.bu_recursive_unit = self.create_bu_recursive_unit()
        self.leaf_unit = self.create_leaf_unit()
        
    def forward(self, x_word, x_index, num_parent, tree,y,lr):
        td_final_state = self.td_compute_tree(x_word[0], x_index[0], num_parent, tree[0]).type(torch_dtype)
        bu_final_state = self.bu_compute_tree(x_word[1], x_index[1], num_parent, tree[1]).type(torch_dtype)
        #print(td_final_state.dtype, bu_final_state.dtype)
        final_state = torch.cat((td_final_state, bu_final_state), dim=0)
        pred_y = self.output_fn(final_state)
        loss = self.loss_fn(y, pred_y)
        self.gradient_descent(loss, lr)

        return loss, pred_y.tolist()

    def predict_up(self, x_word, x_index, num_parent, tree):
        # similar to forward function.
        # except loss, gradient part.\
        td_final_state = self.td_compute_tree(x_word[0], x_index[0], num_parent, tree[0]).type(torch_dtype)
        bu_final_state = self.bu_compute_tree(x_word[1], x_index[1], num_parent, tree[1]).type(torch_dtype)
        final_state = torch.cat((td_final_state, bu_final_state), dim=0)
        pred_y = self.output_fn(final_state)
        return pred_y.tolist()

    def init_matrix(self, shape):
        return np.random.normal(scale=0.1, size=shape).astype(dtype)

    def init_vector(self, shape):
        return np.zeros(shape, dtype=dtype)

    def create_output_fn(self):
        self.W_out = torch.tensor(self.init_matrix([self.Nclass, self.hidden_dim*2]), requires_grad = True, device=self.device)
        self.b_out = torch.tensor(self.init_vector([self.Nclass]),requires_grad = True, device=self.device)
        self.params.extend([self.W_out, self.b_out])

        def fn(final_state):
            return F.softmax(torch.matmul(self.W_out, final_state)+
                             self.b_out, dim=0)
        return fn

    def create_td_recursive_unit(self):
        self.E_td = torch.tensor(self.init_matrix([self.hidden_dim, self.word_dim]),requires_grad = True, device=self.device)
        self.W_z_td = torch.tensor(self.init_matrix([self.hidden_dim, self.hidden_dim]),requires_grad = True, device=self.device)
        self.U_z_td = torch.tensor(self.init_matrix([self.hidden_dim, self.hidden_dim]),requires_grad = True, device=self.device)
        self.b_z_td = torch.tensor(self.init_vector([self.hidden_dim]),requires_grad = True, device=self.device)
        self.W_r_td = torch.tensor(self.init_matrix([self.hidden_dim, self.hidden_dim]),requires_grad = True, device=self.device)
        self.U_r_td = torch.tensor(self.init_matrix([self.hidden_dim, self.hidden_dim]),requires_grad = True, device=self.device)
        self.b_r_td = torch.tensor(self.init_vector([self.hidden_dim]),requires_grad = True, device=self.device)
        self.W_h_td = torch.tensor(self.init_matrix([self.hidden_dim, self.hidden_dim]),requires_grad = True, device=self.device)
        self.U_h_td = torch.tensor(self.init_matrix([self.hidden_dim, self.hidden_dim]),requires_grad = True, device=self.device)
        self.b_h_td = torch.tensor(self.init_vector([self.hidden_dim]),requires_grad = True, device=self.device)
        self.params.extend([self.E_td, self.W_z_td, self.U_z_td, self.b_z_td, self.W_r_td, self.U_r_td, \
                self.b_r_td, self.W_h_td, self.U_h_td, self.b_h_td])
        
        def unit(child_word, child_index, parent_h):
            child_xe = self.E_td[:, child_index].mul(torch.tensor(child_word, device=self.device)).sum(dim=1)
            z_td = hard_sigmoid(self.W_z_td.mul(child_xe).sum(dim=1) + self.U_z_td.mul(parent_h).sum(dim=1) + self.b_z_td)
            r_td = hard_sigmoid(self.W_r_td.mul(child_xe).sum(dim=1) + self.U_r_td.mul(parent_h).sum(dim=1) + self.b_r_td)
            c = torch.tanh(self.W_h_td.mul(child_xe).sum(dim=1) + self.U_h_td.mul(parent_h * r_td).sum(dim=1) + self.b_h_td)
            h_td = z_td * parent_h + (1 - z_td) * c
            return h_td
        return unit

    def td_compute_tree(self, x_word, x_index, num_parent, tree):
        self.num_nodes = x_word.shape[0]
        node_h = torch.tensor(self.init_vector([self.num_nodes, self.hidden_dim]), requires_grad = True, device=self.device)

        # use recurrence to compute internal node hidden states
        def _recurrence(x_word, x_index, node_info, node_h):
            parent_h = node_h[node_info[0]]
            child_h = self.td_recursive_unit(x_word, x_index, parent_h)
            node_h = torch.cat([node_h[:node_info[1]],
                                    child_h.reshape([1, self.hidden_dim]),
                                    node_h[node_info[1]+1:]])
            return node_h, child_h

        child_hs = []
        for x_word_i, x_index_i, tree_i in zip(x_word, x_index, tree) :
            #print(tree_i)
            node_h, child_hs_i =_recurrence(x_word_i, x_index_i, tree_i, node_h)
            child_hs.append(child_hs_i.reshape(1, -1))
        return torch.cat(child_hs[num_parent-1:], dim=0).max(dim=0).values
    
    def create_leaf_unit(self):
        dummy = torch.zeros([self.degree, self.hidden_dim], dtype=torch_dtype, device=self.device)
        def unit(leaf_word, leaf_index):
            return self.bu_recursive_unit( leaf_word, leaf_index, dummy)
        return unit

    def create_bu_recursive_unit(self):
        self.E_bu = torch.tensor(self.init_matrix([self.hidden_dim, self.word_dim]),requires_grad = True, device=self.device)
        self.W_z_bu = torch.tensor(self.init_matrix([self.hidden_dim, self.hidden_dim]),requires_grad = True, device=self.device)
        self.U_z_bu = torch.tensor(self.init_matrix([self.hidden_dim, self.hidden_dim]),requires_grad = True, device=self.device)
        self.b_z_bu = torch.tensor(self.init_vector([self.hidden_dim]),requires_grad = True, device=self.device)
        self.W_r_bu = torch.tensor(self.init_matrix([self.hidden_dim, self.hidden_dim]),requires_grad = True, device=self.device)
        self.U_r_bu = torch.tensor(self.init_matrix([self.hidden_dim, self.hidden_dim]),requires_grad = True, device=self.device)
        self.b_r_bu = torch.tensor(self.init_vector([self.hidden_dim]),requires_grad = True, device=self.device)
        self.W_h_bu = torch.tensor(self.init_matrix([self.hidden_dim, self.hidden_dim]),requires_grad = True, device=self.device)
        self.U_h_bu = torch.tensor(self.init_matrix([self.hidden_dim, self.hidden_dim]),requires_grad = True, device=self.device)
        self.b_h_bu = torch.tensor(self.init_vector([self.hidden_dim]),requires_grad = True, device=self.device)
        self.params.extend([self.E_bu, self.W_z_bu, self.U_z_bu, self.b_z_bu, self.W_r_bu, self.U_r_bu, self.b_r_bu, \
                self.W_h_bu, self.U_h_bu, self.b_h_bu])
        
        def unit(parent_word, parent_index, child_h):
            h_tilde = child_h.sum(dim=0)
            parent_xe = self.E_bu[:, parent_index].mul(torch.tensor(parent_word, device=self.device)).sum(dim=1)
            z_bu = hard_sigmoid(self.W_z_bu.mul(parent_xe).sum(dim=1) + self.U_z_bu.mul(h_tilde).sum(dim=1) + self.b_z_bu)
            r_bu = hard_sigmoid(self.W_r_bu.mul(parent_xe).sum(dim=1) + self.U_r_bu.mul(h_tilde).sum(dim=1) + self.b_r_bu)
            c = torch.tanh(self.W_h_bu.mul(parent_xe).sum(dim=1) + self.U_h_bu.mul(h_tilde * r_bu).sum(dim=1) + self.b_h_bu)
            h_bu = z_bu * h_tilde + (1 - z_bu) * c
            return h_bu
        return unit

    def bu_compute_tree(self, x_word, x_index, num_parents, tree):
        num_nodes = x_word.shape[0]
        if num_nodes == 1:  
            return self.leaf_unit(x_word[0], x_index[0]) 
        num_leaves = num_nodes - num_parents+1

        # compute leaf hidden states
        leaf_h= torch.stack([self.leaf_unit(w, i) for w, i in zip(x_word[:num_leaves], x_index[:num_leaves])], dim=0)
        init_node_h = leaf_h

        # use recurrence to compute internal node hidden states
        def _recurrence(x_word, x_index, tree, idx, node_h):
            #node_h means node's hidden state (start from only leaf and their upper node are added)
            # tree size is num_leaves
            child_exists = tree[:-1]
            # maybe child_h means (tree>-1)th nodes are children of parent(x_word, x_index)
            child_h = node_h[child_exists] 
            # pass all children of one parent to parent node as hidden state
            parent_h = self.bu_recursive_unit(x_word, x_index, child_h) # parent node's hidden state
            node_h = torch.cat((node_h, parent_h.view(1, -1)), 0) # add parent node to input
            return node_h, parent_h

        node_h = init_node_h # only leaf nodes
        # for num_parent step running 
        for idx, (w, x, t) in enumerate(zip(x_word[num_leaves:], x_index[num_leaves:], tree)):
            #node_h means node's hidden state (start from only leaf and their upper node are added)
            #print("BU",t)
            node_h, parent_h=_recurrence(w, x, t, idx, node_h)

        return parent_h

    def loss_fn(self, y, pred_y):

        return torch.sum((torch.tensor(y, device=self.device) - pred_y).pow(2))


    def gradient_descent(self, loss, lr):
        """Momentum GD with gradient clipping."""
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()

        grad = [p.grad for p in self.params]
        self.momentum_velocity_ = [0.] * len(grad)
        grad_norm = torch.sqrt(sum(map(lambda x: x.pow(2).sum(), grad)))
        not_finite = torch.isnan(grad_norm)|torch.isinf(grad_norm)
        scaling_den = max(5.0, grad_norm)
        for n, (param, grad_i) in enumerate(zip(self.params, grad)):
            param.requires_grad = False
            if not_finite :
                grad_i = 0.1 * param
            else :
                grad_i = grad_i * (5.0 / scaling_den)
            velocity = self.momentum_velocity_[n]
            update_step = self.momentum * velocity - lr * grad_i
            self.momentum_velocity_[n] = update_step
            param.add_(update_step)
            param.requires_grad = True
            param.grad.zero_()
            # param = param + update_step

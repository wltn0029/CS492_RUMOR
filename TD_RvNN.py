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

class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.word = []
        self.index = []
        self.parent = None

################################# generate tree structure ##############################
def gen_nn_inputs(root_node, ini_word):
    """Given a root node, returns the appropriate inputs to NN.

    The NN takes in
        x: the values at the leaves (e.g. word indices)
        tree: a (n x degree) matrix that provides the computation order.
            Namely, a row tree[i] = [a, b, c] in tree signifies that a
            and b are children of c, and that the computation
            f(a, b) -> c should happen on step i.

    """
    tree = [[0, root_node.idx]]
    X_word, X_index = [root_node.word], [root_node.index]
    internal_tree, internal_word, internal_index  = _get_tree_path(root_node)
    tree.extend(internal_tree)
    X_word.extend(internal_word)
    X_index.extend(internal_index)
    X_word.append(ini_word)
    ##### debug here #####
    return (np.array(X_word, dtype='float32'),
            np.array(X_index, dtype='int32'),
            np.array(tree, dtype='int32'))

def _get_tree_path(root_node):
    """Get computation order of leaves -> root."""
    if not root_node.children:
        return [], [], []
    layers = []
    layer = [root_node]
    while layer:
        layers.append(layer[:])
        next_layer = []
        [next_layer.extend([child for child in node.children if child])
         for node in layer]
        layer = next_layer
    tree = []
    word = []
    index = []
    for layer in layers:
        for node in layer:
            if not node.children:
               continue
            for child in node.children:
                tree.append([node.idx, child.idx])
                word.append(child.word if child.word is not None else -1)
                index.append(child.index if child.index is not None else -1)

    return tree, word, index

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

        self.output_fn = self.create_output_fn()
        self.recursive_unit = self.create_recursive_unit()

    def forward(self, x_word, x_index, num_parent, tree,y,lr) :
        # tree_states = self.compute_tree(x_word, x_index, num_parent, tree)
        # final_state = torch.max(tree_states, dim=0).values
        final_state = self.compute_tree(x_word, x_index, num_parent, tree)
        pred_y = self.output_fn(final_state)
        loss = self.loss_fn(y, pred_y)
        self.gradient_descent(loss, lr)

        return loss, pred_y

    def predict_up(self, x_word, x_index, num_parent, tree):
       # similar to forward function.
       # except loss, gradient part.
        # tree_states = self.compute_tree(x_word, x_index, num_parent, tree)
        # final_state = torch.max(tree_states, dim=0).values
        final_state = self.compute_tree(x_word, x_index, num_parent, tree)
        pred_y = self.output_fn(final_state)
        return pred_y

    def init_matrix(self, shape):
        return np.random.normal(scale=0.1, size=shape).astype(dtype)

    def init_vector(self, shape):
        return np.zeros(shape, dtype=dtype)

    def create_output_fn(self):
        self.W_out = torch.tensor(self.init_matrix([self.Nclass, self.hidden_dim]), requires_grad = True, device=self.device)
        self.b_out = torch.tensor(self.init_vector([self.Nclass]),requires_grad = True, device=self.device)
        self.params.extend([self.W_out, self.b_out])

        def fn(final_state):
            return F.softmax(torch.matmul(self.W_out, final_state)+
                             self.b_out)
        return fn


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
        """
    def compute_tree(self, x_word, x_index, num_parent, tree):
        num_nodes = x_word.shape[0]
        node_h = torch.tensor(self.init_vector([num_nodes, self.hidden_dim]), device=self.device)
        # use recurrence to compute internal node hidden states
        def _recurrence(x_word, x_index, node_info, node):
            helper_h = torch.ones([num_nodes, self.hidden_dim], device=self.device)
            parent_h = node[node_info[0]]
            child_h = self.recursive_unit(x_word, x_index, parent_h)
            node[node_info[1]].add(child_h)
            return node
        for x_word_i, x_index_i, tree_i in zip(x_word, x_index, tree) :
            #print(tree_i)
            node_h =_recurrence(x_word_i, x_index_i, tree_i, node_h)
        return torch.max(node_h[num_parent:],dim=0).values
    """
    def compute_tree(self, x_word, x_index, num_parent, tree):
        self.num_nodes = x_word.shape[0]
        init_node_h = torch.tensor(self.init_vector([self.num_nodes, self.hidden_dim]), requires_grad = True, device=self.device)

        # use recurrence to compute internal node hidden states
        def _recurrence(x_word, x_index, node_info, node_h):
            parent_h = node_h[node_info[0]]
            child_h = self.recursive_unit(x_word, x_index, parent_h)
            #node_h[node_info[1]] = child_h
            node_h = torch.cat([node_h[:node_info[1]],
                                    child_h.reshape([1, self.hidden_dim]),
                                    node_h[node_info[1]+1:]])
            return node_h, child_h

        child_hs = []
        node_h = init_node_h
        for x_word_i, x_index_i, tree_i in zip(x_word[:-1], x_index, tree) :
            print(tree_i)
            (updated_node_h, child_hs_i)=_recurrence(x_word_i, x_index_i, tree_i, node_h)
            child_hs.append(child_hs_i.reshape(1, -1))
            node_h = updated_node_h
        exit()
        return torch.cat(child_hs[num_parent-1:], 0)
    """
    def compute_tree_test(self, x_word, x_index, tree):
        self.recursive_unit = self.create_recursive_unit()
        def ini_unit(x):
            return torch.tensor(self.init_vector([self.hidden_dim]), device=self.device)
        init_node_h, _ = theano.scan(
            fn=ini_unit,
            sequences=[ x_word ])

        def _recurrence(x_word, x_index, node_info, node_h, last_h):
            parent_h = node_h[node_info[0]]
            child_h = self.recursive_unit(x_word, x_index, parent_h)
            node_h = T.concatenate([node_h[:node_info[1]],
                                    child_h.reshape([1, self.hidden_dim]),
                                    node_h[node_info[1]+1:] ])
            return node_h, child_h

        dummy = torch.tensor(self.init_vector([self.hidden_dim]), device=self.device)
        (_, child_hs), _ = theano.scan(
            fn=_recurrence,
            outputs_info=[init_node_h, dummy],
            sequences=[x_word[:-1], x_index, tree])
        return child_hs
    """
    def loss_fn(self, y, pred_y):

        return torch.sum((torch.tensor(y, device=self.device) - pred_y).pow(2))


    def gradient_descent(self, loss, lr):
        """Momentum GD with gradient clipping."""
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

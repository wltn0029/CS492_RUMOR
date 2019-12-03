__doc__ = """Tree GRU aka Recursive Neural Networks."""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
#from collections import OrderedDict


#torch_dtype = 'float64'
torch_dtype = torch.float32

class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = []
        #self.index = index
        self.idx = idx
        self.word = []
        self.index = []
        #self.height = 1
        #self.size = 1
        #self.num_leaves = 1
        self.parent = None
        #self.label = None

################################# generate tree structure ##############################
def gen_nn_inputs(root_node, max_degree=None, only_leaves_have_vals=True, with_labels=False):
    """Given a root node, returns the appropriate inputs to NN.

    The NN takes in
        x: the values at the leaves (e.g. word indices)
        tree: a (n x degree) matrix that provides the computation order.
            Namely, a row tree[i] = [a, b, c] in tree signifies that a
            and b are children of c, and that the computation
            f(a, b) -> c should happen on step i.

    """
    _clear_indices(root_node)
    #x, leaf_labels = _get_leaf_vals(root_node)
    X_word, X_index = _get_leaf_vals(root_node)
    tree, internal_word, internal_index = _get_tree_traversal(root_node, len(X_word), max_degree)
    #assert all(v is not None for v in x)
    #if not only_leaves_have_vals:
    #    assert all(v is not None for v in internal_x)
    X_word.extend(internal_word)
    X_index.extend(internal_index)
    if max_degree is not None:
        assert all(len(t) == max_degree + 1 for t in tree)

    return (np.array(X_word, dtype='float64'),
            np.array(X_index, dtype='int32'),
            np.array(tree, dtype='int32')) 


def _clear_indices(root_node):
    root_node.idx = None
    [_clear_indices(child) for child in root_node.children if child]


def _get_leaf_vals(root_node):
    """Get leaf values in deep-to-shallow, left-to-right order."""
    all_leaves = []
    layer = [root_node]
    while layer:
        next_layer = []
        for node in layer:
            if not node.children:
                all_leaves.append(node)
            else:
                next_layer.extend([child for child in node.children[::-1] if child])
        layer = next_layer

    X_word = []
    X_index = []
    for idx, leaf in enumerate(reversed(all_leaves)):
        leaf.idx = idx
        X_word.append(leaf.word)
        X_index.append(leaf.index)
        #print idx, leaf
        #print leaf.word
    return X_word, X_index


def _get_tree_traversal(root_node, start_idx=0, max_degree=None):
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
    internal_word = []
    internal_index = []
    idx = start_idx
    for layer in reversed(layers):
        for node in layer:
            if node.idx is not None:
                # must be leaf
                assert all(child is None for child in node.children)
                continue

            child_idxs = [(child.idx if child else -1)
                          for child in node.children]  ## idx of child node
            if max_degree is not None:
                child_idxs.extend([-1] * (max_degree - len(child_idxs)))
            assert not any(idx is None for idx in child_idxs)

            node.idx = idx
            tree.append(child_idxs + [node.idx])
            internal_word.append(node.word if node.word is not None else -1)
            internal_index.append(node.index if node.index is not None else -1)
            idx += 1

    return tree, internal_word, internal_index


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
                 irregular_tree=True):                 
        assert word_dim > 1 and hidden_dim > 1
        #super(RvNN, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.Nclass = Nclass
        self.degree = degree
        self.momentum = momentum
        self.irregular_tree = irregular_tree

        self.params = []
        self.output_fn = self.create_output_fn()
        self.recursive_unit = self.create_recursive_unit()
        
        
    def forward(self, x_word, x_index, tree, y, lr):
        self.num_nodes = x_word.shape[0]
        tree_states = self.compute_tree(x_word, x_index, tree)
        final_state = self.tree_states[-1]
        pred_y = self.create_output_fn(final_state)
        loss = self.loss_fn(y, pred_y)
        self.gradient_descent(loss, lr)
        return loss, pred_y
        
    """
    def evaluate(self, root_node):
        self.tree_states = self.compute_tree(self.x_word, self.x_index, self.tree)
        self.final_state = self.tree_states[-1]
        self._evaluate = theano.function([self.x_word, self.x_index, self.tree], self.final_state)

        x, tree = gen_nn_inputs(root_node, max_degree=self.degree, only_leaves_have_vals=False)
        self._check_input(x, tree)
        return self._evaluate(x, tree[:, :-1])
    """

    def init_matrix(self, shape):
        return np.random.normal(scale=0.1, size=shape).astype(np.float)

    def init_vector(self, shape):
        return np.zeros(shape, dtype=np.float)

    def create_output_fn(self):
        self.W_out = torch.tensor(self.init_matrix([self.Nclass, self.hidden_dim]),requires_grad = True)
        self.b_out = torch.tensor(self.init_vector([self.Nclass]), requires_grad = True)
        self.params.extend([self.W_out, self.b_out])
        def fn(final_state):
            return F.softmax( self.W_out.dot(final_state)+ self.b_out )
        return fn

    def create_recursive_unit(self):
        self.E = torch.tensor(self.init_matrix([self.hidden_dim, self.word_dim]),requires_grad = True)
        self.W_z = torch.tensor(self.init_matrix([self.hidden_dim, self.hidden_dim]),requires_grad = True)
        self.U_z = torch.tensor(self.init_matrix([self.hidden_dim, self.hidden_dim]),requires_grad = True)
        self.b_z = torch.tensor(self.init_vector([self.hidden_dim]),requires_grad = True)
        self.W_r = torch.tensor(self.init_matrix([self.hidden_dim, self.hidden_dim]),requires_grad = True)
        self.U_r = torch.tensor(self.init_matrix([self.hidden_dim, self.hidden_dim]),requires_grad = True)
        self.b_r = torch.tensor(self.init_vector([self.hidden_dim]),requires_grad = True)
        self.W_h = torch.tensor(self.init_matrix([self.hidden_dim, self.hidden_dim]),requires_grad = True)
        self.U_h = torch.tensor(self.init_matrix([self.hidden_dim, self.hidden_dim]),requires_grad = True)
        self.b_h = torch.tensor(self.init_vector([self.hidden_dim]),requires_grad = True)
        self.params.extend([self.E, self.W_z, self.U_z, self.b_z, self.W_r, self.U_r, self.b_r, self.W_h, self.U_h, self.b_h])
        def unit(parent_word, parent_index, child_h, child_exists):
            h_tilde = torch.sum(child_h, dim=0)
            print(parent_word.shape)
            parent_xe = self.E[:,parent_index].dot(parent_word)
            z = Thard_sigmoid(self.W_z.dot(parent_xe)+self.U_z.dot(h_tilde)+self.b_z)
            r = hard_sigmoid(self.W_r.dot(parent_xe)+self.U_r.dot(h_tilde)+self.b_r)
            c = torch.nn.tanh(self.W_h.dot(parent_xe)+self.U_h.dot(h_tilde * r)+self.b_h)
            h = z*h_tilde + (1-z)*c
            return h
        return unit

    def create_leaf_unit(self):
        dummy = torch.zeros([self.degree, self.hidden_dim], dtype=torch_dtype)
        def unit(leaf_word, leaf_index):
            return self.recursive_unit( leaf_word, leaf_index, dummy, dummy.sum(axis=1))
        return unit

    def compute_tree(self, x_word, x_index, tree):
        self.leaf_unit = self.create_leaf_unit()

        num_parents = tree.shape[0]  # num internal nodes
        num_leaves = self.num_nodes - num_parents

        # compute leaf hidden states
        leaf_h= torch.tensor([self.leaf_unit(w, i) for w, i in zip(x_word[:num_leaves], x_index[:num_leaves])])

        if self.irregular_tree:
            init_node_h = torch.cat([leaf_h, leaf_h, leaf_h], axis=0)
        else:
            init_node_h = leaf_h

        # use recurrence to compute internal node hidden states
        def _recurrence(x_word, x_index, node_info, t, node_h, last_h):
            child_exists = node_info > -1
            offset = 2*num_leaves * int(self.irregular_tree) - child_exists * t ### offset???
            child_h = node_h[node_info + offset] * child_exists.dimshuffle(0, 'x') ### transpose??
            parent_h = self.recursive_unit(x_word, x_index, child_h, child_exists)
            node_h = torch.cat([node_h,
                                    parent_h.reshape([1, self.hidden_dim])])
            return node_h[1:], parent_h

        dummy = torch.zeros([self.hidden_dim], dtype=torch_dtype)
        parent_h=[]
        for step in range(num_parents):
            w, x, t = word[num_leaves+step], x_index[num_leaves+step], tree[step] 
            parent_h.append(_recurrence(w,x,t,step,init_node_h, dummy)[1])
        parent_h = torch.tensor(parent_h)
        return torch.cat([leaf_h, parent_h], axis=0)

    def loss_fn(self, y, pred_y):
        return torch.sum((y - pred_y)**2)

    def gradient_descent(self, loss, lr):
        """Momentum GD with gradient clipping."""
        loss.backward()
        grad = [p.grad for p in self.params]
        self.momentum_velocity_ = [0.] * len(grad)
        grad_norm = torch.sqrt(sum(map(lambda x: x.pow(2).sum(), grad)))
        not_finite = torch.isnan(grad_norm)|torch.isinf(grad_norm)
        scaling_den = torch.max(5.0, grad_norm)
        for n, (param, grad_i) in enumerate(zip(self.params, grad)):
            if not_finite:
                grad_i = 0.1 * param
            else:
                grad_i = grad_i * (5.0 / scaling_den)
            velocity = self.momentum_velocity_[n]
            update_step = self.momentum * velocity - lr * grad_i
            self.momentum_velocity_[n] = update_step
            param.add_(update_step)
        
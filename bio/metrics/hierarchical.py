"""Legacy hierarchical-clustering utilities used by archived experiments."""
import numpy as np
# from sklearn.utils.linear_assignment_ import linear_assignment
import os
import argparse
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
import attr
import typing
from typing import Optional
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def cluster_acc(y_true, y_pred, return_index=False):
    """
    Calculate clustering accuracy.
    # Arguments
            y: true labels, numpy.array with shape `(n_samples,)`
            y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
            accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.astype(int).max(), y_true.astype(int).max()) + 1
    w = np.zeros((int(D), (D)), dtype=np.int64)
    for i in range(y_pred.size):
        w[int(y_pred[i]), int(y_true[i])] += 1
    ind = np.array(linear_assignment(w.max() - w))
    if return_index:
        assert all(ind[0] == range(len(ind[0])))  # Assert rows don't change order
        cluster_acc = sum(w[ind[0], ind[1]]) * 1.0 / y_pred.size
        return cluster_acc
    else:
        return (
            sum([w[ind[0, i], ind[1, i]] for i in range(len(ind[0]))])
            * 1.0
            / y_pred.size
        )


@attr.s(cmp=False)
class DpLeaf(object):
    dp_ids = attr.ib()
    node_id = attr.ib()

    @property
    def children(self):
        return []

    @property
    def is_leaf(self):
        return True


@attr.define()
class DpNode(object):
    """
    node_id should be in such a way that a smaller number means split before a larger number in a top-down manner
    That is the root should have node_id = 0 and the children of the last split should have node id
    2*n_dps-2 and 2*n_dps-1

    """

    left_child: typing.Any = None
    right_child: typing.Any = None
    node_id: Optional[int] = None

    @property
    def children(self):
        return [self.left_child, self.right_child]

    @property
    def is_leaf(self):
        return False


def to_dendrogram_purity_tree(children_array):

    tree_node_root = TreeNode()
    tree_node_root.name = "root"
    tree_node_root.father = None


class TreeNode:
    def __init__(
        self, decoder=False, router=None, left=None, right=None, samples=None, name=""
    ):
        self.decoder = decoder  # Whether this node is a leaf node
        self.router = (
            router  # Stores the router condition if this node is an internal split node
        )
        self.left = left  # Left child
        self.right = right  # Right child
        self.samples = (
            samples  # Stores sample indices for leaf nodes (valid only if decoder=True)
        )
        self.father = None  # Father node (optional, if needed)
        self.name = name  # Node name

    def print_tree(self, depth=0):
        # print('name', self.name)
        if self.decoder:
            # import pdb; pdb.set_trace()
            print("leaf node, number of samples:", self.samples.shape[0])
        else:
            print(
                "router node:",
                self.router,
                "left:",
                self.left.name if self.left else None,
                "right:",
                self.right.name if self.right else None,
            )
            if self.left:
                self.left.print_tree(depth + 1)
            if self.right:
                self.right.print_tree(depth + 1)

    def bool_mergeable(self):
        if self.father.left.name == self.name and self.father.right.left is None:
            return True
        elif self.father.right.name == self.name and self.father.left.left is None:
            return True
        else:
            return False

    def list_all_leaf(self):
        if self.decoder:
            return [self]
        else:
            leaves = []
            if self.left:
                leaves.extend(self.left.list_all_leaf())
            if self.right:
                leaves.extend(self.right.list_all_leaf())
            return leaves

    def plot_all_tree_with_matplotlib(self):

        def plot_tree(node, ax, x, y, x_offset, y_offset):
            """
            node     : current node
            ax       : matplotlib Axes object
            x, y     : coordinates (x, y) of the current node on the canvas
            x_offset : horizontal offset for child nodes
            y_offset : vertical offset for child nodes
            """
            if node is None:
                return

            # If node is a leaf, use red; otherwise use blue
            color = "red" if node.decoder else "blue"

            # Plot current node at (x, y)
            ax.text(
                x,
                y,
                node.name,
                ha="center",
                va="center",
                bbox=dict(facecolor=color, alpha=0.5),
            )

            # If there is a left child, draw a line and recurse
            if node.left:
                # Draw a line from current node (x, y) to (x - x_offset, y - y_offset)
                ax.plot([x, x - x_offset], [y, y - y_offset], "k-")
                plot_tree(
                    node.left, ax, x - x_offset, y - y_offset, x_offset * 0.5, y_offset
                )

            # If there is a right child, draw a line and recurse
            if node.right:
                ax.plot([x, x + x_offset], [y, y - y_offset], "k-")
                plot_tree(
                    node.right, ax, x + x_offset, y - y_offset, x_offset * 0.5, y_offset
                )

        # Create a figure and an Axes
        fig, ax = plt.subplots(figsize=(12, 8))  # Adjust according to your needs

        # Start plotting from the root node (self), with given coordinates and offsets
        # x=0, y=0       : root node at the origin
        # x_offset=3     : horizontal offset for the children
        # y_offset=1.5   : vertical offset for the children
        plot_tree(self, ax, x=0, y=0, x_offset=5, y_offset=1.5)

        # Manually set axis limits so that the entire tree is visible
        # Increase or decrease the range if the tree is very large or very small
        ax.set_xlim(-15, 15)
        ax.set_ylim(-10, 5)

        # Alternatively, you could use ax.autoscale() if you prefer automatic scaling,
        # but you may need to remove bbox_inches='tight' to avoid text clipping
        # ax.autoscale(enable=True, axis='both', tight=True)

        # Hide the axes (optional)
        plt.axis("off")

        # Save the figure and close
        # plt.savefig(path, bbox_inches='tight')
        # plt.close(fig)
        return fig

    def output_label_list(self):

        # import pdb; pdb.set_trace()

        num_sample_list = [l.samples.shape[0] for l in reversed(self.list_all_leaf())]
        num_sample_all = np.sum(num_sample_list)
        predict_label = np.zeros(num_sample_all)
        for i, l in enumerate(reversed(self.list_all_leaf())):
            predict_label[l.samples] = i
        return predict_label


def build_tree(cluster_centers, predict_label):

    leaf_nodes = []
    for i in range(cluster_centers.shape[0]):
        # print(f'cluster {i} center:', cluster_centers[i])
        node = TreeNode(decoder=True, name=f"leaf_{i}")
        node.samples = np.where(predict_label == i)[0]
        leaf_nodes.append(node)

    # merge the leaf nodes
    while len(leaf_nodes) > 1:
        # import pdb; pdb.set_trace()
        node1 = leaf_nodes.pop(0)
        node2 = leaf_nodes.pop(0)
        node = TreeNode(decoder=False, name=f"router_{node1.name}_{node2.name}")
        node.left = node1
        node.right = node2
        node1.father = node
        node2.father = node
        leaf_nodes.append(node)

    return leaf_nodes[0]


def count_values_in_sequence(sequence):
    counts = {}
    for value in sequence:
        if value in counts:
            counts[value] += 1
        else:
            counts[value] = 1
    return counts


def dendrogram_purity(tree_root, ground_truth, ind_samples_of_leaves):

    from scipy.special import comb

    total_per_label_frequencies = count_values_in_sequence(ground_truth)
    total_per_label_pairs_count = {
        k: comb(v, 2, repetition=True) for k, v in total_per_label_frequencies.items()
    }
    total_n_of_pairs = sum(total_per_label_pairs_count.values())
    one_div_total_n_of_pairs = 1.0 / total_n_of_pairs
    purity = 0.0

    def calculate_purity(node, level):
        nonlocal purity
        if node.decoder:
            # Match node to leaf samples
            ind_leaf = np.where(
                [
                    node == ind_samples_of_leaves[ind_leaf][0]
                    for ind_leaf in range(len(ind_samples_of_leaves))
                ]
            )[0].item()
            ind_samples_of_leaf = ind_samples_of_leaves[ind_leaf][1]
            node_total_dp_count = len(ind_samples_of_leaf)
            # Count how many samples of given leaf fall into which ground-truth class (-> For treevae make use of ground_truth(to which class a sample belongs)&yy(into which leaf a sample falls))
            node_per_label_frequencies = count_values_in_sequence(
                [ground_truth[id] for id in ind_samples_of_leaf]
            )
            # From above, deduct how many pairs will fall into same leaf
            node_per_label_pairs_count = {
                k: comb(v, 2, repetition=True)
                for k, v in node_per_label_frequencies.items()
            }

        elif node.router is None and node.decoder is None:
            # We are in an internal node with pruned leaves and thus only have one child. Therefore no prunity calculation here!
            node_left, node_right = node.left, node.right
            child = node_left if node_left is not None else node_right
            node_per_label_frequencies, node_total_dp_count = calculate_purity(
                child, level + 1
            )
            return node_per_label_frequencies, node_total_dp_count
        else:
            # it is an inner splitting node
            left_child_per_label_freq, left_child_total_dp_count = calculate_purity(
                node.left, level + 1
            )
            right_child_per_label_freq, right_child_total_dp_count = calculate_purity(
                node.right, level + 1
            )
            node_total_dp_count = left_child_total_dp_count + right_child_total_dp_count
            # Count how many samples of given internal node fall into which ground-truth class (=sum of their children's values)
            node_per_label_frequencies = {
                k: left_child_per_label_freq.get(k, 0)
                + right_child_per_label_freq.get(k, 0)
                for k in set(left_child_per_label_freq)
                | set(right_child_per_label_freq)
            }

            # Class-wisedly count how many pairs of samples of a class will have this node as least common ancestor (=mult. of their children's values, bcs this is all possible pairs coming from different sides)
            node_per_label_pairs_count = {
                k: left_child_per_label_freq.get(k) * right_child_per_label_freq.get(k)
                for k in set(left_child_per_label_freq)
                & set(right_child_per_label_freq)
            }

        # Given the class-wise number of pairs with given node as least common ancestor node, calculate their purity
        for label, pair_count in node_per_label_pairs_count.items():
            label_freq = node_per_label_frequencies[label]
            label_pairs = node_per_label_pairs_count[label]
            purity += (
                one_div_total_n_of_pairs
                * label_freq
                / node_total_dp_count
                * label_pairs
            )  # (1/n_all_pairs) * purity(=n_samples_of_this_class_in_node/n_samples) * n_class_pairs_with_this_node_being_least_common_ancestor(this last term represents sum over pairs with this node being least common ancestor)
        return node_per_label_frequencies, node_total_dp_count

    calculate_purity(tree_root, 0)
    return purity


def leaf_purity(tree_root, ground_truth, ind_samples_of_leaves):
    values = []  # purity rate per leaf
    weights = []  # n_samples per leaf

    # For each leaf calculate the maximum over classes for in-leaf purity (i.e. majority class / n_samples_in_leaf)
    def get_leaf_purities(node):
        nonlocal values
        nonlocal weights
        if node.decoder:
            ind_leaf = np.where(
                [
                    node == ind_samples_of_leaves[ind_leaf][0]
                    for ind_leaf in range(len(ind_samples_of_leaves))
                ]
            )[0].item()
            ind_samples_of_leaf = ind_samples_of_leaves[ind_leaf][1]
            node_total_dp_count = len(ind_samples_of_leaf)
            node_per_label_counts = count_values_in_sequence(
                [ground_truth[id] for id in ind_samples_of_leaf]
            )
            if node_total_dp_count > 0:
                purity_rate = max(node_per_label_counts.values()) / node_total_dp_count
            else:
                purity_rate = 1.0
            values.append(purity_rate)
            weights.append(node_total_dp_count)
        elif node.router is None and node.decoder is None:
            # We are in an internal node with pruned leaves and thus only have one child.
            node_left, node_right = node.left, node.right
            child = node_left if node_left is not None else node_right
            get_leaf_purities(child)
        else:
            get_leaf_purities(node.left)
            get_leaf_purities(node.right)

    get_leaf_purities(tree_root)
    assert len(values) == len(
        ind_samples_of_leaves
    ), "Didn't iterate through all leaves"
    # Return mean leaf_purity
    return np.average(values, weights=weights)

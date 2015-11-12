import logging
from scipy.cluster.hierarchy import linkage
from operator import attrgetter
from abc import abstractmethod
import numpy as np

logger = logging.getLogger(__name__)


def disjoint_subtrees(trees):
    """
    Given a selected set of subtrees, return the largest possible set of
    them, spanning the most nodes, that are disjoint from each other.

    The strategy for this is:

    - Order the trees from children to parents
    - For each tree in order:
      - Find its descendants in the `accepted` list
      - If there is 1 of them, kick it out of the `accepted` list
      - If there are now no descendants of this tree in the `accepted`
        list, add this tree to it
    """
    ordered_trees = sorted(trees, reverse=True)
    accepted = []
    for tree in ordered_trees:
        descendants = []
        for prevtree in accepted:
            if tree.is_ancestor_of(prevtree):
                descendants.append(prevtree)
        if len(descendants) == 1:
            accepted.remove(descendants[0])
        if len(descendants) <= 1:
            accepted.append(tree)
    return accepted[::-1]


class RepNode:
    """
    An abstract base class, implemented by RepTree (for non-terminal nodes)
    and RepLeaf (for terminal nodes).

    Each RepLeaf and RepTree defines the following properties:

    - `term`: a term dictionary that serves as a label for the node.

    - `termlist`: a list of terms that the node contains, in descending order
      of relevance.

    - `filtered_termlist`: the `termlist` with some terms excluded based on
      the overall tree structure.

    - `size`: the number of terms that the node contains.

    - `score`: a representation of how good a split the node represents. A
      node and its sibling will have the same score.

    - `tree_score`: the maximum score of this node and any of its descendants.
      This helps to avoid recursing into irrelevant subtrees.

    - `dist`: the diameter of this node, according to the clustering method.
      Leaves have a diameter of 0.

    - `left_index` and `right_index`: the span of leaf indices that the node
      contains, where the leaves are numbered from left to right. Once these
      indices are set, it will always be the case that
      `right_index - left_index == size`.

    Many of these properties are not set on initialization; they're only
    set once the entire tree has been built and `.assign_labels()` has been
    run on it recursively. Such properties, of course, can't be used within
    `.assign_labels()`.
    """
    @abstractmethod
    def assign_labels(self, left_index, exclude):
        raise NotImplementedError

    @abstractmethod
    def _show_tree_lines(self, max_depth, min_score):
        raise NotImplementedError

    def is_ancestor_of(self, other: 'RepNode'):
        """
        Determine whether self is an ancestor of other. This uses the
        data in .left_index and .right_index, so the tree must have already
        had .assign_labels() run on them.
        """
        if self.left_index is None or other.left_index is None:
            raise RuntimeError(
                ".is_ancestor_of() may not be used on nodes that have not run "
                ".assign_labels() yet."
            )
        return (
            self.left_index <= other.left_index
            and self.right_index >= other.right_index
        )

    def all_subtrees(self):
        """
        Return all subtrees (but not leaves).
        """
        return []

    def show_tree(self, max_depth=10, min_score=1, list_items=5):
        """
        Get a line-by-line representation of the tree, and join it into a
        single string representation.
        """
        return '\n'.join(self._show_tree_lines(max_depth, min_score))

    # Compare nodes by identity. This should work as long as we never have
    # multiple copies of the same tree.
    def __hash__(self):
        return hash(id(self))

    def __lt__(self, other):
        """
        Once assign_labels has been run, nodes can be compared to each other.
        They're ordered from parents to children, left to right.
        """
        if self.left_index is None or other.left_index is None:
            raise ValueError("Can't compare trees before .assign_labels()")
        return (self.left_index, -self.right_index) < (other.left_index, -other.right_index)

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other

    def __str__(self):
        """
        Show the tree's string representation with default settings.
        """
        return self.show_tree()


class RepLeaf(RepNode):
    """
    A RepLeaf contains a single term, which acts as its label. This term is
    exempt from the requirement that it doesn't duplicate the label of a
    parent node, for two reasons:

    - It likely does duplicate a parent node
    - This is the ground truth of where terms come from

    The filtered_termlist of a RepLeaf is unlikely to be useful, so we leave
    it empty.
    """
    def __init__(self, term: dict):
        self.term = term
        self.termlist = [term]
        self.filtered_termlist = []
        self.size = 1
        self.score = 1
        self.tree_score = 1
        self.dist = 0
        self.left_index = None
        self.right_index = None

    def assign_labels(self, left_index: int, exclude: frozenset):
        """
        We don't need a term label, but we do need to set our `left_index`
        and `right_index`.
        """
        self.left_index = left_index
        self.right_index = left_index + 1

    def node_str(self, list_items=15):
        """
        Visually distinguish leaves by marking them with an asterisk.
        """
        return '* %s (%2.2f)' % (self.term['text'], self.score)

    def _show_tree_lines(self, max_depth, min_score):
        """
        If the `min_score` is low enough, we'll show leaves.
        """
        if self.score >= min_score:
            return [self.node_str()]
        else:
            return []

    def __repr__(self):
        return "<RepLeaf: %r>" % self.term['text']


class RepTree(RepNode):
    def __init__(self, left: RepNode, right: RepNode, dist: float=0.):
        """
        A RepTree is a node of a binary tree, so it branches into a left and
        right RepNode.
        """
        # Store the left and right nodes
        self.left = left
        self.right = right

        # Our position in the tree will be determined later
        self.left_index = None
        self.right_index = None

        # Calculate the total size of this tree
        self.size = self.left.size + self.right.size

        # Set our distance (diameter) to what the clustering method says it is
        self.dist = dist

        # We don't know our own score until we have a sibling, or become the
        # root node.
        self.score = 0
        self.tree_score = 0

        # Find the score of our two children, and set it on them.
        child_score = (
            min(self.left.size, self.right.size) + self.dist
            - max(self.left.dist, self.right.dist)
        )
        self.left.score = self.right.score = child_score

        # We don't know our labels until we run `assign_labels`.
        self.term = None
        self.filtered_termlist = []

        # Get the complete list of terms that this tree encompasses, and sort
        # them by relevance and similarity to the mean of this cluster.
        self.termlist = self.left.termlist + self.right.termlist
        vectors = np.vstack([term['vector'] for term in self.termlist])
        term_scores = {}
        for term in self.termlist:
            median_similarity = np.median(vectors.dot(term['vector']))
            term_scores[term['term']] = term['relevance'] * median_similarity

        self.termlist.sort(key=lambda term: term_scores[term['term']], reverse=True)

    def assign_labels(self, left_index: int, exclude: frozenset=frozenset()):
        """
        Recursively walk through this tree, assigning labels to every
        non-terminal node, and assigning tree spans to every node.

        The key restriction on these labels is that they cannot duplicate the
        label of a parent node. `exclude` keeps track of the labels that are
        already claimed by parents.
        """
        # Find the most relevant non-excluded term, and claim it as a label.
        for term in self.termlist:
            if term['text'] not in exclude:
                self.term = term
                break

        # Some nodes will remain unlabeled, because all their possible terms
        # have been excluded by parent nodes.
        #
        # `exclude_next` is the new set of terms that will be excluded from
        # this node's children. Add our own label if we have one.
        if self.term is None:
            exclude_next = exclude
        else:
            exclude_next = exclude | {self.term['text']}

        # Get our list of additional terms, which is our termlist minus all
        # those that have already been used as labels. These terms remain in
        # relevance order, and indicate how to use this node as a topic.
        self.filtered_termlist = [
            term for term in self.termlist
            if term['text'] not in exclude_next
        ]

        # Recursively assign labels to all descendants.
        self.left_index = left_index
        self.left.assign_labels(left_index, exclude_next)
        self.right.assign_labels(self.left.right_index, exclude_next)
        self.right_index = self.right.right_index

        # While we're recursing, calculate our tree_score based on our
        # children's tree_scores.
        self.tree_score = max(
            self.left.tree_score, self.right.tree_score, self.score
        )

    def node_str(self, list_items=5):
        """
        Get a string representation of just this node.
        """
        termtext = ''
        if self.term is not None:
            termtext = self.term['text']
        shown_terms = [repr(term['text']) for term in self.filtered_termlist[:list_items]]
        if len(self.filtered_termlist) > list_items:
            shown_terms.append('...')
        terms_inner = ', '.join(shown_terms)
        line = '- %s (%2.2f) [%s]' % (termtext, self.score, terms_inner)
        return line

    def _show_tree_lines(self, max_depth, min_score, list_items=5):
        """
        Build a line-by-line text representation of the tree, showing only
        the nodes that meet certain criteria.
        """
        # Keep track of whether anything from our children is visible.
        lines = []

        # We can descend into any child with a high enough tree_score, as
        # long as we haven't reached the maximum depth.
        #
        # Subtract 1 from max_depth as we recurse, and keep track of whether
        # we got any recursive results.
        if max_depth > 0:
            if self.left.tree_score >= min_score:
                more = self.left._show_tree_lines(max_depth - 1, min_score)
                if more:
                    lines.extend(more)
            if self.right.tree_score >= min_score:
                more = self.right._show_tree_lines(max_depth - 1, min_score)
                if more:
                    lines.extend(more)

        # Should we show this node itself? There are some intermediate nodes
        # that we want to skip because their own score is too low. This will
        # tend to occur when there's only one interesting child, so we'll just
        # show that single child without any additional indentation.
        if self.score >= min_score:
            first_line = self.node_str(list_items)

            # Prepend the first line, and indent the children by two spaces.
            lines = [first_line] + ['  ' + line for line in lines]

        return lines

    def all_subtrees(self):
        """
        Return all subtrees (but not leaves).
        """
        return [self] + self.left.all_subtrees() + self.right.all_subtrees()

    def flat_topic_list(self, count=15):
        trees = self.all_subtrees()
        trees.sort(key=attrgetter('score'), reverse=True)

        topics = []
        for tree in trees:
            if tree.term is not None:
                topics.append(tree)
                disjoint_topics = disjoint_subtrees(topics)
                if len(disjoint_topics) >= count:
                    return disjoint_topics[:count]

        return disjoint_subtrees(topics)

    def describe(self, list_items=3):
        shown_terms = [self.term['text']] + [term['text'] for term in self.filtered_termlist[:list_items - 1]]
        return ', '.join(shown_terms)

    def __str__(self):
        return '%s (score=%2.2f)' % (self.describe(), self.score)

    def __repr__(self):
        return "<RepTree with score %2.2f, labeled %r>" % (self.score, self.term['text'])

    @staticmethod
    def from_term_list(terms):
        """
        Build a RepTree from a list of term dictionaries.
        """
        count = len(terms)
        term_mat = np.array([term['vector'] for term in terms])

        # Build linkage matrix
        logger.info('Building linkage matrix')
        link_mat = linkage(term_mat, metric='cosine', method='average')

        nodes = []

        # Start by building leaf objects
        leaves = [RepLeaf(term) for term in terms]

        for i, j, dist, n in link_mat:
            # Convert NumPy types to base Python types
            i = int(i)
            j = int(j)
            dist = float(dist)
            n = int(n)

            # Indices that are greater than the number of leaves refer to
            # previously-built nodes
            i_node = leaves[i] if i < count else nodes[i - count]
            j_node = leaves[j] if j < count else nodes[j - count]

            # Build the node and add it to our indexable list of nodes
            node = RepTree(i_node, j_node, dist)
            assert node.size == n
            nodes.append(node)

        # The last node we built represents the entire tree.
        tree = nodes[-1]
        assert tree.size == count

        # Now that all nodes are built, assign labels throughout the tree.
        tree.assign_labels(0)
        assert tree.left_index == 0
        assert tree.right_index == tree.size

        # Give the root node a high score so that it's visible.
        tree.score = tree.size

        return tree

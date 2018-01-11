"""
This module provides an implementation of hierarchical agglomerative clustering
that also takes relevance into account.  Clusters are determined based on the
distance between the vectors of terms, and they are labeled based on the
relevance of terms in the cluster.  You can make a tree by passing a list of
terms to ClusterTree.from_term_list(), and you can get out a list of non-
overlapping clusters by using the flat_cluster_list() method of that tree.
"""


from operator import attrgetter, itemgetter
from scipy.cluster.hierarchy import linkage


class ClusterNode:
    """
    An abstract base class, implemented by ClusterTree (for non-terminal nodes)
    and ClusterLeaf (for terminal nodes).

    Each ClusterLeaf and ClusterTree defines the following properties:

    - `term`: a term dictionary that serves as a label for the node.

    - `termlist`: a list of terms that the node contains, in descending order
      of relevance.

    - `filtered_termlist`: the `termlist` with some terms excluded based on the
      overall tree structure.

    - `size`: the number of terms that the node contains.

    - `score`: a representation of how good a split the node represents. A node
      and its sibling will have the same score.

    - `dist`: the diameter of this node, according to the clustering method.
      Leaves have a diameter of 0.

    - `left_index` and `right_index`: the span of leaf indices that the node
      contains, where the leaves are numbered from left to right.  Once these
      indices are set, it will always be the case that
      `right_index - left_index == size`.

    Many of these properties are not set on initialization; they're only set
    once the entire tree has been built and `assign_labels()` has been run on
    it recursively.  Such properties, of course, can't be used within
    `assign_labels()`.
    """
    def assign_labels(self, left_index, exclude):
        raise NotImplementedError

    def _show_tree_lines(self, max_depth, min_score):
        raise NotImplementedError

    def all_subtrees(self):
        raise NotImplementedError

    def is_ancestor_of(self, other):
        """
        Determine whether self is an ancestor of other.  This uses the data in
        left_index and right_index, so the tree must have already had
        assign_labels() run on them.
        """
        if self.left_index is None or other.left_index is None:
            raise RuntimeError(
                "is_ancestor_of() may not be used on nodes that have not run "
                "assign_labels() yet."
            )
        return (
            self.left_index <= other.left_index and
            self.right_index >= other.right_index
        )

    def show_tree(self, max_depth=10, min_score=1, list_items=5):
        """
        Get a line-by-line representation of the tree, and join it into a
        single string representation.
        """
        return '\n'.join(self._show_tree_lines(max_depth, min_score))

    def __str__(self):
        """
        Show the tree's string representation with default settings.
        """
        return self.show_tree()


class ClusterLeaf(ClusterNode):
    """
    A ClusterLeaf contains a single term, which acts as its label.  This term
    is exempt from the requirement that it doesn't duplicate the label of a
    parent node, for two reasons:

    - It likely does duplicate a parent node.
    - This is the ground truth of where terms come from.

    The `filtered_termlist` of a ClusterLeaf is unlikely to be useful, so we
    leave it empty.
    """
    def __init__(self, term):
        self.term = term
        self.termlist = [term]
        self.filtered_termlist = []
        self.size = 1
        self.score = 1
        self.dist = 0
        self.left_index = None
        self.right_index = None

    def assign_labels(self, left_index, exclude):
        """
        We don't need a term label, but we do need to set our `left_index` and
        `right_index`.
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

    def all_subtrees(self):
        return []

    def __repr__(self):
        return "<ClusterLeaf: %r>" % self.term['text']


class ClusterTree(ClusterNode):
    def __init__(self, left, right, dist):
        """
        A ClusterTree is a node of a binary tree, so it branches into a left
        and right ClusterNode.
        """
        # Store the left and right nodes
        self.left = left
        self.right = right

        # Our position in the tree will be determined later
        self.left_index = None
        self.right_index = None

        # Calculate the total size of this tree.  This is the number of
        # *leaves* in the tree, not the number of nodes.
        self.size = self.left.size + self.right.size

        # Set our distance (diameter) to what the clustering method says it is
        self.dist = dist

        # We don't know our own score until we have a sibling, or become the
        # root node.
        self.score = 0

        # Find the score of our two children, and set it on them.
        child_score = (
            min(self.left.size, self.right.size) + self.dist -
            max(self.left.dist, self.right.dist)
        )
        self.left.score = self.right.score = child_score

        # We don't know our labels until we run `assign_labels()`.
        self.term = None
        self.filtered_termlist = []

        # Get the complete list of terms that this tree encompasses, and sort
        # them by relevance.
        self.termlist = self.left.termlist + self.right.termlist
        self.termlist.sort(key=itemgetter('score'), reverse=True)

    def assign_labels(self, left_index=0, exclude=frozenset()):
        """
        Recursively walk through this tree, assigning labels to every
        non-terminal node, and assigning tree spans to every node.

        The key restriction on these labels is that they cannot duplicate the
        label of a parent node.  `exclude` keeps track of the labels that are
        already claimed by ancestors.
        """
        # Find the most relevant non-excluded term, and claim it as a label.
        for term in self.termlist:
            if term['text'] not in exclude:
                self.term = term
                break

        # Some nodes will remain unlabeled, because all their possible terms
        # have been excluded by parent nodes.  If we have a label, though, add
        # it to `exclude`, so it will be excluded from the node's children.
        if self.term is not None:
            exclude = exclude | {self.term['text']}

        # Get our list of additional terms, which is our termlist minus all
        # those that have already been used as labels.  These terms remain in
        # relevance order, and indicate how to use this node as a topic.
        self.filtered_termlist = [term for term in self.termlist
                                  if term['text'] not in exclude]

        # Recursively assign labels to all descendants.
        self.left_index = left_index
        self.left.assign_labels(left_index=left_index, exclude=exclude)
        self.right.assign_labels(left_index=self.left.right_index,
                                 exclude=exclude)
        self.right_index = self.right.right_index

    def node_str(self, list_items=5):
        """
        Get a string representation of just this node.
        """
        termtext = self.term['text'] if self.term is not None else ''
        termlist = self.filtered_termlist
        # The repr() here is just to put the text in quotes
        shown_terms = [repr(term['text']) for term in termlist[:list_items]]
        if len(termlist) > list_items:
            shown_terms.append('...')
        terms_inner = ', '.join(shown_terms)
        return '- %s (%2.2f) [%s]' % (termtext, self.score, terms_inner)

    def _show_tree_lines(self, max_depth, min_score, list_items=5):
        """
        Build a line-by-line text representation of the tree, showing only the
        nodes that meet certain criteria.
        """
        # Get the lines (if any) to display for children
        lines = []
        if max_depth > 0:
            for tree in (self.left, self.right):
                lines.extend(tree._show_tree_lines(max_depth - 1, min_score))

        # Should we show this node itself?  There are some intermediate nodes
        # that we want to skip because their own score is too low.  This will
        # tend to occur when there's only one interesting child, so we'll just
        # show that single child without any additional indentation.
        if self.score >= min_score:
            # Prepend the first line, and indent the children by two spaces.
            first_line = self.node_str(list_items)
            lines = [first_line] + ['  ' + line for line in lines]

        return lines

    def all_subtrees(self):
        """
        Return all subtrees (but not leaves).
        """
        return [self] + self.left.all_subtrees() + self.right.all_subtrees()

    def flat_cluster_list(self, count=15):
        """
        Return the important clusters in the tree.

        Specifically, each cluster is a disjoint subtree from among the top
        scoring subtrees; larger subtrees are preferred when possible.  The
        specified number of clusters is generated while considering as few top
        scoring subtrees as possible; fewer clusters than specified may be
        returned if the tree is not large enough.
        """
        subtrees = self.all_subtrees()
        subtrees.sort(key=attrgetter('score'), reverse=True)

        # The maximum number of clusters arises when there is one for each
        # subtree with no descendants yet under consideration.  Although we
        # eventually want to replace these subtrees with their ancestors when
        # possible, it's easiest to track these and handle the ancestors later.
        ancestors = []
        clusters = []
        for subtree in subtrees:
            if subtree.term is not None:
                for i, cluster in enumerate(clusters):
                    if subtree.is_ancestor_of(cluster):
                        # This subtree is an ancestor; just hold it
                        ancestors.append(subtree)
                        break
                    if cluster.is_ancestor_of(subtree):
                        # This subtree is a descendant; replace its ancestor
                        ancestors.append(cluster)
                        clusters[i] = subtree
                        break
                else:
                    # This subtree is a brand new cluster
                    clusters.append(subtree)
                    if len(clusters) >= count:
                        break

        # Now go back and replace only children with their ancestors
        for ancestor in ancestors:
            descendants = [c for c in clusters if ancestor.is_ancestor_of(c)]
            if len(descendants) == 1:
                clusters[clusters.index(descendants[0])] = ancestor

        # For backwards compatibility, sort left to right
        clusters.sort(key=attrgetter('left_index'))
        return clusters

    def __repr__(self):
        label = self.term['text'] if self.term is not None else '---'
        return ("<ClusterTree with score %2.2f, labeled %r>"
                % (self.score, label))

    @staticmethod
    def from_term_list(terms):
        """
        Build a ClusterTree from a list of term dictionaries.  These term
        dictionaries should have at least a 'text', 'score' (relevance), and
        'vector' (unpacked) fields, and should probably also have a 'term'
        field, though it's not strictly necessary.  This is our main method of
        constructing ClusterTrees.
        """
        # SciPy catches these two conditions, but with really cryptic errors
        if len(terms) < 2:
            raise ValueError('Cannot cluster fewer than two terms.')

        vectors = [term['vector'] for term in terms]
        # We check for this particular condition because terms with no
        # associations in Analytics end up with None in 'vector'; it can't just
        # be "if None in vectors" because of a NumPy warning
        if any(v is None for v in vectors):
            raise ValueError('Cannot cluster terms without vectors.')

        # Build linkage matrix
        link_mat = linkage(vectors, metric='cosine', method='average')

        # Iterate through the linkage matrix to determine which nodes to join.
        # Each row in the matrix represents a tree node, and contains:
        #   * Indexes into the node list for the node's children; indexes less
        #     than the length of the input refer to leaf nodes, and the rest to
        #     previous entries in the matrix.
        #   * The distance between the node's children; this should be
        #     monotonically increasing.
        #   * The number of leaves under the node.
        # See the SciPy documentation for further information.
        nodes = [ClusterLeaf(term) for term in terms]
        for i, j, dist, n in link_mat:
            # Convert NumPy types to regular Python types
            node = ClusterTree(nodes[int(i)], nodes[int(j)], float(dist))
            assert node.size == int(n)
            nodes.append(node)

        # The last node we built represents the entire tree.
        tree = nodes[-1]
        assert tree.size == len(terms)

        # Now that all nodes are built, assign labels throughout the tree.
        tree.assign_labels()
        assert tree.left_index == 0
        assert tree.right_index == tree.size

        # Give the root node a high score so that it gets listed first.
        tree.score = tree.size

        return tree


def cluster_term_dicts(terms, max_clusters):
    '''
    Main interface for the clustering module.  Pass a collection of term dicts
    with (at minimum) the following keys:
      * 'text', the label for the term
      * 'vector', the vector for the term
      * 'score', the relevance score of the term
    Also specify the number of clusters desired; this is the number that will
    be returned unless there are not enough terms with unique texts.

    Returns a list of clusters, each of which is a list of terms.  The first
    entry in each cluster is the main label for the cluster node; subsequent
    entries are the other available labels for the node in descending order by
    relevance score.

    This is a wrapper around the ClusterTree class; for experimentation it may
    be preferable to use that class directly.
    '''
    tree = ClusterTree.from_term_list(terms)
    return [[node.term] + node.filtered_termlist
            for node in tree.flat_cluster_list(count=max_clusters)]

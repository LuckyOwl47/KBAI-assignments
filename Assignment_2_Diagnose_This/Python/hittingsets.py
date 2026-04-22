

class HittingSetTreeNode:
    """
    A class that stores all the needed information for the tree structure.
    Each node stores S(v): the set of edge labels on the path from the root to this node,
    as well as the parent, children and lables
    """

    checkmark = 'v' # checkmark for if the node has all conflict sets hit

    def __init__(self, path_labels, parent=None, edge_label=None):
        self.path_labels = set(path_labels)  # S(v): edges from root to here
        self.parent = parent                 # the node's parent
        self.edge_label = edge_label         # label on the edge coming in
        self.label = None                    # label of current node
        self.children = []                   # the node's children

    def is_checkmark(self):
        return self.label == HittingSetTreeNode.checkmark


def run_hitting_set_algorithm(conflict_sets, heuristic=None):
    """
    Algorithm that handles conflict sets to hitting sets

    Args:
        conflict_sets (list of lists): list of conflict sets
    
    Returns: 
        tuple: the hitting sets and minimal hitting sets as list of lists
    """
    if heuristic is None:
        heuristic = select_conflict_set

    conflict_sets = [set(cs) for cs in conflict_sets] # Comprehension that makes the conflict sets into actual sets instead of lists

    tree_root = build_hitting_set_tree(conflict_sets, heuristic)
    hitting_sets = collect_hitting_sets(tree_root)
    minimal_hitting_sets = filter_minimal(hitting_sets)

    return (
        [sorted(hs) for hs in hitting_sets],
        [sorted(hs) for hs in minimal_hitting_sets],
    )


def build_hitting_set_tree(conflict_sets, heuristic=None):
    """
    Constructs the hitting set tree iteratively with a stack like in the slidws

    Args: 
        conflict_sets (list of sets): list of conflict sets as sets of component names
        heuristic (set to none as default) 
    
    Returns: 
        the root node of the constructed tree
    """

    if heuristic is None:
        heuristic = select_conflict_set

    root = HittingSetTreeNode(path_labels=set())  # Calls HittingSetTreeNode with an ampty set to start with an empty accusations list

    stack = []
    stack.append(root)

    while stack:
        node = stack.pop()  # Step 2.1: remove the top node from the stack
        # choose the node's label
        unhit = select_conflict_set(node.path_labels, conflict_sets) # Either none if there is a hit, or returns the conflict set
        if unhit is None:
            node.label = HittingSetTreeNode.checkmark   
            continue  # leaf: no children to create
        node.label = unhit

        # Step 2.2: for every element in the label, create a child node
        # connected by an edge labelled u, and push it on the stack.
        # The baseline heuristic branches in lexicographic order.
        for component in sorted(unhit):
            child_path = node.path_labels | {component}
            child = HittingSetTreeNode(
                path_labels=child_path,
                parent=node,
                edge_label=component,
            )
            node.children.append(child)
            stack.append(child)

    # Step 3: return the tree.
    return root


def select_conflict_set(path_labels, conflict_sets):
    """
    Baseline heuristic for step 2.1: return the first conflict set (in the
    order given as input) that is not yet hit by `path_labels`, or None if
    every conflict set is already hit.
    """
    for cs in conflict_sets:
        if path_labels.isdisjoint(cs):  # if there's an overlap between the conflict set and the accusation list, return the conflict set
            return cs
    return None # Or else return None


def collect_hitting_sets(root):
    """
    Read hitting sets off the tree. By the definition of the algorithm, the
    edge labels on the path from the root to any checkmark leaf form a
    hitting set. We traverse the tree and collect the path_labels of every
    checkmark node.

    :param root: root node of the hitting set tree
    :return: list of hitting sets, each as a set of component names
    """
    hitting_sets = []

    stack = []
    stack.append(root)
    while stack:
        node = stack.pop()
        if node.is_checkmark():
            hitting_sets.append(set(node.path_labels))
        for child in node.children:
            stack.append(child)

    return hitting_sets


def filter_minimal(hitting_sets):
    """
    Keep only the subset-minimal hitting sets: those for which no other
    hitting set in the collection is a strict subset. Duplicate hitting sets
    (produced when different branching orders reach the same set) are
    collapsed in the process.
    """
    unique = {frozenset(hs) for hs in hitting_sets}
    minimal = []
    for hs in unique:
        if not any(other < hs for other in unique):
            minimal.append(set(hs))
    return minimal

def select_conflict_set_most_shared(path_labels, conflict_sets):
    '''
    This is our (attempt at) an improved heuristic. It selects the conflict set that shares the most 
    components with other conflict sets. 

    input: 
    - path_labels: the set of components already accused on the path to the current node
    - conflict_sets: the list of all conflict sets as sets of component names

    output:
    - the conflict set that shares the most components with other conflict sets, 
        or None if all conflict sets are already hit
    '''
    unresolved = [cs for cs in conflict_sets if path_labels.isdisjoint(cs)]
    if not unresolved:
        return None

    # Component -> number of unresolved conflict sets containing it.
    frequency = {}
    for cs in unresolved:
        for component in cs:
            frequency[component] = frequency.get(component, 0) + 1

    # score(CS) counts shared occurrences only, i.e. we subtract the 1 that
    # comes from CS itself for each of its elements.
    best_cs = unresolved[0]
    best_score = -1
    for cs in unresolved:
        score = sum(frequency[c] - 1 for c in cs)
        if score > best_score:
            best_score = score
            best_cs = cs
    return best_cs
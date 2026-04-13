class HSTree:
    def __init__(self, conflict_sets, heuristic):
        self.label_set = set()
        self.children = {}
        self.status = "open"
        self.conflict_used = None
        # build immediately
        self._expand(conflict_sets, heuristic, closed_ticks=[])

    def _expand(self, conflict_sets, heuristic, closed_ticks):
        # reuse rule: if some already-closed is a subset of me, prune
        for prior in closed_ticks:
            if prior <= self.label_set:
                self.status = "reused"
                return

        unresolved = [cs for cs in conflict_sets
                      if not (set(cs) & self.label_set)]
        if not unresolved:
            self.status = "closed-tick"
            closed_ticks.append(self.label_set)
            return

        chosen = heuristic(unresolved)
        self.conflict_used = chosen
        for component in chosen: # recursive loop for each component in the chosen conflict set
            child = HSTree.__new__(HSTree)   # avoid re-running __init__
            child.label_set = self.label_set | {component}
            child.children = {}
            child.status = "open"
            child.conflict_used = None
            self.children[component] = child
            child._expand(conflict_sets, heuristic, closed_ticks) # recursive expansion


        for component in chosen:
            child = HSTree(label_set=self.label_set | {component})
            self.children[component] = child
            child._expand(conflict_sets, heuristic, closed_ticks) # recursive expansion














def run_hitting_set_algorithm(conflict_sets):
    """
    Algorithm that handles the entire process from conflict sets to hitting sets

    :param conflict_sets: list of conflict sets as list
    :return: the hitting sets and minimal hitting sets as list of lists
    """
    return None, None

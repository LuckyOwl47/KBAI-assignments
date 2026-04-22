"""
Benchmark the heuristics in hittingsets.py across all seven circuits.

Reports per (circuit, heuristic):
    - number of nodes in the HS-tree
    - number of hitting sets found
    - number of minimal hitting sets
    - wall-clock runtime of the tree construction (averaged over N runs)

The minimal hitting sets must agree across heuristics for every circuit:
the heuristic only changes HOW the tree is built, not the set of minimal
hitting sets that fall out of it. The benchmark asserts this.

    Author: Jacob Wamon
"""
from z3 import *
import time
from os.path import join


from conflictsets import ConflictSetRetriever
from hittingsets import (
    build_hitting_set_tree,
    collect_hitting_sets,
    filter_minimal,
    select_conflict_set,
    select_conflict_set_most_shared,
)

heuristics = [
    ("first-unhit (baseline)", select_conflict_set),
    ("most-shared",            select_conflict_set_most_shared),
]

circuits = [f"circuit{i}.txt" for i in range(1, 8)]

repeats = 50  # average timing over this many runs per (circuit, heuristic)


def count_nodes(root):
    """Count all nodes in a built HS-tree by walking it."""
    count = 0
    stack = [root]
    while stack:
        node = stack.pop()
        count += 1
        for child in node.children:
            stack.append(child)
    return count


def run_once(conflict_sets_raw, heuristic):
    """Build the tree once with the given heuristic and return stats."""
    conflict_sets = [set(cs) for cs in conflict_sets_raw]

    start = time.perf_counter()
    root = build_hitting_set_tree(conflict_sets, heuristic)
    elapsed = time.perf_counter() - start

    node_count = count_nodes(root)
    hitting_sets = collect_hitting_sets(root)
    minimal = filter_minimal(hitting_sets)

    minimal_frozen = {frozenset(hs) for hs in minimal}
    return node_count, len(hitting_sets), minimal_frozen, elapsed


def benchmark_circuit(document):
    csr = ConflictSetRetriever(join("circuits", document))
    conflict_sets = csr.retrieve_conflict_sets()

    print(f"\n--- {document} ---")
    print(f"conflict sets ({len(conflict_sets)}): {conflict_sets}")

    if not conflict_sets:
        print("  (no faults — skipping)")
        return

    reference_minimal = None
    for name, heuristic in heuristics:
        total = 0.0
        node_count = hs_count = 0
        minimal = None
        for _ in range(repeats): # average over multiple runs to get a more stable timing measurement
            node_count, hs_count, minimal, elapsed = run_once(
                conflict_sets, heuristic
            )
            total += elapsed
        avg_ms = (total / repeats) * 1000

        if reference_minimal is None:
            reference_minimal = minimal
        else:
            assert minimal == reference_minimal, (
                f"{name} produced different minimal hitting sets than the "
                f"baseline on {document}"
            )

        print(
            f"  {name:25s}  nodes={node_count:4d}  "
            f"hitting_sets={hs_count:3d}  "
            f"minimal={len(minimal):2d}  "
            f"time={avg_ms:7.3f} ms"
        )


def main():
    for doc in circuits:
        benchmark_circuit(doc)


if __name__ == "__main__":
    main()
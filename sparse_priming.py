import re
from typing import List

def create_conjunctions(cluster: List) -> List[List[int]]:
    """
    Create conjunctions for a cluster of rollups.

    Args:
        cluster: A list of rollup objects

    Returns:
        A sparse representation of conjunctions

    Raises:
        ValueError: If cluster is empty
    """
    if not cluster:
        raise ValueError("Cluster cannot be empty")

    # Convert cluster roll-ups to a list of strings
    rollups = [str(rollup) for rollup in cluster]

    # Extract individual words and remove stop words
    words = set()
    for rollup in rollups:
        rollup_words = re.findall(r'\w+', rollup.lower())
        words.update(rollup_words)
    stop_words = {"a", "an", "the", "and", "or", "not", "of", "in", "on", "at", "to", "from"}
    words = words.difference(stop_words)

    # Create conjunctions
    conjunctions = []
    for word in words:
        conjunction = []
        for rollup in rollups:
            if word in rollup.lower():
                conjunction.append(1)
            else:
                conjunction.append(0)
        conjunctions.append(conjunction)

    # Convert conjunctions to a sparse representation
    sparse_rep = []
    for conjunction in conjunctions:
        if sum(conjunction) > 1:
            sparse_rep.append(conjunction)

    return sparse_rep
